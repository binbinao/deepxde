"""Optimized DeepXDE PINN solver for the 2D Helmholtz radiation-slot problem.

Two additions over the baseline ``pinn_solver.PINNSolver`` that together aim to
escape the trivial-boundary-layer local minimum of the vanilla formulation:

1. **Scattered-field decomposition.**
   Total field = incident + scattered, Ez = Ez_inc + Ez_sc. The incident TE10
   traveling wave is written analytically:

       Ez_inc(ξ, η) = sin(π η / η_b) · exp(-j k̃_x ξ_x)

   where k̃_x = sqrt(1 - (π/η_b)²) is the dimensionless guided propagation
   constant. The network learns only the scattered field Ez_sc.

   After substitution the BCs transform as follows:
     PEC:        Ez_sc = 0                              (unchanged, since Ez_inc=0 on top/bottom walls)
     port_in:    ∂Ez_sc/∂ξ_x - j Ez_sc = 0              (incident absorbs the -2j RHS)
     port_out:   ∂Ez_sc/∂ξ_x + j Ez_sc = 0              (incident already satisfies Mur exactly)
     radiation:  ∂Ez_sc/∂ñ + j Ez_sc = -(∂Ez_inc/∂ñ + j Ez_inc)   (NEW non-zero RHS on
                                                          the buffer faces — this is
                                                          what forces Ez_sc to be non-trivial)

   The PDE residual is unchanged (linear equation, and Ez_inc is an exact
   solution inside the waveguide region). Now Ez_sc = 0 violates *only* the
   radiation BC, so the optimizer cannot settle for it.

2. **Fourier-feature input embedding.**
   Instead of feeding raw (ξ, η) into the FNN, we first pass them through

       φ(ξ, η) = [sin(m ξ), cos(m ξ), sin(m η), cos(m η)]   for m = 1, 2, ..., M

   giving the network a 4M-dimensional input with built-in oscillatory basis
   functions. See Tancik et al. 2020, NeurIPS.

Both additions are wired inside a single ``net.apply_feature_transform`` call
for the Fourier embedding, and the scattered decomposition lives entirely
inside the ``pde`` and BC value functions (so ``predict_on_grid`` still
returns the physical total field Ez).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np

import deepxde as dde  # noqa: E402

from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry


# -----------------------------------------------------------------------------
# Backend-agnostic tensor primitives for the PDE/BC value functions.
#
# In deepxde, the ``pde`` function receives backend tensors for both x and y,
# while ``dde.icbc.RobinBC`` calls value(X, outputs) with X as a numpy array
# and outputs as a backend tensor. We need helpers that dispatch to the right
# sin/cos/exp depending on the argument type.
# -----------------------------------------------------------------------------

def _bk_sin(x):
    """Backend-aware sin: numpy.ndarray → np.sin, backend tensor → backend.sin."""
    if isinstance(x, np.ndarray):
        return np.sin(x)
    # Dispatch per active backend
    name = dde.backend.backend_name
    if name == "paddle":
        import paddle
        return paddle.sin(x)
    if name == "pytorch":
        import torch
        return torch.sin(x)
    if name in ("tensorflow", "tensorflow.compat.v1"):
        import tensorflow as tf
        return tf.sin(x)
    if name == "jax":
        import jax.numpy as jnp
        return jnp.sin(x)
    raise RuntimeError(f"unsupported backend: {name}")


def _bk_cos(x):
    if isinstance(x, np.ndarray):
        return np.cos(x)
    name = dde.backend.backend_name
    if name == "paddle":
        import paddle
        return paddle.cos(x)
    if name == "pytorch":
        import torch
        return torch.cos(x)
    if name in ("tensorflow", "tensorflow.compat.v1"):
        import tensorflow as tf
        return tf.cos(x)
    if name == "jax":
        import jax.numpy as jnp
        return jnp.cos(x)
    raise RuntimeError(f"unsupported backend: {name}")


def _bk_concat(tensors, axis=-1):
    if all(isinstance(t, np.ndarray) for t in tensors):
        return np.concatenate(tensors, axis=axis)
    name = dde.backend.backend_name
    if name == "paddle":
        import paddle
        return paddle.concat(list(tensors), axis=axis)
    if name == "pytorch":
        import torch
        return torch.cat(list(tensors), dim=axis)
    if name in ("tensorflow", "tensorflow.compat.v1"):
        import tensorflow as tf
        return tf.concat(list(tensors), axis=axis)
    if name == "jax":
        import jax.numpy as jnp
        return jnp.concatenate(list(tensors), axis=axis)
    raise RuntimeError(f"unsupported backend: {name}")


# -----------------------------------------------------------------------------
# Incident-field analytical helpers (operate on both numpy and backend tensors)
# -----------------------------------------------------------------------------

def _incident_uv(x_dm, pi_over_b, kx_tilde):
    """Return (u_inc, v_inc) = (Re, Im) of Ez_inc at scaled coords x_dm = (ξ, η).

    Ez_inc = sin(π η / η_b) · exp(-j k̃_x ξ_x)
           = sin(π η / η_b) · (cos(k̃_x ξ_x) - j sin(k̃_x ξ_x))
    So:
        u_inc = sin(π η / η_b) · cos(k̃_x ξ_x)
        v_inc = -sin(π η / η_b) · sin(k̃_x ξ_x)
    """
    xi = x_dm[:, 0:1]
    eta = x_dm[:, 1:2]
    sy = _bk_sin(pi_over_b * eta)
    u_inc = sy * _bk_cos(kx_tilde * xi)
    v_inc = -sy * _bk_sin(kx_tilde * xi)
    return u_inc, v_inc


# -----------------------------------------------------------------------------
# Fourier-feature input transform
# -----------------------------------------------------------------------------

def make_fourier_transform(n_frequencies: int = 4, scale: float = 1.0):
    """Build a DeepXDE feature transform that maps (ξ, η) to a 4·n_frequencies
    vector of [sin(m·ξ), cos(m·ξ), sin(m·η), cos(m·η)] for m = 1..n_frequencies.

    Intended for use with ``net.apply_feature_transform``. Returns a callable
    that accepts a backend tensor of shape (N, 2) and returns shape
    (N, 4·n_frequencies).
    """
    # m = scale, 2*scale, ..., n*scale
    modes = np.array(
        [(i + 1) * scale for i in range(n_frequencies)], dtype=np.float32
    )  # shape (n_frequencies,)

    def transform(x):
        # x: (N, 2) backend tensor. Build [sin(m·ξ), cos(m·ξ), sin(m·η), cos(m·η)]
        xi = x[:, 0:1]
        eta = x[:, 1:2]
        feats = []
        for m in modes.tolist():
            feats.append(_bk_sin(m * xi))
            feats.append(_bk_cos(m * xi))
            feats.append(_bk_sin(m * eta))
            feats.append(_bk_cos(m * eta))
        return _bk_concat(feats, axis=-1)

    transform.output_dim = 4 * n_frequencies
    return transform


# -----------------------------------------------------------------------------
# Optimized PINN solver
# -----------------------------------------------------------------------------

@dataclass
class OptimizedPINNSolver:
    geometry: RadiationSlotGeometry
    frequency_ghz: float
    num_domain: int = 8000
    num_boundary: int = 800
    num_test: int = 5000
    hidden_layers: tuple = (256, 256, 256, 256)
    activation: str = "sin"
    initializer: str = "Glorot uniform"
    # Fourier-feature settings (None = disabled, use raw (ξ, η))
    fourier_frequencies: int | None = 6
    fourier_scale: float = 1.0
    # Scattered-field decomposition toggle
    scattered_field: bool = True

    model: object = field(default=None, repr=False)
    geom: object = field(default=None, repr=False)

    def __post_init__(self):
        self.k0 = free_space_wavenumber(
            self.frequency_ghz, eps_r=self.geometry.medium_epsilon
        )
        # Non-dimensional sister geometry (every linear length × k0)
        g = self.geometry
        k0 = self.k0
        self.geometry_nd = RadiationSlotGeometry(
            waveguide_width=g.waveguide_width * k0,
            waveguide_height=g.waveguide_height * k0,
            slot_length=g.slot_length * k0,
            slot_width=g.slot_width * k0,
            slot_position=g.slot_position,
            buffer_height=g.buffer_height * k0,
            medium_epsilon=g.medium_epsilon,
            medium_mu=g.medium_mu,
        )
        self.geom = self.geometry_nd.build_dde_geometry()

        # Dimensionless helpers
        self.b_nd = self.geometry_nd.waveguide_height           # = k0 b
        self.pi_over_b = float(np.pi / self.b_nd)
        kx2 = 1.0 - (np.pi / self.b_nd) ** 2
        if kx2 <= 0:
            raise ValueError(
                f"TE10 is below cutoff: b_nd={self.b_nd}, need b_nd > π. "
                f"Increase waveguide_height or frequency."
            )
        self.kx_tilde = float(np.sqrt(kx2))

    # ----------------------------------------------------------------- PDE
    def pde(self, x, y):
        """∇̃²u + ε_r u = 0 and ∇̃²v + ε_r v = 0 in scaled coords.

        With the scattered-field decomposition enabled, y is the *scattered*
        field (u_sc, v_sc), which satisfies the same homogeneous Helmholtz
        equation as the total field because Ez_inc is itself an exact solution
        inside the waveguide region.
        """
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        eps_r = self.geometry.medium_epsilon
        return [
            u_xx + u_yy + eps_r * y[:, 0:1],
            v_xx + v_yy + eps_r * y[:, 1:2],
        ]

    # ----------------------------------------------------------------- BCs
    def _build_bcs(self):
        g_nd = self.geometry_nd
        pi_over_b = self.pi_over_b
        kx_tilde = self.kx_tilde
        scattered = self.scattered_field

        def label_of(x):
            return g_nd.boundary_marker(np.atleast_2d(x))[0]

        def on_pec(x, on_boundary):
            return on_boundary and label_of(x) == "pec"

        def on_port_in(x, on_boundary):
            return on_boundary and label_of(x) == "port_in"

        def on_port_out(x, on_boundary):
            return on_boundary and label_of(x) == "port_out"

        def on_radiation(x, on_boundary):
            return on_boundary and label_of(x) == "radiation"

        # PEC: Ez = 0 everywhere on PEC walls. The incident field satisfies this
        # on the waveguide top/bottom (sin(π·0/b) = sin(π) = 0), so Ez_sc = 0
        # there as well. Buffer vertical walls are labeled 'radiation', not PEC.
        pec_re = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=0)
        pec_im = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=1)

        # port_in: ∂Ez/∂ñ (= -∂Ez/∂ξ_x at x=0)
        #   Total field BC:  ∂Ez/∂ñ =  v + 2 sin(π η/η_b)   for im part
        #                     ∂Ez/∂ñ = -v                    for re part (no source)
        #   With scattered decomposition the incident exactly takes care of
        #   the 2 sin(...) source, so we only need the homogeneous Mur on Ez_sc:
        #     ∂u_sc/∂ñ = -v_sc
        #     ∂v_sc/∂ñ =  u_sc
        if scattered:
            def port_in_re(x, y):
                return -y[:, 1:2]

            def port_in_im(x, y):
                return y[:, 0:1]
        else:
            def port_in_re(x, y):
                return -y[:, 1:2]

            def port_in_im(x, y):
                return y[:, 0:1] + 2.0 * np.sin(pi_over_b * x[:, 1:2])

        port_in_bc_re = dde.icbc.RobinBC(self.geom, port_in_re, on_port_in, component=0)
        port_in_bc_im = dde.icbc.RobinBC(self.geom, port_in_im, on_port_in, component=1)

        # port_out and (for scattered=False) radiation: standard Mur on total field.
        # Both become the same homogeneous Mur on Ez_sc when scattered=True.
        def mur_re(x, y):
            return y[:, 1:2]

        def mur_im(x, y):
            return -y[:, 0:1]

        port_out_re = dde.icbc.RobinBC(self.geom, mur_re, on_port_out, component=0)
        port_out_im = dde.icbc.RobinBC(self.geom, mur_im, on_port_out, component=1)

        # radiation BCs depend on whether we're using scattered decomposition.
        if not scattered:
            rad_re = dde.icbc.RobinBC(self.geom, mur_re, on_radiation, component=0)
            rad_im = dde.icbc.RobinBC(self.geom, mur_im, on_radiation, component=1)
        else:
            # With scattered decomposition, on radiation boundaries we get:
            #   ∂Ez_sc/∂ñ + j Ez_sc = -(∂Ez_inc/∂ñ + j Ez_inc)
            # Decomposed:
            #   ∂u_sc/∂ñ =  v_sc - (∂u_inc/∂ñ - v_inc)    NOTE: -j·(j·v_inc) term moves sign
            # Let's rederive carefully. Write everything in terms of real & imag.
            #   (∂u/∂ñ + j ∂v/∂ñ) + j(u + j v) = (∂u/∂ñ - v) + j(∂v/∂ñ + u)
            # So the real-part Robin BC reads:
            #   ∂u/∂ñ - v = 0         →  ∂u/∂ñ =  v
            # And the imag-part:
            #   ∂v/∂ñ + u = 0         →  ∂v/∂ñ = -u
            # (matches our `mur_re`, `mur_im` above.)
            # For scattered, apply the same operator to (u_sc, v_sc) with RHS:
            #   ∂u_sc/∂ñ - v_sc = -(∂u_inc/∂ñ - v_inc)   ... (*)
            #   ∂v_sc/∂ñ + u_sc = -(∂v_inc/∂ñ + u_inc)   ... (**)
            # RHS values depend on the outward-normal direction. On the buffer
            # top wall n̂̃ = +η̂, so ∂·/∂ñ = ∂·/∂η.
            # On the buffer side walls n̂̃ = ±ξ̂_x, so ∂·/∂ñ = ±∂·/∂ξ_x.
            # We precompute a numeric normal map by querying geometry.
            def _outward_normal(x):
                """Return (N, 2) outward unit normal for each point on radiation boundary."""
                slot_lo, slot_hi = g_nd.slot_x_range()
                wg_h = g_nd.waveguide_height
                buf_top = wg_h + g_nd.buffer_height
                atol = 1e-6
                nrm = np.zeros_like(x)
                on_top = np.isclose(x[:, 1], buf_top, atol=atol)
                on_left = np.isclose(x[:, 0], slot_lo, atol=atol)
                on_right = np.isclose(x[:, 0], slot_hi, atol=atol)
                nrm[on_top, 1] = 1.0
                nrm[on_left, 0] = -1.0
                nrm[on_right, 0] = 1.0
                return nrm

            def _inc_and_dn_inc(x):
                """Compute u_inc, v_inc and their outward-normal derivatives at
                radiation boundary points x (numpy array of shape (N, 2)).

                ∂u_inc/∂ξ = -sin(π η/η_b) · k_x̃ · sin(k_x̃ ξ) · (-1) · k_x̃ ?? let's just differentiate analytically.
                u_inc = sy(η) · cos(k_x̃ ξ)
                v_inc = -sy(η) · sin(k_x̃ ξ)
                where sy(η) = sin(π η/η_b).
                ∂u_inc/∂ξ = -sy(η) · k_x̃ · sin(k_x̃ ξ)  =  k_x̃ · v_inc
                ∂u_inc/∂η = (π/η_b) · cos(π η/η_b) · cos(k_x̃ ξ)
                ∂v_inc/∂ξ = -sy(η) · k_x̃ · cos(k_x̃ ξ) = -k_x̃ · u_inc
                ∂v_inc/∂η = -(π/η_b) · cos(π η/η_b) · sin(k_x̃ ξ)
                """
                xi = x[:, 0:1]
                eta = x[:, 1:2]
                sy = np.sin(pi_over_b * eta)
                cy = np.cos(pi_over_b * eta)
                cx = np.cos(kx_tilde * xi)
                sx = np.sin(kx_tilde * xi)

                u_inc = sy * cx
                v_inc = -sy * sx
                du_dxi = kx_tilde * v_inc
                du_deta = pi_over_b * cy * cx
                dv_dxi = -kx_tilde * u_inc
                dv_deta = -pi_over_b * cy * sx
                return u_inc, v_inc, du_dxi, du_deta, dv_dxi, dv_deta

            def rad_re_value(x, y):
                # x: numpy (N, 2). y: backend tensor (N, 2) — unused here because
                # the RHS is data-driven.
                nrm = _outward_normal(x)
                u_inc, v_inc, du_dxi, du_deta, dv_dxi, dv_deta = _inc_and_dn_inc(x)
                # ∂u_inc/∂ñ = nx * du_dxi + ny * du_deta
                du_dn_inc = nrm[:, 0:1] * du_dxi + nrm[:, 1:2] * du_deta
                # RHS from (*): ∂u_sc/∂ñ = v_sc - (∂u_inc/∂ñ - v_inc)
                # In dde.RobinBC the value must equal the normal derivative.
                # Network output is (u_sc, v_sc). So:
                #   value = v_sc - (∂u_inc/∂ñ - v_inc)
                return y[:, 1:2] - (du_dn_inc - v_inc)

            def rad_im_value(x, y):
                nrm = _outward_normal(x)
                u_inc, v_inc, du_dxi, du_deta, dv_dxi, dv_deta = _inc_and_dn_inc(x)
                dv_dn_inc = nrm[:, 0:1] * dv_dxi + nrm[:, 1:2] * dv_deta
                # ∂v_sc/∂ñ = -u_sc - (∂v_inc/∂ñ + u_inc)
                return -y[:, 0:1] - (dv_dn_inc + u_inc)

            rad_re = dde.icbc.RobinBC(self.geom, rad_re_value, on_radiation, component=0)
            rad_im = dde.icbc.RobinBC(self.geom, rad_im_value, on_radiation, component=1)

        return [
            pec_re, pec_im,
            port_in_bc_re, port_in_bc_im,
            port_out_re, port_out_im,
            rad_re, rad_im,
        ]

    # ----------------------------------------------------------------- training
    def train(self, iterations: int = 15000, lr: float = 1e-3,
              restore_from: str | None = None, save_path: str | None = None):
        bcs = self._build_bcs()

        # All residuals are O(1). Emphasize the radiation BC because that's the
        # only term preventing Ez_sc = 0 collapse.
        if self.scattered_field:
            loss_weights = (
                1.0, 1.0,            # PDE re/im
                10.0, 10.0,          # PEC re/im
                10.0, 10.0,          # port_in re/im (homogeneous now)
                10.0, 10.0,          # port_out Mur re/im
                100.0, 100.0,        # radiation Mur re/im  ←  source of non-triviality
            )
        else:
            loss_weights = (
                1.0, 1.0,
                10.0, 10.0,
                100.0, 100.0,        # port_in carries the source
                10.0, 10.0,
                10.0, 10.0,
            )

        data = dde.data.PDE(
            self.geom, self.pde, bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_test=self.num_test,
        )

        # Build the network — with optional Fourier feature input embedding.
        if self.fourier_frequencies:
            fourier_tf = make_fourier_transform(
                n_frequencies=self.fourier_frequencies,
                scale=self.fourier_scale,
            )
            input_dim = fourier_tf.output_dim
            net = dde.nn.FNN(
                [input_dim] + list(self.hidden_layers) + [2],
                self.activation,
                self.initializer,
            )
            net.apply_feature_transform(fourier_tf)
        else:
            net = dde.nn.FNN(
                [2] + list(self.hidden_layers) + [2],
                self.activation,
                self.initializer,
            )

        self.model = dde.Model(data, net)
        self.model.compile("adam", lr=lr, loss_weights=loss_weights)
        if restore_from is not None:
            self.model.restore(restore_from, verbose=1)
        return self.model.train(
            iterations=iterations,
            display_every=max(1, iterations // 10),
            model_save_path=save_path,
        )

    def finetune_lbfgs(self, max_iter: int = 5000):
        dde.optimizers.config.set_LBFGS_options(maxiter=max_iter)
        self.model.compile("L-BFGS")
        return self.model.train()

    # ----------------------------------------------------------------- prediction
    def predict_on_grid(self, X, Y, mask) -> np.ndarray:
        """Predict the *total* Ez on a physical-coordinate grid.

        Internally the network outputs the scattered field; we add the
        analytical incident field before returning. If ``scattered_field`` is
        False, the network already represents Ez directly and we skip the
        incident addition.
        """
        k0 = self.k0
        pts = np.column_stack([(X * k0).ravel(), (Y * k0).ravel()]).astype(np.float32)
        y = self.model.predict(pts)
        u_sc = y[:, 0].reshape(X.shape).astype(np.float64)
        v_sc = y[:, 1].reshape(X.shape).astype(np.float64)

        if self.scattered_field:
            # Add incident inside the waveguide region only. The incident sin
            # profile analytically continues as an imaginary exponential in y
            # outside the waveguide, which is NOT the physical incident outside
            # a PEC waveguide (there the waveguide ceiling blocks the wave
            # except through the slot). So we apply incident only where
            # Y ≤ waveguide_height.
            wg_h = self.geometry.waveguide_height
            Ez_inc = np.zeros_like(X, dtype=np.complex128)
            inside_wg = Y <= wg_h + 1e-9
            sy = np.sin(np.pi * Y[inside_wg] / wg_h)
            kx_phys = self.kx_tilde * k0     # k̃_x = k_x / k0, so k_x = k̃_x·k0
            Ez_inc[inside_wg] = sy * np.exp(-1j * kx_phys * X[inside_wg])
            Ez = (u_sc + 1j * v_sc) + Ez_inc
        else:
            Ez = (u_sc + 1j * v_sc).astype(np.complex128)

        Ez[~mask] = np.nan
        return Ez
