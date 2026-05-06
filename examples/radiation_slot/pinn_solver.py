"""DeepXDE PINN solver for the 2D Helmholtz radiation-slot problem.

Outputs y = (u, v) ≈ (Re Ez, Im Ez). Real and imaginary parts share the network
but are decoded by component=0/1 in the BCs.

Non-dimensionalization
----------------------
The training problem is solved in scaled coordinates ``ξ = k0 · x`` to keep
all PDE/BC residual magnitudes near O(1). With ``ξ`` substituted in,

    ∇²_x Ez + k0² ε_r Ez = 0           →    ∇̃² Ez + ε_r Ez = 0
    ∂Ez/∂n + j k0 Ez     = 0           →    ∂Ez/∂ñ + j Ez = 0
    ∂Ez/∂x − j k0 Ez = -2j k0 sin(πy/b) →  ∂Ez/∂ξ_x − j Ez = -2j sin(π η_y / η_b)

where η = k0 · y, η_b = k0 · b. All k0 factors collapse out, so the BC RHS
goes from ~6.3 (in physical units, where it dominated the loss) to ~2.

The user-facing API still consumes/produces *physical* coordinates and Ez:
``predict_on_grid(X, Y, mask)`` rescales the grid by k0 internally before
calling the network.

The default backend on the dev machine is PaddlePaddle. BC value functions are
written so that ``x`` is a numpy array (boundary coords) and ``y`` is a backend
tensor — see the comment on ``port_in_im`` for details.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np

import deepxde as dde  # noqa: E402

from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry


@dataclass
class PINNSolver:
    geometry: RadiationSlotGeometry
    frequency_ghz: float
    num_domain: int = 8000
    num_boundary: int = 800
    num_test: int = 40000
    hidden_layers: tuple = (256, 256, 256, 256)
    activation: str = "sin"
    initializer: str = "Glorot uniform"
    model: object = field(default=None, repr=False)
    geom: object = field(default=None, repr=False)

    def __post_init__(self):
        self.k0 = free_space_wavenumber(
            self.frequency_ghz, eps_r=self.geometry.medium_epsilon
        )
        # Build the *non-dimensional* geometry: every linear length is multiplied
        # by k0. We do this by constructing a sister RadiationSlotGeometry whose
        # numeric fields are scaled. The boundary_marker / build_dde_geometry
        # methods are reused unchanged because they treat their inputs as plain
        # numbers.
        g = self.geometry
        k0 = self.k0
        self.geometry_nd = RadiationSlotGeometry(
            waveguide_width=g.waveguide_width * k0,
            waveguide_height=g.waveguide_height * k0,
            slot_length=g.slot_length * k0,
            slot_width=g.slot_width * k0,
            slot_position=g.slot_position,            # dimensionless ratio
            buffer_height=g.buffer_height * k0,
            medium_epsilon=g.medium_epsilon,
            medium_mu=g.medium_mu,
        )
        self.geom = self.geometry_nd.build_dde_geometry()

    # --------------------------------------------------------------- PDE
    def pde(self, x, y):
        """∇̃²u + ε_r u = 0 and ∇̃²v + ε_r v = 0 in scaled coords ξ = k0 x.

        After the substitution the k0 factor is absorbed into the coordinates,
        so the PDE no longer carries the large k0² coefficient.
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

    # --------------------------------------------------------------- BCs
    def _build_bcs(self):
        """PEC + port_in (TE10 + Mur) + port_out (Mur) + radiation (Mur) BCs.

        In scaled coordinates Mur on outward normal n̂̃ becomes
            ∂Ez/∂ñ + j Ez = 0
        Decomposed (Ez = u + j v):
            ∂u/∂ñ =  v
            ∂v/∂ñ = -u

        port_in (outward normal n̂̃ = -ξ̂) carries the TE10 incident wave:
            ∂Ez/∂ξ_x − j Ez = -2 j sin(π η_y / η_b)
        Switching to outward-normal form (∂/∂ñ = -∂/∂ξ_x) and decomposing:
            ∂u/∂ñ = -v
            ∂v/∂ñ =  u + 2 sin(π η_y / η_b)
        """
        g_nd = self.geometry_nd
        b_nd = g_nd.waveguide_height                # = k0 · b (already scaled)
        pi_over_b_nd = float(np.pi / b_nd)

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

        # PEC
        pec_re = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=0)
        pec_im = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=1)

        # port_in: TE10 + Mur (outward normal -x̂̃). All k0 factors are gone.
        # NOTE: dde.icbc.RobinBC calls value_func(X, outputs) where X is a NUMPY
        # array (boundary coordinates) and outputs is a backend TENSOR. So we
        # use np.sin on x[:, 1:2] (numpy) and let paddle/torch broadcast-add it
        # to the tensor expression involving y.
        def port_in_re(x, y):
            return -y[:, 1:2]

        def port_in_im(x, y):
            return y[:, 0:1] + 2.0 * np.sin(pi_over_b_nd * x[:, 1:2])

        port_in_bc_re = dde.icbc.RobinBC(self.geom, port_in_re, on_port_in, component=0)
        port_in_bc_im = dde.icbc.RobinBC(self.geom, port_in_im, on_port_in, component=1)

        # port_out + radiation: ∂u/∂ñ =  v, ∂v/∂ñ = -u
        def mur_re(x, y):
            return y[:, 1:2]

        def mur_im(x, y):
            return -y[:, 0:1]

        port_out_re = dde.icbc.RobinBC(self.geom, mur_re, on_port_out, component=0)
        port_out_im = dde.icbc.RobinBC(self.geom, mur_im, on_port_out, component=1)
        rad_re = dde.icbc.RobinBC(self.geom, mur_re, on_radiation, component=0)
        rad_im = dde.icbc.RobinBC(self.geom, mur_im, on_radiation, component=1)

        return [
            pec_re, pec_im,
            port_in_bc_re, port_in_bc_im,
            port_out_re, port_out_im,
            rad_re, rad_im,
        ]

    # --------------------------------------------------------------- training
    def train(
        self,
        iterations: int = 15000,
        lr: float = 1e-3,
        restore_from: str | None = None,
        save_path: str | None = None,
    ):
        bcs = self._build_bcs()
        # All residuals are now O(1) thanks to non-dimensionalization. The TE10
        # source on port_in is the *only* term carrying a non-zero RHS — without
        # heavy weighting on it, the network falls into the trivial Ez ≈ 0 local
        # minimum (every other residual term is satisfied by zero). Weighting
        # port_in 100x forces the network to learn the correct boundary value,
        # which then "seeds" the wave that has to satisfy the PDE everywhere.
        loss_weights = (
            1.0, 1.0,            # PDE re/im
            10.0, 10.0,          # PEC re/im
            100.0, 100.0,        # port_in re/im (carries the source — must dominate)
            10.0, 10.0,          # port_out Mur re/im
            10.0, 10.0,          # radiation Mur re/im
        )

        data = dde.data.PDE(
            self.geom,
            self.pde,
            bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_test=self.num_test,
        )
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

    # --------------------------------------------------------------- L-BFGS fine-tune
    def finetune_lbfgs(self, max_iter: int = 5000):
        """Switch optimizer to L-BFGS and run until convergence."""
        dde.optimizers.config.set_LBFGS_options(maxiter=max_iter)
        self.model.compile("L-BFGS")
        return self.model.train()

    # --------------------------------------------------------------- prediction
    def predict_on_grid(self, X, Y, mask) -> np.ndarray:
        """Predict Ez on a *physical-coordinate* Cartesian grid.

        Internally rescales (X, Y) → (k0 X, k0 Y) before calling the network,
        so the user never sees the non-dimensionalization. Outside-mask cells
        are set to NaN.
        """
        k0 = self.k0
        pts = np.column_stack([(X * k0).ravel(), (Y * k0).ravel()]).astype(np.float32)
        y = self.model.predict(pts)
        u = y[:, 0].reshape(X.shape).astype(np.float64)
        v = y[:, 1].reshape(X.shape).astype(np.float64)
        Ez = (u + 1j * v).astype(np.complex128)
        Ez[~mask] = np.nan
        return Ez
