"""DeepXDE PINN solver for the 2D Helmholtz radiation-slot problem.

Outputs y = (u, v) ≈ (Re Ez, Im Ez). Real and imaginary parts share the network
but are decoded by component=0/1 in the BCs.

The default backend on the dev machine is PaddlePaddle. The PDE residual and
all BC value functions use ``dde.backend`` primitives (sin/abs) so they remain
backend-agnostic.
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
    activation: str = "tanh"
    initializer: str = "Glorot uniform"
    model: object = field(default=None, repr=False)
    geom: object = field(default=None, repr=False)

    def __post_init__(self):
        self.k0 = free_space_wavenumber(
            self.frequency_ghz, eps_r=self.geometry.medium_epsilon
        )
        self.geom = self.geometry.build_dde_geometry()

    # --------------------------------------------------------------- PDE
    def pde(self, x, y):
        """∇²u + k² u = 0 and ∇²v + k² v = 0  (real & imag parts decoupled)."""
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        k2 = self.k0 ** 2 * self.geometry.medium_epsilon
        return [
            u_xx + u_yy + k2 * y[:, 0:1],
            v_xx + v_yy + k2 * y[:, 1:2],
        ]

    # --------------------------------------------------------------- BCs
    def _build_bcs(self):
        """PEC + port_in (TE10 + Mur) + port_out (Mur) + radiation (Mur) BCs.

        Mur on outward normal n̂:  ∂Ez/∂n + j k0 Ez = 0
        Decomposed (Ez = u + j v):
            ∂u/∂n =  k0 v
            ∂v/∂n = -k0 u

        port_in (outward normal n̂ = -x̂) carries the TE10 incident wave:
            ∂Ez/∂x − j k0 Ez = -2 j k0 sin(πy/b)
        Switching to outward-normal form (∂/∂n = -∂/∂x) and decomposing:
            ∂u/∂n = -k0 v
            ∂v/∂n =  k0 u + 2 k0 sin(πy/b)
        """
        g = self.geometry
        k0 = self.k0
        b_h = g.waveguide_height
        pi_over_b = float(np.pi / b_h)

        def label_of(x):
            return g.boundary_marker(np.atleast_2d(x))[0]

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

        # port_in: TE10 + Mur (outward normal -x̂)
        # NOTE: dde.icbc.RobinBC calls value_func(X, outputs) where X is a NUMPY
        # array (boundary coordinates) and outputs is a backend TENSOR. So we
        # use np.sin on x[:, 1:2] and let paddle/torch broadcast-add it to the
        # tensor expression involving y.
        def port_in_re(x, y):
            return -k0 * y[:, 1:2]

        def port_in_im(x, y):
            return k0 * y[:, 0:1] + 2.0 * k0 * np.sin(pi_over_b * x[:, 1:2])

        port_in_bc_re = dde.icbc.RobinBC(self.geom, port_in_re, on_port_in, component=0)
        port_in_bc_im = dde.icbc.RobinBC(self.geom, port_in_im, on_port_in, component=1)

        # port_out + radiation: ∂u/∂n = k0 v, ∂v/∂n = -k0 u
        def mur_re(x, y):
            return k0 * y[:, 1:2]

        def mur_im(x, y):
            return -k0 * y[:, 0:1]

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
        loss_weights = (1.0, 1.0) + tuple(100.0 for _ in bcs)

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
