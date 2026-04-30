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
        """PEC-only stub. Full Mur+TE10 BC set is added in Task 13."""
        g = self.geometry

        def on_pec(x, on_boundary):
            return on_boundary and g.boundary_marker(np.atleast_2d(x))[0] == "pec"

        return [
            dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=0),
            dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=1),
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
