"""PINN smoke tests. Accuracy is validated end-to-end via main.py."""
import os

# Backend selection — keep whatever DeepXDE has already set (defaults to paddle
# on this machine). DDE_BACKEND env var can override before pytest is launched.
os.environ.setdefault("DDE_BACKEND", os.environ.get("DDE_BACKEND", "paddle"))

import pytest

from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.pinn_solver import PINNSolver


def test_smoke_train_50_steps():
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=400, num_boundary=80, num_test=400)
    losshistory, _ = solver.train(iterations=50, lr=1e-3)
    assert len(losshistory.loss_train) >= 1


def test_build_bcs_returns_full_set():
    """PEC re/im (2) + port_in re/im (2) + port_out re/im (2) + radiation re/im (2) = 8."""
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=10, num_boundary=10, num_test=10)
    bcs = solver._build_bcs()
    assert len(bcs) == 8
    classes = {type(b).__name__ for b in bcs}
    assert "DirichletBC" in classes
    assert "RobinBC" in classes


def test_predict_on_grid_shape():
    import numpy as np
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=200, num_boundary=40, num_test=200)
    solver.train(iterations=10, lr=1e-3)
    X, Y, mask = g.fdfd_grid(0.2)
    Ez = solver.predict_on_grid(X, Y, mask)
    assert Ez.shape == X.shape
    assert np.iscomplexobj(Ez)


def test_lbfgs_finetune_runs():
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=200, num_boundary=40, num_test=200)
    solver.train(iterations=10, lr=1e-3)
    solver.finetune_lbfgs(max_iter=20)
    assert solver.model is not None
