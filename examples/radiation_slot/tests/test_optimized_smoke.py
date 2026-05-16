"""Smoke tests for OptimizedPINNSolver."""
import os

os.environ.setdefault("DDE_BACKEND", os.environ.get("DDE_BACKEND", "paddle"))

import numpy as np

from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.pinn_solver_optimized import (
    OptimizedPINNSolver,
    make_fourier_transform,
)


def test_fourier_transform_output_dim():
    tf = make_fourier_transform(n_frequencies=3, scale=1.0)
    assert tf.output_dim == 12


def test_optimized_smoke_scattered_fourier():
    g = RadiationSlotGeometry()
    solver = OptimizedPINNSolver(
        g, frequency_ghz=15.0,
        num_domain=400, num_boundary=80, num_test=400,
        fourier_frequencies=3,
        scattered_field=True,
    )
    losshistory, _ = solver.train(iterations=50, lr=1e-3)
    assert len(losshistory.loss_train) >= 1
    # Prediction must return a complex field of the right shape
    X, Y, mask = g.fdfd_grid(0.2)
    Ez = solver.predict_on_grid(X, Y, mask)
    assert Ez.shape == X.shape
    assert np.iscomplexobj(Ez)


def test_optimized_no_scattered_no_fourier_smoke():
    """Baseline-equivalent mode: should also run without errors."""
    g = RadiationSlotGeometry()
    solver = OptimizedPINNSolver(
        g, frequency_ghz=15.0,
        num_domain=400, num_boundary=80, num_test=400,
        fourier_frequencies=None,
        scattered_field=False,
    )
    solver.train(iterations=50, lr=1e-3)
    X, Y, mask = g.fdfd_grid(0.2)
    Ez = solver.predict_on_grid(X, Y, mask)
    assert Ez.shape == X.shape
