"""Long-training run V2: num_domain=30k, fourier=8, 15k+5k.

Goal: address overfitting seen in V1 (test_loss / train_loss ≈ 16000×).
- num_domain 8000 → 30000 to match Fourier representation density
- num_test 5000 → 15000 to give a meaningful generalization signal
- num_boundary 800 → 2000 for proportional boundary coverage
- fourier_frequencies 12 → 8 to balance expressivity vs overfit risk
"""
import os
os.environ.setdefault("DDE_BACKEND", "paddle")

import time
from pathlib import Path
import numpy as np
import yaml

from examples.radiation_slot.comparator import field_metrics, pattern_metrics
from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import (
    aperture_field, near_to_far_field, plot_field, plot_radiation_pattern, s_parameters,
)
from examples.radiation_slot.pinn_solver_optimized import OptimizedPINNSolver


def main():
    outdir = Path("examples/radiation_slot/outputs/ab_compare")
    outdir.mkdir(parents=True, exist_ok=True)

    g = RadiationSlotGeometry()
    print("[1/4] FDFD reference at 15.0 GHz ...", flush=True)
    fdfd_out = FDFDSolver(g, frequency_ghz=15.0, mesh_size=0.05).solve()
    k0 = free_space_wavenumber(15.0)

    print("[2/4] PINN (fourier=8, scattered, num_domain=30k, 15k Adam + 5k L-BFGS) ...",
          flush=True)
    t0 = time.time()
    solver = OptimizedPINNSolver(
        g, frequency_ghz=15.0,
        num_domain=30000, num_boundary=2000, num_test=15000,
        fourier_frequencies=8,
        fourier_scale=1.0,
        scattered_field=True,
    )
    solver.train(iterations=15000, lr=1e-3)
    solver.finetune_lbfgs(max_iter=5000)
    runtime = time.time() - t0
    print(f"[3/4] Predicting (train runtime = {runtime:.1f}s) ...", flush=True)

    Ez_pinn = solver.predict_on_grid(fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"])
    metrics = field_metrics(fdfd_out["Ez"], Ez_pinn, fdfd_out["mask"])
    s11_p, s21_p = s_parameters(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], g, k0)
    s11_f, s21_f = s_parameters(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g, k0)

    x_ap_p, ez_ap_p = aperture_field(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], g)
    x_ap_f, ez_ap_f = aperture_field(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g)
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    pat_p = near_to_far_field(ez_ap_p, x_ap_p, k0, theta)
    pat_f = near_to_far_field(ez_ap_f, x_ap_f, k0, theta)
    pat_p_db = 20*np.log10(np.maximum(np.abs(pat_p), 1e-12)/np.max(np.abs(pat_p)))
    pat_f_db = 20*np.log10(np.abs(pat_f)/np.max(np.abs(pat_f)))
    pat_metrics = pattern_metrics(theta, pat_f_db, pat_p_db)

    plot_field(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / "field_optimized_v2.png"),
               title="Optimized PINN |Ez| @ 15 GHz (V2: f=8, dom=30k, 15k+5k)")
    plot_radiation_pattern(theta, pat_p_db, str(outdir / "pattern_optimized_v2.png"),
                           title="Optimized PINN pattern (V2)")

    summary = {
        "config": "V2: fourier=8, scattered=True, num_domain=30000, num_test=15000",
        "iters_adam": 15000,
        "iters_lbfgs": 5000,
        "runtime_s": round(runtime, 1),
        "field": {k: float(v) for k, v in metrics.items()},
        "pattern": {k: float(v) for k, v in pat_metrics.items()},
        "S11_fdfd_dB": float(20*np.log10(max(abs(s11_f), 1e-12))),
        "S11_pinn_dB": float(20*np.log10(max(abs(s11_p), 1e-12))),
        "S21_fdfd_dB": float(20*np.log10(max(abs(s21_f), 1e-12))),
        "S21_pinn_dB": float(20*np.log10(max(abs(s21_p), 1e-12))),
    }
    with open(outdir / "summary_optimized_v2.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    print("\n[4/4] Summary:", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
