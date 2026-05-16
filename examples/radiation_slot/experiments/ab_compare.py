"""A/B comparison: baseline vs scattered-field vs scattered+Fourier.

Usage:
    cd /data/deepxde
    python3 -m examples.radiation_slot.experiments.ab_compare --iters 3000 --lbfgs 1000
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from examples.radiation_slot.comparator import field_metrics, pattern_metrics
from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import (
    aperture_field, near_to_far_field, plot_field, s_parameters,
)


def run_config(name: str, freq_ghz: float, iters: int, lbfgs: int,
               fourier_freqs: int | None, scattered: bool,
               fdfd_out: dict, g: RadiationSlotGeometry,
               outdir: Path) -> dict:
    """Train one configuration and return metrics + runtime."""
    print(f"\n{'='*70}\n[ {name} ]  fourier_freqs={fourier_freqs}  scattered={scattered}\n{'='*70}")
    from examples.radiation_slot.pinn_solver_optimized import OptimizedPINNSolver

    t0 = time.time()
    solver = OptimizedPINNSolver(
        g, frequency_ghz=freq_ghz,
        num_domain=8000, num_boundary=800, num_test=5000,
        fourier_frequencies=fourier_freqs,
        scattered_field=scattered,
    )
    losshistory, _ = solver.train(iterations=iters, lr=1e-3)
    if lbfgs > 0:
        solver.finetune_lbfgs(max_iter=lbfgs)
    t_train = time.time() - t0

    Ez_pinn = solver.predict_on_grid(fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"])
    k0 = free_space_wavenumber(freq_ghz, eps_r=g.medium_epsilon)

    metrics = field_metrics(fdfd_out["Ez"], Ez_pinn, fdfd_out["mask"])
    s11_p, s21_p = s_parameters(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], g, k0)

    x_ap, ez_ap_p = aperture_field(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], g)
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    pat_p = near_to_far_field(ez_ap_p, x_ap, k0, theta)
    pat_p_db = 20 * np.log10(np.maximum(np.abs(pat_p), 1e-12) / np.max(np.abs(pat_p)))

    # Reference pattern from FDFD
    x_ap_f, ez_ap_f = aperture_field(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g)
    pat_f = near_to_far_field(ez_ap_f, x_ap_f, k0, theta)
    pat_f_db = 20 * np.log10(np.abs(pat_f) / np.max(np.abs(pat_f)))

    pat_metrics = pattern_metrics(theta, pat_f_db, pat_p_db)

    # Save the field plot for this config
    plot_field(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / f"field_{name}.png"),
               title=f"{name} |Ez| @ {freq_ghz} GHz")

    return {
        "name": name,
        "fourier_freqs": fourier_freqs,
        "scattered": scattered,
        "iters_adam": iters,
        "iters_lbfgs": lbfgs,
        "runtime_s": round(t_train, 1),
        "field_l2_relative": float(metrics["l2_relative"]),
        "field_linf": float(metrics["linf"]),
        "field_corrcoef": float(metrics["corrcoef"]),
        "pattern_main_lobe_deg_diff": float(pat_metrics["main_lobe_deg_diff"]),
        "s11_pinn_dB": float(20 * np.log10(max(abs(s11_p), 1e-12))),
        "s21_pinn_dB": float(20 * np.log10(max(abs(s21_p), 1e-12))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=15.0)
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--lbfgs", type=int, default=1000)
    p.add_argument("--fourier-freqs", type=int, default=6)
    p.add_argument("--outdir", default="examples/radiation_slot/outputs/ab_compare")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    g = RadiationSlotGeometry()
    print(f"[setup] Running FDFD reference at {args.freq} GHz ...")
    fdfd_out = FDFDSolver(g, frequency_ghz=args.freq, mesh_size=0.05).solve()

    configs = [
        ("baseline",        None,               False),
        ("scattered_only",  None,               True),
        ("full_optimized",  args.fourier_freqs, True),
    ]

    results = []
    for name, ff, sc in configs:
        result = run_config(name, args.freq, args.iters, args.lbfgs,
                            ff, sc, fdfd_out, g, outdir)
        results.append(result)
        print(f"\n[{name}] → L2={result['field_l2_relative']:.3f}  "
              f"corrcoef={result['field_corrcoef']:+.3f}  "
              f"main-lobe Δ={result['pattern_main_lobe_deg_diff']:+.1f}°  "
              f"runtime={result['runtime_s']}s")

    with open(outdir / "ab_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== Summary (all three configs, same {args.iters}+{args.lbfgs} budget) ===")
    print(f"{'config':<20} {'L2':>8} {'corrcoef':>10} {'main-lobe Δ°':>15} {'runtime':>10}")
    for r in results:
        print(f"{r['name']:<20} {r['field_l2_relative']:>8.3f} {r['field_corrcoef']:>+10.3f} "
              f"{r['pattern_main_lobe_deg_diff']:>+15.1f} {r['runtime_s']:>8.1f}s")


if __name__ == "__main__":
    main()
