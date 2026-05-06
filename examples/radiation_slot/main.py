"""CLI entry point for radar radiation-slot simulation.

Modes:
    single — run one frequency, full FDFD vs PINN comparison
    scan   — sweep 12-18 GHz at 0.5 GHz steps with PINN warm-starts (M6)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import yaml

from examples.radiation_slot.comparator import field_metrics, pattern_metrics
from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import (
    aperture_field, near_to_far_field, plot_field,
    plot_radiation_pattern, s_parameters,
)


DEFAULT_CONFIG = Path(__file__).parent / "conf" / "radiation_slot.yaml"
DEFAULT_OUT = Path(__file__).parent / "outputs"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "scan"], default="single")
    p.add_argument("--freq", type=float, default=15.0, help="GHz, single mode")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--iters", type=int, default=15000, help="Adam iterations")
    p.add_argument("--lbfgs", type=int, default=5000, help="L-BFGS max iterations")
    p.add_argument("--seed", type=int, default=0)
    return p


def run_single_frequency(freq_ghz: float, cfg: dict, outdir: Path,
                         iters: int = 15000, lbfgs_max: int = 5000) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    g = RadiationSlotGeometry(**cfg["geometry"])

    print(f"[1/4] FDFD reference at {freq_ghz} GHz ...")
    fdfd_out = FDFDSolver(
        g, frequency_ghz=freq_ghz, mesh_size=cfg["fdfd"]["mesh_size"]
    ).solve()

    print(f"[2/4] PINN training ({iters} Adam iters + L-BFGS) ...")
    # Lazy-import PINN solver so importing main.py does not pay DeepXDE init cost.
    from examples.radiation_slot.pinn_solver import PINNSolver
    pinn = PINNSolver(
        g, frequency_ghz=freq_ghz,
        num_domain=cfg["pinn"]["num_domain"],
        num_boundary=cfg["pinn"]["num_boundary"],
        num_test=cfg["pinn"]["num_test"],
    )
    pinn.train(iterations=iters, lr=cfg["pinn"]["adam_lr"])
    if lbfgs_max > 0:
        pinn.finetune_lbfgs(max_iter=lbfgs_max)

    print("[3/4] Predicting PINN field on FDFD grid ...")
    Ez_pinn = pinn.predict_on_grid(fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"])

    print("[4/4] Postprocess + comparator ...")
    k0 = free_space_wavenumber(freq_ghz, eps_r=g.medium_epsilon)

    metrics = field_metrics(fdfd_out["Ez"], Ez_pinn, fdfd_out["mask"])

    s11_f, s21_f = s_parameters(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g, k0)
    s11_p, s21_p = s_parameters(Ez_pinn,        fdfd_out["X"], fdfd_out["Y"], g, k0)

    x_ap, ez_ap_f = aperture_field(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g)
    _,    ez_ap_p = aperture_field(Ez_pinn,        fdfd_out["X"], fdfd_out["Y"], g)
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    pat_f = near_to_far_field(ez_ap_f, x_ap, k0, theta)
    pat_p = near_to_far_field(ez_ap_p, x_ap, k0, theta)
    pat_f_db = 20 * np.log10(np.abs(pat_f) / np.max(np.abs(pat_f)))
    pat_p_db = 20 * np.log10(np.maximum(np.abs(pat_p), 1e-12) / np.max(np.abs(pat_p)))
    pat_metrics = pattern_metrics(theta, pat_f_db, pat_p_db)

    plot_field(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / f"field_fdfd_{freq_ghz}GHz.png"),
               title=f"FDFD |Ez| @ {freq_ghz} GHz")
    plot_field(Ez_pinn, fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / f"field_pinn_{freq_ghz}GHz.png"),
               title=f"PINN |Ez| @ {freq_ghz} GHz")
    plot_radiation_pattern(theta, pat_f_db,
                           str(outdir / f"pattern_fdfd_{freq_ghz}GHz.png"),
                           title=f"FDFD pattern @ {freq_ghz} GHz")
    plot_radiation_pattern(theta, pat_p_db,
                           str(outdir / f"pattern_pinn_{freq_ghz}GHz.png"),
                           title=f"PINN pattern @ {freq_ghz} GHz")

    summary = {
        "frequency_ghz": float(freq_ghz),
        "iters_adam": int(iters),
        "iters_lbfgs_max": int(lbfgs_max),
        "field": {k: float(v) for k, v in metrics.items()},
        "pattern": {k: float(v) for k, v in pat_metrics.items()},
        "S11_fdfd_dB": float(20 * np.log10(max(abs(s11_f), 1e-12))),
        "S11_pinn_dB": float(20 * np.log10(max(abs(s11_p), 1e-12))),
        "S21_fdfd_dB": float(20 * np.log10(max(abs(s21_f), 1e-12))),
        "S21_pinn_dB": float(20 * np.log10(max(abs(s21_p), 1e-12))),
    }
    with open(outdir / f"summary_{freq_ghz}GHz.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    np.random.seed(args.seed)
    if args.mode == "single":
        run_single_frequency(args.freq, cfg, Path(args.outdir),
                             iters=args.iters, lbfgs_max=args.lbfgs)
    elif args.mode == "scan":
        from examples.radiation_slot.main_scan import run_scan
        run_scan(cfg, Path(args.outdir), iters=args.iters, lbfgs_max=args.lbfgs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
