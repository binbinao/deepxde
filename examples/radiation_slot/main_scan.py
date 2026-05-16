"""Frequency-sweep entry point for the radiation-slot simulation.

Per the project plan (Task 16), this module sweeps the FDFD reference solver
across the Ku band (12-18 GHz at 0.5 GHz steps) and optionally trains the
optimized PINN at a single user-selected frequency for a side-by-side
single-frequency comparison.

Why FDFD-only sweep + PINN single point:
    The OptimizedPINNSolver V2 takes ~98 minutes per frequency on this
    machine; a full 13-point PINN sweep would take ~21 hours. FDFD scans
    each frequency in ~3 seconds (sparse LU on a complex 81x71 grid) so
    the sweep runs in under a minute. The PINN is therefore demonstrated
    at a single frequency only, and the FDFD-side S-parameter / pattern
    curves give the full Ku-band picture.

Outputs (under ``examples/radiation_slot/outputs/scan/``):
    s_parameters_scan.png        — |S11| / |S21| dB vs frequency
    pattern_overlay.png          — superimposed E-plane patterns at each freq
    field_pinn_<freq>GHz.png     — PINN |Ez| at the chosen freq
    field_fdfd_<freq>GHz.png     — FDFD |Ez| at the chosen freq
    pattern_pinn_<freq>GHz.png   — PINN radiation pattern at the chosen freq
    pattern_fdfd_<freq>GHz.png   — FDFD radiation pattern at the chosen freq
    scan_summary.yaml            — per-frequency S-parameters + chosen-PINN-freq metrics

Usage:
    python3 -m examples.radiation_slot.main_scan
    python3 -m examples.radiation_slot.main_scan --pinn-freq 15.0 --pinn-iters 15000
    python3 -m examples.radiation_slot.main_scan --no-pinn        # FDFD-only
"""
from __future__ import annotations
import argparse
import os
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


DEFAULT_CONFIG = Path(__file__).parent / "conf" / "radiation_slot.yaml"
DEFAULT_OUT = Path(__file__).parent / "outputs" / "scan"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--f-start", type=float, default=12.0, help="GHz")
    p.add_argument("--f-stop", type=float, default=18.0, help="GHz")
    p.add_argument("--f-step", type=float, default=0.5, help="GHz")
    p.add_argument("--pinn-freq", type=float, default=15.0,
                   help="GHz — single frequency at which to also train the PINN")
    p.add_argument("--pinn-iters", type=int, default=15000, help="Adam iterations")
    p.add_argument("--pinn-lbfgs", type=int, default=5000, help="L-BFGS max iter")
    p.add_argument("--no-pinn", action="store_true",
                   help="Skip PINN training — produce only the FDFD sweep")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    return p


def plot_s_param_curve(freqs, s11_db, s21_db, save_path, title="S-parameters vs frequency"):
    """|S11| / |S21| in dB vs GHz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, s11_db, "-o", label=r"$|S_{11}|$ (FDFD)")
    ax.plot(freqs, s21_db, "-s", label=r"$|S_{21}|$ (FDFD)")
    ax.set_xlabel("Frequency [GHz]")
    ax.set_ylabel("dB")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pattern_overlay(theta, patterns_db_by_freq, save_path,
                         title="Radiation patterns vs frequency"):
    """Overlay normalized E-plane patterns at each frequency on a polar plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    cmap = plt.get_cmap("viridis")
    items = sorted(patterns_db_by_freq.items())
    for i, (f_ghz, p_db) in enumerate(items):
        c = cmap(i / max(len(items) - 1, 1))
        ax.plot(theta, p_db, color=c, label=f"{f_ghz:.1f} GHz", linewidth=1.0)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(-40, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_scan(cfg: dict, outdir: Path, *,
             f_start: float = 12.0, f_stop: float = 18.0, f_step: float = 0.5,
             pinn_freq: float | None = 15.0,
             pinn_iters: int = 15000, pinn_lbfgs: int = 5000) -> dict:
    """Run the FDFD sweep + an optional single-frequency PINN comparison.

    Returns a summary dict (also written to ``scan_summary.yaml``).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    g = RadiationSlotGeometry(**cfg["geometry"])

    freqs = np.round(np.arange(f_start, f_stop + 1e-9, f_step), 6)
    print(f"[scan] FDFD sweep over {len(freqs)} frequencies "
          f"[{f_start}–{f_stop}] GHz @ {f_step} GHz step", flush=True)

    s11_db, s21_db = [], []
    pat_overlay = {}
    theta = np.deg2rad(np.linspace(-90, 90, 361))

    t0 = time.time()
    for i, f in enumerate(freqs):
        f = float(f)
        out = FDFDSolver(g, frequency_ghz=f,
                         mesh_size=cfg["fdfd"]["mesh_size"]).solve()
        k0 = free_space_wavenumber(f, eps_r=g.medium_epsilon)
        s11, s21 = s_parameters(out["Ez"], out["X"], out["Y"], g, k0)
        s11_db.append(20 * np.log10(max(abs(s11), 1e-12)))
        s21_db.append(20 * np.log10(max(abs(s21), 1e-12)))

        x_ap, ez_ap = aperture_field(out["Ez"], out["X"], out["Y"], g)
        pat = near_to_far_field(ez_ap, x_ap, k0, theta)
        pat_overlay[f] = 20 * np.log10(np.abs(pat) / np.max(np.abs(pat)))

        print(f"[scan] {i+1:2d}/{len(freqs)}  f={f:5.2f} GHz  "
              f"|S11|={s11_db[-1]:+6.2f} dB  |S21|={s21_db[-1]:+6.2f} dB",
              flush=True)
    print(f"[scan] FDFD sweep done in {time.time() - t0:.1f}s", flush=True)

    plot_s_param_curve(freqs, s11_db, s21_db,
                       str(outdir / "s_parameters_scan.png"),
                       title=f"FDFD |S11| / |S21| over Ku band ({f_start}-{f_stop} GHz)")
    plot_pattern_overlay(theta, pat_overlay,
                         str(outdir / "pattern_overlay.png"),
                         title="FDFD E-plane patterns over Ku band")

    summary = {
        "fdfd_sweep": {
            "frequencies_ghz": [float(x) for x in freqs.tolist()],
            "S11_dB": [float(x) for x in s11_db],
            "S21_dB": [float(x) for x in s21_db],
        },
        "pinn_single_freq": None,  # filled below if requested
    }

    # Optional single-frequency PINN comparison
    if pinn_freq is not None:
        print(f"\n[pinn] Training OptimizedPINNSolver at {pinn_freq} GHz "
              f"({pinn_iters} Adam + {pinn_lbfgs} L-BFGS) ...", flush=True)
        from examples.radiation_slot.pinn_solver_optimized import OptimizedPINNSolver

        fdfd_pinn = FDFDSolver(g, frequency_ghz=pinn_freq,
                               mesh_size=cfg["fdfd"]["mesh_size"]).solve()
        k0_pinn = free_space_wavenumber(pinn_freq, eps_r=g.medium_epsilon)

        t1 = time.time()
        solver = OptimizedPINNSolver(
            g, frequency_ghz=pinn_freq,
            num_domain=30000, num_boundary=2000, num_test=15000,
            fourier_frequencies=8, fourier_scale=1.0,
            scattered_field=True,
        )
        solver.train(iterations=pinn_iters, lr=cfg["pinn"]["adam_lr"])
        if pinn_lbfgs > 0:
            solver.finetune_lbfgs(max_iter=pinn_lbfgs)
        pinn_runtime = time.time() - t1

        Ez_pinn = solver.predict_on_grid(fdfd_pinn["X"], fdfd_pinn["Y"], fdfd_pinn["mask"])
        fmetrics = field_metrics(fdfd_pinn["Ez"], Ez_pinn, fdfd_pinn["mask"])
        s11_p, s21_p = s_parameters(Ez_pinn, fdfd_pinn["X"], fdfd_pinn["Y"], g, k0_pinn)
        s11_f, s21_f = s_parameters(fdfd_pinn["Ez"], fdfd_pinn["X"], fdfd_pinn["Y"],
                                    g, k0_pinn)

        x_ap_p, ez_ap_p = aperture_field(Ez_pinn, fdfd_pinn["X"], fdfd_pinn["Y"], g)
        x_ap_f, ez_ap_f = aperture_field(fdfd_pinn["Ez"], fdfd_pinn["X"], fdfd_pinn["Y"], g)
        pat_p = near_to_far_field(ez_ap_p, x_ap_p, k0_pinn, theta)
        pat_f = near_to_far_field(ez_ap_f, x_ap_f, k0_pinn, theta)
        pat_p_db = 20 * np.log10(np.maximum(np.abs(pat_p), 1e-12) / np.max(np.abs(pat_p)))
        pat_f_db = 20 * np.log10(np.abs(pat_f) / np.max(np.abs(pat_f)))
        pmetrics = pattern_metrics(theta, pat_f_db, pat_p_db)

        plot_field(fdfd_pinn["Ez"], fdfd_pinn["X"], fdfd_pinn["Y"], fdfd_pinn["mask"],
                   str(outdir / f"field_fdfd_{pinn_freq}GHz.png"),
                   title=f"FDFD |Ez| @ {pinn_freq} GHz")
        plot_field(Ez_pinn, fdfd_pinn["X"], fdfd_pinn["Y"], fdfd_pinn["mask"],
                   str(outdir / f"field_pinn_{pinn_freq}GHz.png"),
                   title=f"Optimized PINN |Ez| @ {pinn_freq} GHz")
        plot_radiation_pattern(theta, pat_f_db,
                               str(outdir / f"pattern_fdfd_{pinn_freq}GHz.png"),
                               title=f"FDFD pattern @ {pinn_freq} GHz")
        plot_radiation_pattern(theta, pat_p_db,
                               str(outdir / f"pattern_pinn_{pinn_freq}GHz.png"),
                               title=f"Optimized PINN pattern @ {pinn_freq} GHz")

        summary["pinn_single_freq"] = {
            "frequency_ghz": float(pinn_freq),
            "iters_adam": int(pinn_iters),
            "iters_lbfgs": int(pinn_lbfgs),
            "runtime_s": round(pinn_runtime, 1),
            "field": {k: float(v) for k, v in fmetrics.items()},
            "pattern": {k: float(v) for k, v in pmetrics.items()},
            "S11_fdfd_dB": float(20 * np.log10(max(abs(s11_f), 1e-12))),
            "S11_pinn_dB": float(20 * np.log10(max(abs(s11_p), 1e-12))),
            "S21_fdfd_dB": float(20 * np.log10(max(abs(s21_f), 1e-12))),
            "S21_pinn_dB": float(20 * np.log10(max(abs(s21_p), 1e-12))),
        }
        print(f"[pinn] done in {pinn_runtime:.1f}s — "
              f"L2={fmetrics['l2_relative']:.3f}, "
              f"corrcoef={fmetrics['corrcoef']:+.3f}, "
              f"main-lobe Δ={pmetrics['main_lobe_deg_diff']:+.2f}°", flush=True)

    with open(outdir / "scan_summary.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    return summary


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    pinn_freq = None if args.no_pinn else args.pinn_freq
    run_scan(
        cfg, Path(args.outdir),
        f_start=args.f_start, f_stop=args.f_stop, f_step=args.f_step,
        pinn_freq=pinn_freq,
        pinn_iters=args.pinn_iters,
        pinn_lbfgs=args.pinn_lbfgs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
