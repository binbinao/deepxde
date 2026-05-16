# Radar Waveguide Radiation-Slot Simulation

2D Helmholtz electromagnetic simulation of a rectangular-waveguide radiating
slot, in the Ku band (12–18 GHz). The package bundles two solvers that share
a single geometry / postprocess / comparator stack:

- **FDFD reference solver** (`fdfd_solver.py`) — SciPy sparse LU on a 5-point
  stencil with PEC walls, first-order Mur ABC, and a sin-profile TE10 port
  excitation. This is the "ground truth" for comparison.
- **DeepXDE PINN solver** (`pinn_solver.py`) — physics-informed neural network
  using `dde.nn.FNN` with real/imag dual outputs, `dde.icbc.DirichletBC` for
  PEC and `dde.icbc.RobinBC` for Mur ABCs. Trained end-to-end with Adam +
  L-BFGS.

Geometry is built with `dde.geometry.Rectangle` + CSG boolean operations
(union of waveguide and above-slot buffer).

## Quick Start

```bash
cd /path/to/deepxde
# Single-frequency run: FDFD reference + PINN + comparison + plots
python3 -m examples.radiation_slot.main --mode single --freq 15.0 \
    --iters 15000 --lbfgs 5000

# All outputs land in examples/radiation_slot/outputs/:
#   field_fdfd_15.0GHz.png     - FDFD |Ez| heatmap
#   field_pinn_15.0GHz.png     - PINN |Ez| heatmap
#   pattern_fdfd_15.0GHz.png   - FDFD far-field radiation pattern
#   pattern_pinn_15.0GHz.png   - PINN far-field radiation pattern
#   summary_15.0GHz.yaml       - S-parameters + comparison metrics
```

Run the unit tests (28 total, ~40 s on CPU):

```bash
python3 -m pytest examples/radiation_slot/tests/ -v
```

## Package Layout

```
examples/radiation_slot/
├── geometry.py              RadiationSlotGeometry: CSG domain + uniform grid + boundary marker
├── fdfd_solver.py           SciPy sparse Helmholtz reference solver
├── pinn_solver.py           DeepXDE PINN (non-dimensionalized ξ = k0·x)
├── postprocess.py           S-parameters (TE10 overlap integral) + NTFF radiation pattern + plots
├── comparator.py            Field/pattern metrics (L2 / L∞ / corrcoef / main-lobe Δ)
├── functions.py             Shared TE10 profile + wavenumber helpers
├── main.py                  CLI entry: single-frequency end-to-end pipeline
├── conf/radiation_slot.yaml Default parameters (geometry / fdfd mesh / pinn hparams / sweep range)
└── tests/                   Unit tests (geometry, fdfd, postprocess, comparator, pinn smoke)
```

## Mathematical Model

### Governing equation (2D, TE10 mode, Ez scalar)

$$
\nabla^2 E_z + k_0^2 \varepsilon_r E_z = 0, \qquad k_0 = 2\pi f / c
$$

Because DeepXDE does not support complex tensors, $E_z = u + j v$ is split into
real and imaginary components, each satisfying the same real Helmholtz equation
(coupled only through the Mur ABC).

### Boundary conditions

| Region | Physical meaning | Equation |
|---|---|---|
| `pec` | Metal walls | $E_z = 0$ |
| `port_in` (x=0) | TE10 incident + Mur absorbing | $\partial_x E_z - j k_0 E_z = -2 j k_0 \sin(\pi y/b)$ |
| `port_out` (x=L) | First-order Mur exit | $\partial_x E_z + j k_0 E_z = 0$ |
| `radiation` (buffer top/sides) | First-order Mur (4 normals) | $\partial_n E_z + j k_0 E_z = 0$ |

### PINN non-dimensionalization

Substituting $\xi = k_0 x$, $\eta = k_0 y$ inside the network:

- PDE: $\tilde\nabla^2 E_z + \varepsilon_r E_z = 0$ (no $k_0^2$ coefficient)
- Mur: $\partial_{\tilde n} E_z + j E_z = 0$ (no $k_0$)
- Port_in RHS: $-2 j \sin(\pi \eta / \tilde b)$ where $\tilde b = k_0 b$ (magnitude $\sim 2$ instead of $\sim 6.3$)

All residual terms start at $\mathcal O(1)$, so the training loss is balanced.
`PINNSolver.predict_on_grid(X, Y, mask)` takes *physical* coordinates and
rescales internally — the caller never sees the scaled frame.

## Design Decisions & Spec Corrections

The original specification (`docs/superpowers/specs/2026-04-30-radar-radiation-slot-design.md`)
was revised twice during implementation; both changes are documented in the
commit history:

1. **`waveguide_height` default 0.51 cm → 1.5 cm** (commit 840d17f)

   At 15 GHz the cross-section dimension that governs TE10 propagation must
   satisfy $b > c/(2f) = 1$ cm. With the original 0.51 cm the mode is below
   cutoff and the analytical plane-wave validation in `test_fdfd_solver.py`
   has $\sqrt{k_0^2 - (\pi/b)^2}$ imaginary. 1.5 cm gives guided wavelength
   $\lambda_g \approx 2.68$ cm, so a 4 cm waveguide hosts ~1.5 periods — enough
   to see standing-wave structure produced by the radiating slot.

2. **FDFD plane-wave test changed from single-L2 to three physical invariants**
   (commit 840d17f)

   First-order Mur absorbs only 90–95 % of the outgoing wave, so the spec's
   original ≤ 2 % full-domain L2 tolerance is unattainable with this BC class
   (measured error ~24 % even with a perfectly-assembled stencil). The new
   checks — transverse profile correlation, midline $k_x$ slope, midline
   VSWR — test the physics directly and remain invariant to acceptable Mur
   reflection levels.

## PINN Convergence — Known Limitation

**Status: the PINN solver does not reach the spec §4.2 accuracy gates on this
problem with vanilla DeepXDE + standard optimization.** The FDFD reference
and the full pipeline (geometry → solve → postprocess → compare → plot)
work correctly; the limitation is specific to the PINN solver's ability to
learn the oscillatory Helmholtz solution.

### What was tried

| Experiment | Adam iters | PDE residual (final) | Field L2 vs FDFD | corrcoef |
|---|---|---|---|---|
| Baseline (physical coords, tanh, PEC=100) | 3 000 + 1 000 L-BFGS | 0.023 | 1.20 | −0.21 |
| + Non-dimensionalization | 3 000 + 1 000 L-BFGS | 0.004 | 1.18 | −0.13 |
| + sin activation + port_in weight 100 | 3 000 + 1 000 L-BFGS | 0.005 | 1.19 | −0.14 |
| + Network scaled 4×256 → 6×512 | 3 000 + 1 000 L-BFGS | 0.005 | 1.20 | −0.17 |

Every change reduced BC residuals to ~1e-5, but the PDE residual plateaued
around 3–5e-3 regardless of network size. The predicted field magnitude is
close to zero almost everywhere — consistent with the well-documented
**trivial-solution / boundary-layer minimum** that standard PINN training
falls into on Helmholtz problems.

### Why

The TE10 source appears only as a *non-homogeneous term on a 1-D boundary*
(x = 0). Every other boundary is homogeneous, and the PDE itself is
homogeneous. The global minimum of the weighted PINN loss is the true
solution; however, a *local* minimum with almost the same loss value exists
where the network outputs a thin boundary layer near port_in and decays to
zero in the bulk. This local minimum is reached quickly and escaping it
requires techniques beyond standard Adam / L-BFGS on a plain FNN.

### Directions that would likely close the gap

These are out of scope for the present implementation but are well-supported
in the literature and in DeepXDE itself:

1. **Residual-based adaptive refinement (RAR)** — re-sample domain points
   where the PDE residual is largest. DeepXDE exposes
   `dde.callbacks.PDEPointResampler`. Proven to help Helmholtz-like problems
   in Wu et al., *Comput. Methods Appl. Mech. Eng.* (2023).

2. **Fourier feature input embedding** — pre-map $(\xi, \eta)$ through a bank
   of random $(\sin, \cos)$ projections before feeding the FNN. See Tancik et
   al. (2020) and spec §6 "risk mitigations" #3.

3. **Causal training / loss reweighting** — Wang et al. (2022) showed
   time-/space-marched causal weighting converges on wave problems where
   vanilla PINNs stall.

4. **Hard-constraint input transform** — encode the TE10 incident field
   directly via `net.apply_output_transform`, so the network only has to
   learn the scattered field (which is small) rather than the full solution.

5. **Expanded operator library** — DeepXDE has `gPINN` and `DeepONet` modules
   that can be composed with the same geometry; both are outside the scope
   of this example.

### What still works correctly

Everything except the PINN accuracy gate:

- ✅ `RadiationSlotGeometry` builds the correct CSG domain, grid, boundary marker
- ✅ `FDFDSolver` passes its analytical plane-wave and radiating-regime tests
- ✅ `s_parameters()` returns $|S_{11}| \approx 0, |S_{21}| \approx 1$ on a pure TE10 wave
- ✅ `near_to_far_field()` reproduces the expected main-lobe direction for a uniform aperture
- ✅ `PINNSolver.train` / `.finetune_lbfgs` / `.predict_on_grid` all run to completion on CPU and GPU (paddle/pytorch)
- ✅ `main.py` produces all four PNG plots and the summary YAML
- ✅ 28 unit tests pass (~40 s wall time)

Users wanting a *working* 2D Helmholtz reference for this geometry should use
`FDFDSolver` directly; users wanting to *research* Helmholtz PINN convergence
techniques can use `PINNSolver` as the starting baseline and plug in one of
the five directions above.

## Optimized PINN Solver

`pinn_solver_optimized.py` ships an `OptimizedPINNSolver` that combines the
first two remediation directions from the list above:

1. **Scattered-field decomposition** — total field $E_z = E_z^{inc} + E_z^{sc}$
   where $E_z^{inc}$ is the analytical TE10 traveling wave; the network only
   learns $E_z^{sc}$. This folds the port_in source term into the analytical
   incident, replacing it with a non-zero radiation BC RHS that prevents the
   trivial $E_z^{sc} = 0$ collapse.

2. **Fourier-feature input embedding** — the (ξ, η) coordinates are passed
   through $[\sin(m\xi), \cos(m\xi), \sin(m\eta), \cos(m\eta)]$ for $m = 1..M$
   before the FNN, providing oscillatory basis functions for the high-frequency
   solution.

Both knobs are independently configurable. `predict_on_grid` still returns
the *total* physical $E_z$, so `main.py` and the postprocess/comparator stack
work unchanged.

### Ablation (3k Adam + 1k L-BFGS @ 15 GHz, default geometry)

| Configuration | L2 | corrcoef | main-lobe Δ° |
| --- | ---: | ---: | ---: |
| Baseline (no Fourier, no scattered) | 1.17 | −0.11 | +56° |
| Scattered only | 1.21 | −0.06 | +84° |
| Fourier only ($M = 6$) | 1.06 | −0.02 | +32° |
| **Fourier + scattered** | **0.63** | **+0.84** | **−14°** |

Either knob alone is essentially ineffective; **the two together** flip the
correlation coefficient from negative to strongly positive. The Fourier
features give the network the basis it needs to express oscillatory solutions
while the scattered decomposition removes the trivial-solution local minimum
that traps standard training.

To reproduce: `python3 -m examples.radiation_slot.experiments.ab_compare`.

### Extended training results

Two extended runs at 15 GHz with the full optimization stack:

| Run | Fourier $M$ | num_domain | Adam + L-BFGS | L2 | corrcoef | main-lobe Δ° | \|S<sub>11</sub>\| Δ dB | test/train |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| V1 | 12 | 8 000 | 15k + 5k | 0.71 | +0.77 | +99.5° | 1.86 | 16 000× ⚠ |
| **V2** | **8** | **30 000** | **15k + 5k** | **0.66** | **+0.82** | **−3.5°** | **3.26** | **688×** |

V2 was tuned in response to V1's severe overfitting (test_loss / train_loss
≈ 16 000×). Quadrupling the domain training points and reducing the Fourier
embedding to $M = 8$ brought the ratio down by ~23×; the V1 |S<sub>11</sub>|
score below 2 dB was a coincidental product of overfitting and is not
replicated in V2's more genuine generalization.

The most physically meaningful indicator — the far-field main-lobe direction —
went from completely wrong in V1 (+99.5° off-axis) to within −3.5° of the
FDFD reference in V2, just outside the spec tolerance of ±2°.

To reproduce V2: `python3 -m examples.radiation_slot.experiments.run_v2_fourier8_dom30k`
(allow ~90 minutes on a single GPU).

### Status against spec §4.2

| Metric | Spec gate | V2 result | Notes |
| --- | --- | --- | --- |
| Field L2 relative | ≤ 0.05 | 0.66 | ❌ open |
| Field correlation | ≥ 0.98 | 0.82 | ❌ open |
| Main-lobe Δ° | ≤ 2° | 3.5° | ⚠ near miss |
| \|S<sub>11</sub>\| Δ dB | ≤ 2 dB | 3.3 dB | ⚠ near miss |

V2 closes most of the gap on physically dominant integrated quantities
(main lobe, $|S_{11}|$). The remaining gap on local field metrics (L2 and
correlation) is consistent with the Fourier-feature embedding capturing the
dominant modes but missing finer spatial structure. The remaining headroom
should come from the next remediation directions in this README's
"Directions that would likely close the gap" list — particularly **RAR**
(residual-based adaptive refinement) and **causal training**, which were
not attempted in this set of experiments.

## Frequency Sweep (M6)

The 13-point Ku-band sweep (12–18 GHz at 0.5 GHz steps) with PINN warm-starts
is specified in the plan (Task 16 / `main_scan.py`) but not implemented, since
the single-frequency PINN accuracy problem reported above would compound
across frequencies without first being resolved. The FDFD side of the sweep
is trivial to run in a Python loop with the existing `FDFDSolver` and
`s_parameters()`; add a 20-line driver once the PINN convergence path is
chosen.

## References

- Lu, Meng, Mao & Karniadakis, *DeepXDE: A deep learning library for solving
  differential equations*, SIAM Rev. 63 (2021), 208–228.
- Wang, Yu & Perdikaris, *When and why PINNs fail to train: a neural tangent
  kernel perspective*, J. Comput. Phys. 449 (2022), 110768.
- Wu, Hu, Pang & Karniadakis, *A comprehensive study of non-adaptive and
  residual-based adaptive sampling for physics-informed neural networks*,
  CMAME 403 (2023), 115671.
- Tancik, Srinivasan, Mildenhall et al., *Fourier features let networks learn
  high frequency functions in low dimensional domains*, NeurIPS (2020).
- Mur, *Absorbing boundary conditions for the finite-difference approximation
  of the time-domain electromagnetic-field equations*, IEEE TEMC 23 (1981).
