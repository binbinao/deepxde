# Radar Radiation Slot Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 2D Helmholtz radar radiation slot simulator using DeepXDE PINN, validated against a SciPy-based FDFD reference solver, with single-frequency MVP at 15 GHz and optional 13-point Ku-band sweep.

**Architecture:** Modular sub-project under `examples/radiation_slot/`. Geometry built via `dde.geometry` CSG; FDFD uses `scipy.sparse.linalg.spsolve` over complex five-point stencil with PEC + first-order Mur ABC + sin-profile TE10 port; PINN uses `dde.nn.FNN([2,256,256,256,256,2])` with real/imag dual outputs and `RobinBC` mirroring the FDFD boundary conditions. Postprocess shares NTFF + S-parameter routines between solvers; comparator quantifies L2/L∞/correlation/main-lobe metrics.

**Tech Stack:** Python 3, DeepXDE (default backend = PyTorch), NumPy, SciPy (sparse + spsolve), Matplotlib, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-04-30-radar-radiation-slot-design.md`

---

## Conventions

- All paths under `examples/radiation_slot/` unless noted.
- All `pytest` commands run from repo root: `cd /data/deepxde && pytest examples/radiation_slot/tests/...`
- Backend selection per script: `os.environ.setdefault("DDE_BACKEND", "pytorch")` before `import deepxde`.
- Each task ends with a single git commit so history mirrors milestones.

---

## Milestone M1: Geometry + Infrastructure

### Task 1: Scaffold the package skeleton

**Files:** Create `__init__.py`, `conf/radiation_slot.yaml`, `tests/__init__.py`, `outputs/.gitkeep`, `.gitignore`.

- [ ] **Step 1: Create `examples/radiation_slot/__init__.py`**

```python
"""Radar rectangular-waveguide radiation slot simulation example."""
from examples.radiation_slot.geometry import RadiationSlotGeometry  # noqa: F401
```

- [ ] **Step 2: Create `examples/radiation_slot/tests/__init__.py`** (empty file).

- [ ] **Step 3: Create `examples/radiation_slot/.gitignore`**

```
outputs/*
!outputs/.gitkeep
__pycache__/
*.pyc
```

- [ ] **Step 4: Create `examples/radiation_slot/outputs/.gitkeep`** (empty).

- [ ] **Step 5: Create `examples/radiation_slot/conf/radiation_slot.yaml`**

```yaml
geometry:
  waveguide_width: 4.0
  waveguide_height: 0.51
  slot_length: 1.5
  slot_width: 0.16
  slot_position: 0.5
  buffer_height: 1.5
  medium_epsilon: 1.0
  medium_mu: 1.0
fdfd:
  mesh_size: 0.05
pinn:
  hidden_layers: [256, 256, 256, 256]
  activation: "tanh"
  initializer: "Glorot uniform"
  num_domain: 8000
  num_boundary: 800
  num_test: 40000
  adam_iterations: 15000
  adam_lr: 1.0e-3
  lbfgs_max_iter: 5000
  loss_weights: {pde: 1.0, pec: 100.0, mur: 100.0}
frequencies:
  default_ghz: 15.0
  scan_start_ghz: 12.0
  scan_stop_ghz: 18.0
  scan_step_ghz: 0.5
```

- [ ] **Step 6: Verify import skeleton fails as expected (geometry not yet implemented)**

Run: `cd /data/deepxde && python -c "import examples.radiation_slot" 2>&1 | head -3`
Expected: `ModuleNotFoundError: No module named 'examples.radiation_slot.geometry'` — confirms package wiring.

- [ ] **Step 7: Commit**

```bash
cd /data/deepxde
git add examples/radiation_slot
git commit -m "feat(radiation_slot): scaffold package skeleton + default YAML config"
```

---

### Task 2: Geometry — dataclass + `build_dde_geometry`

**Files:** Create `geometry.py`, `tests/test_geometry.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_geometry.py
import numpy as np
import pytest

from examples.radiation_slot.geometry import RadiationSlotGeometry


def test_dataclass_defaults():
    g = RadiationSlotGeometry()
    assert (g.waveguide_width, g.waveguide_height) == (4.0, 0.51)
    assert (g.slot_length, g.slot_width, g.slot_position) == (1.5, 0.16, 0.5)
    assert (g.buffer_height, g.medium_epsilon, g.medium_mu) == (1.5, 1.0, 1.0)


def test_csg_inside_outside():
    g = RadiationSlotGeometry()
    geom = g.build_dde_geometry()
    assert geom.inside(np.array([[2.0, 0.25]]))[0]                    # waveguide
    assert geom.inside(np.array([[2.0, 0.51 + 0.5]]))[0]              # buffer above slot
    assert not geom.inside(np.array([[2.0, 0.51 + 1.5 + 0.1]]))[0]    # above buffer
    assert not geom.inside(np.array([[0.2, 0.51 + 0.1]]))[0]          # above wg, NOT under slot
```

- [ ] **Step 2: Run test → expect `ImportError`**

Run: `cd /data/deepxde && pytest examples/radiation_slot/tests/test_geometry.py -v`

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/geometry.py
"""Geometry, uniform FDFD grid, and boundary marker for the radiation-slot demo."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import deepxde as dde


@dataclass
class RadiationSlotGeometry:
    """All lengths in cm. 2D domain = waveguide ∪ buffer (above the slot opening)."""
    waveguide_width: float = 4.0      # propagation direction (x in 2D)
    waveguide_height: float = 0.51    # TE10 cutoff dimension (physical b)
    slot_length: float = 1.5
    slot_width: float = 0.16
    slot_position: float = 0.5
    buffer_height: float = 1.5
    medium_epsilon: float = 1.0
    medium_mu: float = 1.0

    def slot_x_range(self) -> tuple[float, float]:
        c = self.slot_position * self.waveguide_width
        return c - self.slot_length / 2, c + self.slot_length / 2

    def build_dde_geometry(self) -> dde.geometry.Geometry:
        wg = dde.geometry.Rectangle([0.0, 0.0], [self.waveguide_width, self.waveguide_height])
        if self.slot_length <= 0 or self.buffer_height <= 0:
            return wg
        x_lo, x_hi = self.slot_x_range()
        buf = dde.geometry.Rectangle(
            [x_lo, self.waveguide_height],
            [x_hi, self.waveguide_height + self.buffer_height],
        )
        return wg | buf
```

- [ ] **Step 4: Run test → expect 2 passed**

Run: `cd /data/deepxde && pytest examples/radiation_slot/tests/test_geometry.py -v`

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/geometry.py examples/radiation_slot/tests/test_geometry.py
git commit -m "feat(radiation_slot): geometry dataclass + dde CSG domain"
```

---

### Task 3: Geometry — `fdfd_grid()`

**Files:** Modify `geometry.py`, `tests/test_geometry.py`.

- [ ] **Step 1: Append failing test**

```python
def test_fdfd_grid_shape_and_mask():
    g = RadiationSlotGeometry()
    X, Y, mask = g.fdfd_grid(mesh_size=0.05)
    assert X.shape == Y.shape == mask.shape
    assert X.shape[1] == int(round(g.waveguide_width / 0.05)) + 1
    assert X.shape[0] == int(round((g.waveguide_height + g.buffer_height) / 0.05)) + 1
    area_grid = mask.sum() * 0.05 * 0.05
    area_true = g.waveguide_width * g.waveguide_height + g.slot_length * g.buffer_height
    assert area_grid == pytest.approx(area_true, rel=0.05)
```

- [ ] **Step 2: Run → expect `AttributeError: fdfd_grid`**

- [ ] **Step 3: Append to `geometry.py`**

```python
    def fdfd_grid(self, mesh_size: float):
        """Uniform Cartesian grid + boolean domain mask. Returns (X, Y, mask)."""
        h = mesh_size
        nx = int(round(self.waveguide_width / h)) + 1
        ny = int(round((self.waveguide_height + self.buffer_height) / h)) + 1
        x = np.linspace(0.0, self.waveguide_width, nx)
        y = np.linspace(0.0, self.waveguide_height + self.buffer_height, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")
        in_wg = Y <= self.waveguide_height + 1e-12
        x_lo, x_hi = self.slot_x_range()
        in_buf = (Y > self.waveguide_height + 1e-12) & (X >= x_lo - 1e-12) & (X <= x_hi + 1e-12)
        return X, Y, in_wg | in_buf
```

- [ ] **Step 4: Run → expect 3 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/geometry.py examples/radiation_slot/tests/test_geometry.py
git commit -m "feat(radiation_slot): uniform FDFD grid + domain mask"
```

---

### Task 4: Geometry — `boundary_marker()`

**Files:** Modify `geometry.py`, `tests/test_geometry.py`.

- [ ] **Step 1: Append failing tests**

```python
def test_boundary_marker_basic_labels():
    g = RadiationSlotGeometry()
    pts = np.array([
        [0.0, 0.25],                                    # port_in
        [g.waveguide_width, 0.25],                      # port_out
        [2.0, 0.0],                                     # pec (waveguide bottom)
        [0.5, g.waveguide_height],                      # pec (top wall, NOT under slot)
        [2.0, g.waveguide_height + g.buffer_height],    # radiation (buffer top)
        [g.slot_x_range()[0], g.waveguide_height + 0.5],# radiation (buffer side)
        [2.0, 0.25],                                    # interior
    ])
    assert g.boundary_marker(pts).tolist() == [
        "port_in", "port_out", "pec", "pec", "radiation", "radiation", "interior"
    ]


def test_boundary_marker_pec_radiation_disjoint():
    g = RadiationSlotGeometry()
    pts = g.build_dde_geometry().random_boundary_points(500)
    labels = g.boundary_marker(pts)
    assert not ((labels == "pec") & (labels == "radiation")).any()
    assert (labels != "interior").all()
```

- [ ] **Step 2: Run → expect 2 new failures**

- [ ] **Step 3: Append to `geometry.py`**

```python
    def boundary_marker(self, x: np.ndarray, atol: float = 1e-9) -> np.ndarray:
        """Classify points (N, 2) into {'port_in','port_out','pec','radiation','interior'}."""
        x = np.atleast_2d(x)
        labels = np.full(x.shape[0], "interior", dtype=object)
        wg_w, wg_h = self.waveguide_width, self.waveguide_height
        buf_top = wg_h + self.buffer_height
        slot_lo, slot_hi = self.slot_x_range()

        on_x0 = np.isclose(x[:, 0], 0.0, atol=atol)
        on_xL = np.isclose(x[:, 0], wg_w, atol=atol)
        on_y0 = np.isclose(x[:, 1], 0.0, atol=atol)
        on_yh = np.isclose(x[:, 1], wg_h, atol=atol)
        on_yt = np.isclose(x[:, 1], buf_top, atol=atol)

        in_wg_y = (x[:, 1] >= -atol) & (x[:, 1] <= wg_h + atol)
        in_buf_x = (x[:, 0] >= slot_lo - atol) & (x[:, 0] <= slot_hi + atol)

        labels[on_x0 & in_wg_y] = "port_in"
        labels[on_xL & in_wg_y] = "port_out"

        bottom = on_y0 & (x[:, 0] >= -atol) & (x[:, 0] <= wg_w + atol)
        top_left = on_yh & (x[:, 0] < slot_lo - atol)
        top_right = on_yh & (x[:, 0] > slot_hi + atol)
        labels[bottom | top_left | top_right] = "pec"

        on_buf_left = np.isclose(x[:, 0], slot_lo, atol=atol) & (x[:, 1] > wg_h + atol)
        on_buf_right = np.isclose(x[:, 0], slot_hi, atol=atol) & (x[:, 1] > wg_h + atol)
        on_buf_top = on_yt & in_buf_x
        labels[on_buf_left | on_buf_right | on_buf_top] = "radiation"

        return labels
```

- [ ] **Step 4: Run → expect 5 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/geometry.py examples/radiation_slot/tests/test_geometry.py
git commit -m "feat(radiation_slot): boundary marker (port_in/port_out/pec/radiation)"
```

---

### Task 5: Shared `functions.py`

**Files:** Create `functions.py`, `tests/test_functions.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_functions.py
import numpy as np
import pytest
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile, te10_norm


def test_wavenumber_15ghz():
    # f=15GHz, c=3e10 cm/s → k0 = 2π·15e9/3e10 = π rad/cm
    assert free_space_wavenumber(15.0) == pytest.approx(np.pi, rel=1e-3)


def test_te10_profile_endpoints_zero():
    assert te10_profile(np.array([0.0, 0.51]), b=0.51) == pytest.approx([0.0, 0.0], abs=1e-12)


def test_te10_profile_peak_at_midline():
    assert te10_profile(np.array([0.255]), b=0.51)[0] == pytest.approx(1.0, rel=1e-6)


def test_te10_norm_is_b_over_2():
    assert te10_norm(b=0.51) == pytest.approx(0.255, rel=1e-12)
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/functions.py
"""Shared math utilities. Lengths in cm, frequencies in GHz."""
from __future__ import annotations
import numpy as np

C_CM = 3.0e10  # speed of light in cm/s


def free_space_wavenumber(frequency_ghz: float, eps_r: float = 1.0) -> float:
    return 2.0 * np.pi * frequency_ghz * 1e9 * np.sqrt(eps_r) / C_CM


def te10_profile(y: np.ndarray, b: float) -> np.ndarray:
    return np.sin(np.pi * y / b)


def te10_norm(b: float) -> float:
    return b / 2.0
```

- [ ] **Step 4: Run → 4 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/functions.py examples/radiation_slot/tests/test_functions.py
git commit -m "feat(radiation_slot): TE10 profile + wavenumber helpers"
```

---

## Milestone M2: FDFD Reference Solver

### Task 6: FDFD — assembly + PEC + Mur on no-slot waveguide

**Files:** Create `fdfd_solver.py`, `tests/test_fdfd_solver.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_fdfd_solver.py
"""FDFD tests. Benchmark uses a no-slot waveguide so the analytical solution
is the pure TE10 traveling wave Ez(x,y) = sin(πy/b)·exp(-jkx x).
Isolates: 5-pt stencil + PEC top/bottom + Mur exit + sin port excitation."""
import numpy as np
import pytest
from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile
from examples.radiation_slot.geometry import RadiationSlotGeometry


@pytest.fixture
def no_slot_geometry():
    return RadiationSlotGeometry(slot_length=0.0, slot_width=0.0, buffer_height=0.0)


def test_uniform_medium_plane_wave(no_slot_geometry):
    f = 15.0
    out = FDFDSolver(no_slot_geometry, frequency_ghz=f, mesh_size=0.05).solve()
    X, Y, Ez, mask = out["X"], out["Y"], out["Ez"], out["mask"]
    k0 = free_space_wavenumber(f)
    b = no_slot_geometry.waveguide_height
    kx = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)
    Ez_ref = te10_profile(Y, b=b) * np.exp(-1j * kx * X)
    err = np.linalg.norm((Ez - Ez_ref)[mask]) / np.linalg.norm(Ez_ref[mask])
    assert err < 0.02, f"L2 relative error {err:.3%} exceeds 2 %"


def test_pec_dirichlet_zero(no_slot_geometry):
    out = FDFDSolver(no_slot_geometry, frequency_ghz=15.0, mesh_size=0.05).solve()
    Y, Ez = out["Y"], out["Ez"]
    pec_top = np.isclose(Y, no_slot_geometry.waveguide_height)
    pec_bot = np.isclose(Y, 0.0)
    assert np.max(np.abs(Ez[pec_top])) < 1e-8
    assert np.max(np.abs(Ez[pec_bot])) < 1e-8


def test_mur_low_reflection(no_slot_geometry):
    f = 15.0
    out = FDFDSolver(no_slot_geometry, frequency_ghz=f, mesh_size=0.05).solve()
    X, Y, Ez = out["X"], out["Y"], out["Ez"]
    b = no_slot_geometry.waveguide_height
    k0 = free_space_wavenumber(f)
    kx = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)
    Ez_inc = te10_profile(Y, b=b) * np.exp(-1j * kx * X)
    inlet = np.isclose(X, 0.0)
    refl = np.linalg.norm(Ez[inlet] - Ez_inc[inlet]) / np.linalg.norm(Ez_inc[inlet])
    assert refl < 0.05, f"Mur reflection {refl:.3%} exceeds 5 %"
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

Run: `cd /data/deepxde && pytest examples/radiation_slot/tests/test_fdfd_solver.py -v`

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/fdfd_solver.py
"""SciPy FDFD reference solver: ∇²Ez + k0² ε_r Ez = 0.

BCs:
    pec       : Ez = 0
    port_in   : ∂Ez/∂x − jk0 Ez = -2 j k0 sin(πy/b)   (TE10 incident + Mur)
    port_out  : ∂Ez/∂x + jk0 Ez = 0                    (1st-order Mur out)
    radiation : ∂Ez/∂n + jk0 Ez = 0 on each outward face
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile
from examples.radiation_slot.geometry import RadiationSlotGeometry


@dataclass
class FDFDSolver:
    geometry: RadiationSlotGeometry
    frequency_ghz: float
    mesh_size: float = 0.05

    def solve(self) -> dict:
        g, h = self.geometry, self.mesh_size
        X, Y, mask = g.fdfd_grid(h)
        ny, nx = X.shape
        N = nx * ny
        k0 = free_space_wavenumber(self.frequency_ghz, eps_r=g.medium_epsilon)
        wg_h = g.waveguide_height
        buf_top = wg_h + g.buffer_height
        slot_lo, slot_hi = g.slot_x_range()
        atol = 1e-9
        idx = lambda j, i: j * nx + i

        A = lil_matrix((N, N), dtype=np.complex128)
        b_rhs = np.zeros(N, dtype=np.complex128)

        for j in range(ny):
            for i in range(nx):
                p = idx(j, i)
                xv, yv = X[j, i], Y[j, i]
                if not mask[j, i]:
                    A[p, p] = 1.0; continue

                on_y0 = np.isclose(yv, 0.0, atol=atol)
                on_yh = np.isclose(yv, wg_h, atol=atol)
                in_slot_x = (xv >= slot_lo - atol) and (xv <= slot_hi + atol)
                if on_y0 or (on_yh and (g.slot_length == 0.0 or not in_slot_x)):
                    A[p, p] = 1.0; continue

                if np.isclose(xv, 0.0, atol=atol) and yv <= wg_h + atol:
                    A[p, p] = -1.0 / h - 1j * k0
                    A[p, idx(j, i + 1)] = 1.0 / h
                    b_rhs[p] = -2j * k0 * te10_profile(np.array([yv]), b=wg_h)[0]
                    continue

                if np.isclose(xv, g.waveguide_width, atol=atol) and yv <= wg_h + atol:
                    A[p, p] = 1.0 / h + 1j * k0
                    A[p, idx(j, i - 1)] = -1.0 / h
                    continue

                on_buf_top = np.isclose(yv, buf_top, atol=atol)
                on_buf_left = np.isclose(xv, slot_lo, atol=atol) and yv > wg_h + atol
                on_buf_right = np.isclose(xv, slot_hi, atol=atol) and yv > wg_h + atol
                if on_buf_top:
                    A[p, p] = 1.0 / h + 1j * k0; A[p, idx(j - 1, i)] = -1.0 / h; continue
                if on_buf_left:
                    A[p, p] = -1.0 / h - 1j * k0; A[p, idx(j, i + 1)] = 1.0 / h; continue
                if on_buf_right:
                    A[p, p] = 1.0 / h + 1j * k0; A[p, idx(j, i - 1)] = -1.0 / h; continue

                A[p, p] = -4.0 / h ** 2 + k0 ** 2 * g.medium_epsilon
                A[p, idx(j, i + 1)] = 1.0 / h ** 2
                A[p, idx(j, i - 1)] = 1.0 / h ** 2
                A[p, idx(j + 1, i)] = 1.0 / h ** 2
                A[p, idx(j - 1, i)] = 1.0 / h ** 2

        Ez = spsolve(A.tocsr(), b_rhs).reshape(ny, nx)
        return {"X": X, "Y": Y, "Ez": Ez, "mask": mask, "k0": k0}
```

- [ ] **Step 4: Run → 3 passed (5–15 s)**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/fdfd_solver.py examples/radiation_slot/tests/test_fdfd_solver.py
git commit -m "feat(radiation_slot): FDFD solver (PEC + sin-port + Mur)"
```

---

### Task 7: FDFD — slot-on smoke test

**Files:** Modify `tests/test_fdfd_solver.py`. (Same solver handles slot==0 and slot>0.)

- [ ] **Step 1: Append test**

```python
def test_fdfd_with_slot_returns_finite_field():
    g = RadiationSlotGeometry()
    out = FDFDSolver(g, frequency_ghz=15.0, mesh_size=0.05).solve()
    Ez, mask = out["Ez"], out["mask"]
    inside = Ez[mask]
    assert np.isfinite(inside).all()
    assert 1e-2 < np.max(np.abs(inside)) < 1e3
```

- [ ] **Step 2: Run → expect PASS immediately**

- [ ] **Step 3: Commit**

```bash
git add examples/radiation_slot/tests/test_fdfd_solver.py
git commit -m "test(radiation_slot): FDFD radiating regime smoke test"
```

---

## Milestone M3: Postprocess + Comparator

### Task 8: Postprocess — S-parameters

**Files:** Create `postprocess.py`, `tests/test_postprocess.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_postprocess.py
import numpy as np
import pytest
from examples.radiation_slot.functions import free_space_wavenumber, te10_profile
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import s_parameters


def test_s_parameters_pure_transmission():
    g = RadiationSlotGeometry(slot_length=0.0, buffer_height=0.0)
    f = 15.0
    k0 = free_space_wavenumber(f)
    X, Y, _ = g.fdfd_grid(0.05)
    b = g.waveguide_height
    kx = np.sqrt(k0 ** 2 - (np.pi / b) ** 2)
    Ez = te10_profile(Y, b=b) * np.exp(-1j * kx * X)
    s11, s21 = s_parameters(Ez, X, Y, g, k0)
    assert abs(s11) < 0.05
    assert abs(abs(s21) - 1.0) < 0.05
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/postprocess.py
"""Postprocess: S-parameters + NTFF + matplotlib plotters."""
from __future__ import annotations
import numpy as np
from examples.radiation_slot.functions import te10_profile, te10_norm


def _column_indices(X, x_target, atol=1e-9):
    return np.where(np.isclose(X[0, :], x_target, atol=atol))[0]


def s_parameters(Ez, X, Y, geometry, k0) -> tuple[complex, complex]:
    """S11=(1/N)∫(Ez(0,y)-Ez_inc)φ̄ dy ; S21=(1/N)∫Ez(L,y)φ̄ dy."""
    b = geometry.waveguide_height
    N = te10_norm(b)
    cols_in = _column_indices(X, 0.0)
    j_wg = np.where(Y[:, cols_in[0]] <= b + 1e-9)[0]
    y_wg = Y[j_wg, cols_in[0]]
    phi = te10_profile(y_wg, b=b)
    inc = phi  # at x=0, exp(-jkx·0) = 1
    s11 = np.trapz((Ez[j_wg, cols_in[0]] - inc) * np.conj(phi), y_wg) / N
    cols_out = _column_indices(X, geometry.waveguide_width)
    s21 = np.trapz(Ez[j_wg, cols_out[0]] * np.conj(phi), y_wg) / N
    return complex(s11), complex(s21)
```

- [ ] **Step 4: Run → 1 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/postprocess.py examples/radiation_slot/tests/test_postprocess.py
git commit -m "feat(radiation_slot): S-parameters via TE10 overlap integral"
```

---

### Task 9: Postprocess — NTFF + aperture extractor

**Files:** Modify `postprocess.py`, `tests/test_postprocess.py`.

- [ ] **Step 1: Append failing test**

```python
def test_ntff_uniform_aperture_main_lobe_at_zero():
    from examples.radiation_slot.postprocess import near_to_far_field
    x_ap = np.linspace(-1.0, 1.0, 201)
    Ez_ap = np.ones_like(x_ap, dtype=complex)
    k0 = np.pi
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    pat = near_to_far_field(Ez_ap, x_ap, k0, theta)
    pat_db = 20 * np.log10(np.abs(pat) / np.max(np.abs(pat)))
    main_lobe_deg = np.rad2deg(theta[int(np.argmax(pat_db))])
    assert abs(main_lobe_deg) < 1.0
```

- [ ] **Step 2: Run → `ImportError`**

- [ ] **Step 3: Append to `postprocess.py`**

```python
def near_to_far_field(Ez_aperture, x_aperture, k0, theta) -> np.ndarray:
    """2D Stratton-Chu: E_far(θ) ∝ ∫ Ez(x) exp(-j k0 x sinθ) dx. Returns (Nθ,) complex."""
    sin_t = np.sin(theta)[:, None]
    kernel = np.exp(-1j * k0 * x_aperture[None, :] * sin_t)
    return np.trapz(kernel * Ez_aperture[None, :], x_aperture, axis=1)


def aperture_field(Ez, X, Y, geometry):
    """Slice field at y = waveguide_height, x ∈ [slot_lo, slot_hi]."""
    b = geometry.waveguide_height
    j = int(np.argmin(np.abs(Y[:, 0] - b)))
    x_lo, x_hi = geometry.slot_x_range()
    cols = np.where((X[j, :] >= x_lo - 1e-9) & (X[j, :] <= x_hi + 1e-9))[0]
    return X[j, cols], Ez[j, cols]
```

- [ ] **Step 4: Run → 2 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/postprocess.py examples/radiation_slot/tests/test_postprocess.py
git commit -m "feat(radiation_slot): NTFF radiation pattern + aperture extractor"
```

---

### Task 10: Postprocess — plotters

**Files:** Modify `postprocess.py`, `tests/test_postprocess.py`.

- [ ] **Step 1: Append failing tests**

```python
def test_plot_field_writes_file(tmp_path):
    from examples.radiation_slot.postprocess import plot_field
    g = RadiationSlotGeometry()
    X, Y, mask = g.fdfd_grid(0.1)
    Ez = np.exp(-((X - 2) ** 2 + (Y - 0.25) ** 2)).astype(complex)
    out = tmp_path / "field.png"
    plot_field(Ez, X, Y, mask, str(out), title="dummy")
    assert out.exists() and out.stat().st_size > 0


def test_plot_radiation_pattern_writes_file(tmp_path):
    from examples.radiation_slot.postprocess import plot_radiation_pattern
    theta = np.linspace(-np.pi / 2, np.pi / 2, 181)
    pat_db = -20 * np.abs(theta)
    out = tmp_path / "pat.png"
    plot_radiation_pattern(theta, pat_db, str(out), title="dummy")
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run → 2 new ImportError failures**

- [ ] **Step 3: Append to `postprocess.py`**

```python
def plot_field(Ez, X, Y, mask, save_path, title="|E_z|"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    amp = np.where(mask, np.abs(Ez), np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(X, Y, amp, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="|E_z|")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]")
    ax.set_aspect("equal"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def plot_radiation_pattern(theta, pattern_db, save_path, title="Radiation pattern"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(theta, pattern_db)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_rlim(-40, max(0, np.max(pattern_db) + 1))
    ax.set_title(title); fig.tight_layout()
    fig.savefig(save_path, dpi=150); plt.close(fig)
```

- [ ] **Step 4: Run → 4 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/postprocess.py examples/radiation_slot/tests/test_postprocess.py
git commit -m "feat(radiation_slot): field + radiation-pattern matplotlib plotters"
```

---

### Task 11: Comparator metrics

**Files:** Create `comparator.py`, `tests/test_comparator.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_comparator.py
import numpy as np
import pytest
from examples.radiation_slot.comparator import field_metrics, pattern_metrics


def test_field_metrics_identity():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((10, 10)) + 1j * rng.standard_normal((10, 10))
    mask = np.ones_like(ref, dtype=bool)
    m = field_metrics(ref, ref, mask)
    assert m["l2_relative"] == pytest.approx(0.0, abs=1e-12)
    assert m["linf"] == pytest.approx(0.0, abs=1e-12)
    assert m["corrcoef"] == pytest.approx(1.0, abs=1e-12)


def test_field_metrics_known_diff():
    ref = np.ones((4, 4), dtype=complex)
    pred = ref * 1.1
    mask = np.ones_like(ref, dtype=bool)
    m = field_metrics(ref, pred, mask)
    assert m["l2_relative"] == pytest.approx(0.1, rel=1e-6)
    assert m["linf"] == pytest.approx(0.1, rel=1e-6)


def test_pattern_metrics_main_lobe_shift():
    theta = np.deg2rad(np.linspace(-90, 90, 361))
    p_ref = -np.abs(theta) * 30
    p_pred = -np.abs(theta - np.deg2rad(3)) * 30
    m = pattern_metrics(theta, p_ref, p_pred)
    assert m["main_lobe_deg_diff"] == pytest.approx(3.0, abs=0.6)
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/comparator.py
"""Comparison metrics between FDFD reference and PINN prediction."""
from __future__ import annotations
import numpy as np


def field_metrics(ref, pred, mask) -> dict:
    """L2-relative / L∞ / Pearson correlation, evaluated only where mask=True."""
    r, p = ref[mask], pred[mask]
    diff = p - r
    l2 = float(np.linalg.norm(diff) / np.linalg.norm(r))
    linf = float(np.max(np.abs(diff)))
    flat_r = np.concatenate([r.real, r.imag])
    flat_p = np.concatenate([p.real, p.imag])
    corr = float(np.corrcoef(flat_r, flat_p)[0, 1])
    return {"l2_relative": l2, "linf": linf, "corrcoef": corr}


def pattern_metrics(theta, p_ref, p_pred) -> dict:
    """Main-lobe direction Δ (degrees) + max ΔdB across all angles."""
    deg_diff = float(np.rad2deg(theta[int(np.argmax(p_pred))] - theta[int(np.argmax(p_ref))]))
    side_lobe = float(np.max(np.abs(p_pred - p_ref)))
    return {"main_lobe_deg_diff": deg_diff, "side_lobe_db_diff": side_lobe}
```

- [ ] **Step 4: Run → 3 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/comparator.py examples/radiation_slot/tests/test_comparator.py
git commit -m "feat(radiation_slot): comparator metrics (field + pattern)"
```

---

## Milestone M4: DeepXDE PINN Solver

### Task 12: PINN — domain PDE residual + smoke training

**Files:** Create `pinn_solver.py`, `tests/test_pinn_solver.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_pinn_solver.py
"""Smoke tests only — accuracy is validated end-to-end via main.py."""
import os
os.environ.setdefault("DDE_BACKEND", "pytorch")

import pytest
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.pinn_solver import PINNSolver


def test_smoke_train_50_steps():
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=400, num_boundary=80, num_test=400)
    losshistory, _ = solver.train(iterations=50, lr=1e-3)
    assert len(losshistory.loss_train) >= 1
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

Run: `cd /data/deepxde && pytest examples/radiation_slot/tests/test_pinn_solver.py -v`

- [ ] **Step 3: Implement minimal version (PEC-only BCs; full BCs added in Task 13)**

```python
# examples/radiation_slot/pinn_solver.py
"""DeepXDE PINN solver for the 2D Helmholtz radiation-slot problem.

Outputs y = (u, v) ≈ (Re Ez, Im Ez). Real and imaginary parts share the network
but are decoded by component=0/1.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

import numpy as np

os.environ.setdefault("DDE_BACKEND", "pytorch")
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
        self.k0 = free_space_wavenumber(self.frequency_ghz, eps_r=self.geometry.medium_epsilon)
        self.geom = self.geometry.build_dde_geometry()

    def pde(self, x, y):
        u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
        k2 = self.k0 ** 2 * self.geometry.medium_epsilon
        return [u_xx + u_yy + k2 * y[:, 0:1],
                v_xx + v_yy + k2 * y[:, 1:2]]

    def _build_bcs(self):
        """PEC-only stub. Replaced with full Mur+TE10 set in Task 13."""
        g = self.geometry

        def on_pec(x, on_boundary):
            return on_boundary and g.boundary_marker(x.reshape(1, -1))[0] == "pec"

        return [
            dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=0),
            dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=1),
        ]

    def train(self, iterations: int = 15000, lr: float = 1e-3,
              restore_from: str | None = None, save_path: str | None = None):
        bcs = self._build_bcs()
        loss_weights = (1.0, 1.0) + tuple(100.0 for _ in bcs)
        data = dde.data.PDE(
            self.geom, self.pde, bcs,
            num_domain=self.num_domain,
            num_boundary=self.num_boundary,
            num_test=self.num_test,
        )
        net = dde.nn.FNN(
            [2] + list(self.hidden_layers) + [2],
            self.activation, self.initializer,
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
```

- [ ] **Step 4: Run → 1 passed (10–60 s)**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/pinn_solver.py examples/radiation_slot/tests/test_pinn_solver.py
git commit -m "feat(radiation_slot): PINN scaffold + PEC BCs + smoke test"
```

---

### Task 13: PINN — full Mur ABC (4 directions) + TE10 port_in

**Files:** Modify `pinn_solver.py`, `tests/test_pinn_solver.py`.

- [ ] **Step 1: Append failing test**

```python
def test_build_bcs_returns_full_set():
    """PEC re/im (2) + port_in re/im (2) + port_out re/im (2) + radiation re/im (2) = 8."""
    g = RadiationSlotGeometry()
    solver = PINNSolver(g, frequency_ghz=15.0, num_domain=10, num_boundary=10, num_test=10)
    bcs = solver._build_bcs()
    assert len(bcs) == 8
    classes = {type(b).__name__ for b in bcs}
    assert "DirichletBC" in classes
    assert "RobinBC" in classes
```

- [ ] **Step 2: Run → FAIL (current `_build_bcs` returns 2)**

- [ ] **Step 3: Replace `_build_bcs` in `pinn_solver.py`**

```python
    def _build_bcs(self):
        """PEC + port_in (TE10 + Mur) + port_out (Mur) + radiation (Mur) RobinBCs.

        Mur on outward normal n̂:  ∂Ez/∂n + j k0 Ez = 0
            → ∂u/∂n =  k0 v
              ∂v/∂n = -k0 u

        port_in (outward normal n̂ = -x̂) carries TE10 incident:
            ∂Ez/∂x − j k0 Ez = -2 j k0 sin(πy/b)
            → in outward-normal form (∂/∂n = -∂/∂x):
              ∂u/∂n = -k0 v
              ∂v/∂n =  k0 u + 2 k0 sin(πy/b)
        """
        import numpy as _np
        g = self.geometry
        k0 = self.k0
        b_h = g.waveguide_height

        def label_of(x):
            return g.boundary_marker(_np.atleast_2d(x))[0]

        def on_pec(x, on_b):       return on_b and label_of(x) == "pec"
        def on_port_in(x, on_b):   return on_b and label_of(x) == "port_in"
        def on_port_out(x, on_b):  return on_b and label_of(x) == "port_out"
        def on_radiation(x, on_b): return on_b and label_of(x) == "radiation"

        # PEC
        pec_re = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=0)
        pec_im = dde.icbc.DirichletBC(self.geom, lambda x: 0.0, on_pec, component=1)

        # port_in TE10 + Mur (outward normal -x̂)
        def port_in_re(x, y):
            return -k0 * y[:, 1:2]

        def port_in_im(x, y):
            # x is a backend tensor here. dde.backend.sin handles it.
            return k0 * y[:, 0:1] + 2.0 * k0 * dde.backend.sin(
                dde.backend.as_tensor(_np.pi / b_h) * x[:, 1:2]
            )

        port_in_bc_re = dde.icbc.RobinBC(self.geom, port_in_re, on_port_in, component=0)
        port_in_bc_im = dde.icbc.RobinBC(self.geom, port_in_im, on_port_in, component=1)

        # port_out + radiation: ∂u/∂n = k0 v, ∂v/∂n = -k0 u
        def mur_re(x, y): return  k0 * y[:, 1:2]
        def mur_im(x, y): return -k0 * y[:, 0:1]

        port_out_re = dde.icbc.RobinBC(self.geom, mur_re, on_port_out, component=0)
        port_out_im = dde.icbc.RobinBC(self.geom, mur_im, on_port_out, component=1)
        rad_re = dde.icbc.RobinBC(self.geom, mur_re, on_radiation, component=0)
        rad_im = dde.icbc.RobinBC(self.geom, mur_im, on_radiation, component=1)

        return [pec_re, pec_im,
                port_in_bc_re, port_in_bc_im,
                port_out_re, port_out_im,
                rad_re, rad_im]
```

> If `dde.backend.sin` / `as_tensor` calls raise on PyTorch backend, fall back to a per-backend dispatch (mirror the pattern in `examples/pinn_forward/Helmholtz_Neumann_2d_hole.py` lines 27–38).

- [ ] **Step 4: Run → 2 passed**

Run: `cd /data/deepxde && pytest examples/radiation_slot/tests/test_pinn_solver.py -v`

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/pinn_solver.py examples/radiation_slot/tests/test_pinn_solver.py
git commit -m "feat(radiation_slot): full Mur+TE10 RobinBC set for PINN"
```

---

### Task 14: PINN — `predict_on_grid()` + L-BFGS fine-tune

**Files:** Modify `pinn_solver.py`, `tests/test_pinn_solver.py`.

- [ ] **Step 1: Append failing test**

```python
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
```

- [ ] **Step 2: Run → 2 new failures (`AttributeError`)**

- [ ] **Step 3: Append to `pinn_solver.py`**

```python
    def predict_on_grid(self, X, Y, mask) -> np.ndarray:
        """Predict Ez on a Cartesian grid; outside-mask cells set to NaN."""
        pts = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
        y = self.model.predict(pts)
        u = y[:, 0].reshape(X.shape)
        v = y[:, 1].reshape(X.shape)
        Ez = (u + 1j * v).astype(np.complex128)
        Ez[~mask] = np.nan
        return Ez

    def finetune_lbfgs(self, max_iter: int = 5000):
        """Switch optimizer to L-BFGS and train until convergence."""
        dde.optimizers.config.set_LBFGS_options(maxiter=max_iter)
        self.model.compile("L-BFGS")
        return self.model.train()
```

- [ ] **Step 4: Run → 4 passed**

- [ ] **Step 5: Commit**

```bash
git add examples/radiation_slot/pinn_solver.py examples/radiation_slot/tests/test_pinn_solver.py
git commit -m "feat(radiation_slot): PINN predict_on_grid + L-BFGS finetune"
```

---

## Milestone M5: End-to-End Single-Frequency MVP

### Task 15: `main.py` single-frequency mode + acceptance script

**Files:** Create `main.py`, `tests/test_main_smoke.py`.

- [ ] **Step 1: Write the failing test (cli smoke only — full validation is manual)**

```python
# examples/radiation_slot/tests/test_main_smoke.py
"""Smoke test: import main.py, build args parser, dispatch to a no-op."""
import sys
import importlib


def test_main_module_imports():
    # No side effects on import
    mod = importlib.import_module("examples.radiation_slot.main")
    assert hasattr(mod, "build_argparser")
    assert hasattr(mod, "run_single_frequency")


def test_argparser_defaults():
    mod = importlib.import_module("examples.radiation_slot.main")
    parser = mod.build_argparser()
    args = parser.parse_args(["--mode", "single", "--freq", "15.0"])
    assert args.mode == "single"
    assert args.freq == 15.0
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

- [ ] **Step 3: Implement `main.py`**

```python
# examples/radiation_slot/main.py
"""CLI entry point for radar radiation-slot simulation.

Modes:
    single  — run one frequency, full FDFD vs PINN comparison
    scan    — sweep 12–18 GHz at 0.5 GHz steps with PINN warm-starts (M6)
"""
from __future__ import annotations
import argparse
import os
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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).parent / "conf" / "radiation_slot.yaml"
DEFAULT_OUT = Path(__file__).parent / "outputs"


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["single", "scan"], default="single")
    p.add_argument("--freq", type=float, default=15.0, help="GHz, used in single mode")
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUT))
    p.add_argument("--iters", type=int, default=15000)
    p.add_argument("--seed", type=int, default=0)
    return p


def run_single_frequency(freq_ghz: float, cfg: dict, outdir: Path, iters: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    g = RadiationSlotGeometry(**cfg["geometry"])

    print(f"[1/4] FDFD reference at {freq_ghz} GHz ...")
    fdfd_out = FDFDSolver(g, frequency_ghz=freq_ghz,
                          mesh_size=cfg["fdfd"]["mesh_size"]).solve()

    print(f"[2/4] PINN training ({iters} Adam iters + L-BFGS) ...")
    # Lazy import — keeps this script importable without DeepXDE backend cost.
    from examples.radiation_slot.pinn_solver import PINNSolver
    pinn = PINNSolver(g, frequency_ghz=freq_ghz,
                      num_domain=cfg["pinn"]["num_domain"],
                      num_boundary=cfg["pinn"]["num_boundary"],
                      num_test=cfg["pinn"]["num_test"])
    pinn.train(iterations=iters, lr=cfg["pinn"]["adam_lr"])
    pinn.finetune_lbfgs(max_iter=cfg["pinn"]["lbfgs_max_iter"])

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
    pat_p_db = 20 * np.log10(np.abs(pat_p) / np.max(np.abs(pat_p)))
    pat_metrics = pattern_metrics(theta, pat_f_db, pat_p_db)

    plot_field(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / f"field_fdfd_{freq_ghz}GHz.png"), title=f"FDFD |Ez| @ {freq_ghz} GHz")
    plot_field(Ez_pinn,        fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"],
               str(outdir / f"field_pinn_{freq_ghz}GHz.png"), title=f"PINN |Ez| @ {freq_ghz} GHz")
    plot_radiation_pattern(theta, pat_f_db, str(outdir / f"pattern_fdfd_{freq_ghz}GHz.png"),
                           title=f"FDFD pattern @ {freq_ghz} GHz")
    plot_radiation_pattern(theta, pat_p_db, str(outdir / f"pattern_pinn_{freq_ghz}GHz.png"),
                           title=f"PINN pattern @ {freq_ghz} GHz")

    summary = {
        "frequency_ghz": freq_ghz,
        "field": metrics,
        "pattern": pat_metrics,
        "S11_fdfd_dB": 20 * np.log10(max(abs(s11_f), 1e-12)),
        "S11_pinn_dB": 20 * np.log10(max(abs(s11_p), 1e-12)),
        "S21_fdfd_dB": 20 * np.log10(max(abs(s21_f), 1e-12)),
        "S21_pinn_dB": 20 * np.log10(max(abs(s21_p), 1e-12)),
    }
    with open(outdir / f"summary_{freq_ghz}GHz.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    print("Summary:", summary)
    return summary


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config)
    np.random.seed(args.seed)
    if args.mode == "single":
        run_single_frequency(args.freq, cfg, Path(args.outdir), args.iters)
    elif args.mode == "scan":
        # Implemented in Task 16
        from examples.radiation_slot.main_scan import run_scan
        run_scan(cfg, Path(args.outdir), args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run → 2 passed**

- [ ] **Step 5: Manual end-to-end validation (NOT a unit test — do this by hand)**

```bash
cd /data/deepxde
python -m examples.radiation_slot.main --mode single --freq 15.0 --iters 15000
```

Read `examples/radiation_slot/outputs/summary_15.0GHz.yaml`. **Acceptance gates** (from spec §4.2):

- `field.l2_relative` ≤ **0.05**
- `field.linf` ≤ **0.15 × max|E_fdfd|**
- `field.corrcoef` ≥ **0.98**
- `pattern.main_lobe_deg_diff` absolute value ≤ **2.0**
- `|S11_fdfd_dB - S11_pinn_dB|` ≤ **2.0**

If any gate fails → invoke `superpowers:systematic-debugging` (do not silently relax thresholds).

- [ ] **Step 6: Commit**

```bash
git add examples/radiation_slot/main.py examples/radiation_slot/tests/test_main_smoke.py
git commit -m "feat(radiation_slot): main.py single-frequency end-to-end pipeline"
```

---

## Milestone M6: Frequency Sweep (Optional Extension)

### Task 16: 13-point Ku-band scan with PINN warm-starts + S-param plot

**Files:** Create `main_scan.py`, `tests/test_main_scan_smoke.py`.

- [ ] **Step 1: Write the failing test**

```python
# examples/radiation_slot/tests/test_main_scan_smoke.py
import importlib
def test_main_scan_imports():
    mod = importlib.import_module("examples.radiation_slot.main_scan")
    assert hasattr(mod, "run_scan")
    assert hasattr(mod, "plot_s_param_curve")
```

- [ ] **Step 2: Run → `ModuleNotFoundError`**

- [ ] **Step 3: Implement**

```python
# examples/radiation_slot/main_scan.py
"""13-point Ku-band scan (12–18 GHz @ 0.5 GHz). Uses PINN warm-starts."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml

from examples.radiation_slot.fdfd_solver import FDFDSolver
from examples.radiation_slot.functions import free_space_wavenumber
from examples.radiation_slot.geometry import RadiationSlotGeometry
from examples.radiation_slot.postprocess import s_parameters


def plot_s_param_curve(freqs, s11_db_fdfd, s21_db_fdfd, s11_db_pinn, s21_db_pinn, save_path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, s11_db_fdfd, "-o", label="|S11| FDFD")
    ax.plot(freqs, s11_db_pinn, "--x", label="|S11| PINN")
    ax.plot(freqs, s21_db_fdfd, "-s", label="|S21| FDFD")
    ax.plot(freqs, s21_db_pinn, "--+", label="|S21| PINN")
    ax.set_xlabel("Frequency [GHz]"); ax.set_ylabel("dB")
    ax.set_title("S-parameters vs Frequency")
    ax.legend(); ax.grid(True)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def run_scan(cfg: dict, outdir: Path, iters: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    g = RadiationSlotGeometry(**cfg["geometry"])
    fs = cfg["frequencies"]
    freqs = np.arange(fs["scan_start_ghz"],
                      fs["scan_stop_ghz"] + 1e-9,
                      fs["scan_step_ghz"])

    s11_db_fdfd, s21_db_fdfd, s11_db_pinn, s21_db_pinn = [], [], [], []
    last_ckpt = None

    from examples.radiation_slot.pinn_solver import PINNSolver

    for i, f in enumerate(freqs):
        print(f"\n=== {i+1}/{len(freqs)}: {f:.1f} GHz ===")
        fdfd_out = FDFDSolver(g, frequency_ghz=float(f),
                              mesh_size=cfg["fdfd"]["mesh_size"]).solve()
        k0 = free_space_wavenumber(float(f), eps_r=g.medium_epsilon)
        s11_f, s21_f = s_parameters(fdfd_out["Ez"], fdfd_out["X"], fdfd_out["Y"], g, k0)

        # Warm-start: first freq full Adam iters, subsequent ones reduced + restore.
        n_iters = iters if i == 0 else max(2000, iters // 4)
        save_path = str(outdir / f"ckpt_{f:.1f}GHz")
        pinn = PINNSolver(g, frequency_ghz=float(f),
                          num_domain=cfg["pinn"]["num_domain"],
                          num_boundary=cfg["pinn"]["num_boundary"],
                          num_test=cfg["pinn"]["num_test"])
        pinn.train(iterations=n_iters, lr=cfg["pinn"]["adam_lr"],
                   restore_from=last_ckpt, save_path=save_path)
        pinn.finetune_lbfgs(max_iter=cfg["pinn"]["lbfgs_max_iter"])
        last_ckpt = save_path + f"-{n_iters}"  # DeepXDE appends iter suffix

        Ez_p = pinn.predict_on_grid(fdfd_out["X"], fdfd_out["Y"], fdfd_out["mask"])
        s11_p, s21_p = s_parameters(Ez_p, fdfd_out["X"], fdfd_out["Y"], g, k0)

        for store, val in [
            (s11_db_fdfd, s11_f), (s21_db_fdfd, s21_f),
            (s11_db_pinn, s11_p), (s21_db_pinn, s21_p),
        ]:
            store.append(20 * np.log10(max(abs(val), 1e-12)))

    plot_s_param_curve(freqs, s11_db_fdfd, s21_db_fdfd,
                       s11_db_pinn, s21_db_pinn,
                       str(outdir / "s_parameters_scan.png"))
    summary = {
        "frequencies_ghz": freqs.tolist(),
        "S11_fdfd_dB": s11_db_fdfd, "S21_fdfd_dB": s21_db_fdfd,
        "S11_pinn_dB": s11_db_pinn, "S21_pinn_dB": s21_db_pinn,
    }
    with open(outdir / "scan_summary.yaml", "w") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)
    return summary
```

- [ ] **Step 4: Run → 1 passed**

- [ ] **Step 5: Manual end-to-end (LONG: ~30 min on GPU)**

```bash
cd /data/deepxde
python -m examples.radiation_slot.main --mode scan --iters 15000
```

Verify `outputs/s_parameters_scan.png` is monotonic-ish (no spikes from training collapses) and `outputs/scan_summary.yaml` has 13 entries.

**Acceptance:** `max(|S11_fdfd_dB - S11_pinn_dB|)` over the 13 points ≤ **3.0 dB** (looser than single-freq because warm-start may degrade some points).

- [ ] **Step 6: Commit**

```bash
git add examples/radiation_slot/main_scan.py examples/radiation_slot/tests/test_main_scan_smoke.py
git commit -m "feat(radiation_slot): 13-point Ku-band scan with warm-starts"
```

---

## Final: Documentation Update

### Task 17: Update top-level docs to mark all todos as completed

**Files:** Modify `docs/demos/radar-radiation/radar-radiation-slot-simulation.md`.

- [ ] **Step 1: Flip all 6 todos in the front-matter from `pending` to `completed` and run a final pytest**

```bash
cd /data/deepxde && pytest examples/radiation_slot/tests/ -v
```

Expected: ~17 passed (4 functions + 5 geometry + 4 fdfd + 4 postprocess + 3 comparator + 3 pinn + 2 main + 1 scan ≈ 26).

- [ ] **Step 2: Edit `docs/demos/radar-radiation/radar-radiation-slot-simulation.md`** front-matter, change every `status: pending` to `status: completed`.

- [ ] **Step 3: Commit**

```bash
git add docs/demos/radar-radiation/radar-radiation-slot-simulation.md
git commit -m "docs(radiation_slot): mark all todos completed; tests green"
```

---

## Self-Review Checklist (run after writing the plan)

**1. Spec coverage** — every spec section maps to a task:
- §1 Architecture overview → Tasks 2 (CSG), 6 (FDFD assembly), 12–13 (PINN)
- §2 Modules & interfaces → all 6 module files created across Tasks 1–14
- §3 Math (PDE, FDFD discretization, PINN residual, S-param, NTFF) → Tasks 6, 8, 9, 12
- §4 Tests (unit + e2e) → Tasks 2–11 cover unit tests; Task 15 step 5 covers e2e gates
- §5 Milestones M1–M6 → Tasks 1–5 (M1), 6–7 (M2), 8–11 (M3), 12–14 (M4), 15 (M5), 16 (M6)
- §6 Risks → mitigation hooks present (loss_weights tunable in conf YAML; backend swap via env var)
- §7 YAGNI exclusions → no PML, no Hydra, no VTU, no 3D — confirmed absent

**2. Placeholder scan** — searched for "TBD/TODO/handle edge cases/similar to" → none found.

**3. Type consistency**
- `RadiationSlotGeometry` fields identical across Tasks 2/3/4/12/15
- `boundary_marker(...)` returns `np.ndarray` of `dtype=object` strings — used as `labels[i] == "pec"` in Tasks 6/13 ✓
- `FDFDSolver.solve()` returns dict keys `{X, Y, Ez, mask, k0}` — consumed by Tasks 7/8/15 ✓
- `s_parameters(Ez, X, Y, geometry, k0)` signature identical in Tasks 8/15/16 ✓
- `near_to_far_field(Ez_ap, x_ap, k0, theta)` signature identical in Tasks 9/15 ✓
- `PINNSolver.predict_on_grid(X, Y, mask)` returns complex (Ny, Nx) — consumed by Task 15 ✓

All consistent. No issues to fix.

---

## Execution Handoff

Plan complete. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance, then code quality). Tight feedback loop, best for learning the system.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for human review.

**Which approach?**
