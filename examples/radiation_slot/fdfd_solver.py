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

        def idx(j, i):
            return j * nx + i

        A = lil_matrix((N, N), dtype=np.complex128)
        b_rhs = np.zeros(N, dtype=np.complex128)

        for j in range(ny):
            for i in range(nx):
                p = idx(j, i)
                xv, yv = X[j, i], Y[j, i]

                # Outside the CSG domain → identity row, RHS 0
                if not mask[j, i]:
                    A[p, p] = 1.0
                    continue

                # PEC: waveguide bottom (y=0) + waveguide top (y=wg_h) excluding
                # the slot opening
                on_y0 = np.isclose(yv, 0.0, atol=atol)
                on_yh = np.isclose(yv, wg_h, atol=atol)
                in_slot_x = (xv >= slot_lo - atol) and (xv <= slot_hi + atol)
                if on_y0 or (on_yh and (g.slot_length == 0.0 or not in_slot_x)):
                    A[p, p] = 1.0
                    continue

                # Port_in (x=0): TE10 + 1st-order Mur
                #     ∂Ez/∂x − jk0 Ez = -2jk0 sin(πy/b)
                # Backward diff: (Ez(i+1) − Ez(i))/h − jk0 Ez(i) = rhs
                if np.isclose(xv, 0.0, atol=atol) and yv <= wg_h + atol:
                    A[p, p] = -1.0 / h - 1j * k0
                    A[p, idx(j, i + 1)] = 1.0 / h
                    b_rhs[p] = -2j * k0 * te10_profile(np.array([yv]), b=wg_h)[0]
                    continue

                # Port_out (x=L): 1st-order Mur out
                #     ∂Ez/∂x + jk0 Ez = 0
                # Forward diff: (Ez(i) − Ez(i-1))/h + jk0 Ez(i) = 0
                if np.isclose(xv, g.waveguide_width, atol=atol) and yv <= wg_h + atol:
                    A[p, p] = 1.0 / h + 1j * k0
                    A[p, idx(j, i - 1)] = -1.0 / h
                    continue

                # Radiation (buffer top y=buf_top, buffer side x=slot_lo/slot_hi)
                on_buf_top = np.isclose(yv, buf_top, atol=atol)
                on_buf_left = np.isclose(xv, slot_lo, atol=atol) and yv > wg_h + atol
                on_buf_right = np.isclose(xv, slot_hi, atol=atol) and yv > wg_h + atol
                if on_buf_top:
                    # outward normal +y; ∂Ez/∂y + jk0 Ez = 0
                    A[p, p] = 1.0 / h + 1j * k0
                    A[p, idx(j - 1, i)] = -1.0 / h
                    continue
                if on_buf_left:
                    # outward normal -x; -∂Ez/∂x + jk0 Ez = 0  →  ∂Ez/∂x − jk0 Ez = 0
                    A[p, p] = -1.0 / h - 1j * k0
                    A[p, idx(j, i + 1)] = 1.0 / h
                    continue
                if on_buf_right:
                    A[p, p] = 1.0 / h + 1j * k0
                    A[p, idx(j, i - 1)] = -1.0 / h
                    continue

                # Interior 5-point Helmholtz stencil
                A[p, p] = -4.0 / h ** 2 + k0 ** 2 * g.medium_epsilon
                A[p, idx(j, i + 1)] = 1.0 / h ** 2
                A[p, idx(j, i - 1)] = 1.0 / h ** 2
                A[p, idx(j + 1, i)] = 1.0 / h ** 2
                A[p, idx(j - 1, i)] = 1.0 / h ** 2

        Ez = spsolve(A.tocsr(), b_rhs).reshape(ny, nx)
        return {"X": X, "Y": Y, "Ez": Ez, "mask": mask, "k0": k0}
