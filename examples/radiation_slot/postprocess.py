"""Postprocess: S-parameters + NTFF + matplotlib plotters."""
from __future__ import annotations
import numpy as np
from examples.radiation_slot.functions import te10_profile, te10_norm


def _column_indices(X, x_target, atol=1e-9):
    return np.where(np.isclose(X[0, :], x_target, atol=atol))[0]


def s_parameters(Ez, X, Y, geometry, k0) -> tuple[complex, complex]:
    """S11=(1/N)∫(Ez(0,y) - Ez_inc) φ̄ dy ; S21=(1/N)∫Ez(L,y) φ̄ dy.

    Uses np.trapz over the inlet/outlet column restricted to the waveguide
    (0 ≤ y ≤ b). At x=0 the incident wave is exp(-jkx·0)·sin(πy/b) = sin(πy/b).
    """
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


def near_to_far_field(Ez_aperture, x_aperture, k0, theta) -> np.ndarray:
    """2D Stratton-Chu approximation:
        E_far(θ) ∝ ∫ Ez(x) exp(-j k0 x sinθ) dx

    Returns complex pattern of shape (Nθ,).
    """
    sin_t = np.sin(theta)[:, None]
    kernel = np.exp(-1j * k0 * x_aperture[None, :] * sin_t)
    return np.trapz(kernel * Ez_aperture[None, :], x_aperture, axis=1)


def aperture_field(Ez, X, Y, geometry):
    """Slice the field at y = waveguide_height, x ∈ [slot_lo, slot_hi].

    Returns (x_aperture, Ez_aperture).
    """
    b = geometry.waveguide_height
    j = int(np.argmin(np.abs(Y[:, 0] - b)))
    x_lo, x_hi = geometry.slot_x_range()
    cols = np.where((X[j, :] >= x_lo - 1e-9) & (X[j, :] <= x_hi + 1e-9))[0]
    return X[j, cols], Ez[j, cols]


def plot_field(Ez, X, Y, mask, save_path, title="|E_z|"):
    """Heatmap of |Ez| over the CSG domain (cells outside mask shown as NaN)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    amp = np.where(mask, np.abs(Ez), np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    pcm = ax.pcolormesh(X, Y, amp, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="|E_z|")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_radiation_pattern(theta, pattern_db, save_path, title="Radiation pattern"):
    """Polar plot of normalized radiation pattern in dB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(theta, pattern_db)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(-40, max(0, np.max(pattern_db) + 1))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
