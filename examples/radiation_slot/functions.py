"""Shared math utilities. Lengths in cm, frequencies in GHz."""
from __future__ import annotations
import numpy as np

C_CM = 3.0e10  # speed of light in cm/s


def free_space_wavenumber(frequency_ghz: float, eps_r: float = 1.0) -> float:
    """k0 = 2π f √ε_r / c, with f in Hz (= GHz × 1e9). Returns rad/cm."""
    return 2.0 * np.pi * frequency_ghz * 1e9 * np.sqrt(eps_r) / C_CM


def te10_profile(y: np.ndarray, b: float) -> np.ndarray:
    """Normalized TE10 transverse profile sin(πy/b) on [0, b]."""
    return np.sin(np.pi * y / b)


def te10_norm(b: float) -> float:
    """∫₀^b sin²(πy/b) dy = b/2."""
    return b / 2.0
