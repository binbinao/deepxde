"""Comparison metrics between FDFD reference and PINN prediction."""
from __future__ import annotations
import numpy as np


def field_metrics(ref, pred, mask) -> dict:
    """L2-relative / L∞ / Pearson correlation, evaluated only where mask=True.

    Complex inputs are flattened component-wise (real + imag concatenated) for
    the correlation computation; magnitudes are used for L2 / L∞.
    """
    r, p = ref[mask], pred[mask]
    diff = p - r
    l2 = float(np.linalg.norm(diff) / np.linalg.norm(r))
    linf = float(np.max(np.abs(diff)))
    flat_r = np.concatenate([r.real, r.imag])
    flat_p = np.concatenate([p.real, p.imag])
    corr = float(np.corrcoef(flat_r, flat_p)[0, 1])
    return {"l2_relative": l2, "linf": linf, "corrcoef": corr}


def pattern_metrics(theta, p_ref, p_pred) -> dict:
    """Main-lobe direction Δ (degrees) and max ΔdB across all angles."""
    deg_diff = float(np.rad2deg(theta[int(np.argmax(p_pred))] - theta[int(np.argmax(p_ref))]))
    side_lobe = float(np.max(np.abs(p_pred - p_ref)))
    return {"main_lobe_deg_diff": deg_diff, "side_lobe_db_diff": side_lobe}
