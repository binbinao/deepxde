"""Geometry, uniform FDFD grid, and boundary marker for the radiation-slot demo."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import deepxde as dde


@dataclass
class RadiationSlotGeometry:
    """All lengths in cm. 2D domain = waveguide ∪ buffer (above the slot opening).

    Dimension naming (post M2 review):
        ``waveguide_width``  is the length along the propagation direction (x in 2D).
        ``waveguide_height`` is the cross-section dimension that determines TE10
        cutoff. For TE10 to propagate at frequency f, we need
        ``waveguide_height > c/(2 f) = 1 cm at 15 GHz``. Default 1.5 cm gives
        kx ≈ 2.34 rad/cm and a guided wavelength λ_g ≈ 2.68 cm at 15 GHz, so a
        4 cm long waveguide hosts ~1.5 periods — enough to see standing-wave
        structure produced by the slot.
    """
    waveguide_width: float = 4.0
    waveguide_height: float = 1.5
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
        wg = dde.geometry.Rectangle(
            [0.0, 0.0], [self.waveguide_width, self.waveguide_height]
        )
        if self.slot_length <= 0 or self.buffer_height <= 0:
            return wg
        x_lo, x_hi = self.slot_x_range()
        buf = dde.geometry.Rectangle(
            [x_lo, self.waveguide_height],
            [x_hi, self.waveguide_height + self.buffer_height],
        )
        return wg | buf

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
        in_buf = (
            (Y > self.waveguide_height + 1e-12)
            & (X >= x_lo - 1e-12)
            & (X <= x_hi + 1e-12)
        )
        return X, Y, in_wg | in_buf

    def boundary_marker(self, x: np.ndarray, atol: float = 1e-9) -> np.ndarray:
        """Classify points (N, 2) into one of:
        ``{'port_in', 'port_out', 'pec', 'radiation', 'interior'}``.

        Order of precedence (later wins):
            1. PEC walls
            2. Radiation walls (buffer side + buffer top + slot-edge corners on
               the waveguide top, where the field is *not* bounded by metal)
            3. Ports (so the ports win over PEC at the bottom/top inlet corners)

        With this ordering every boundary point of the CSG domain falls in exactly
        one of {port_in, port_out, pec, radiation}, and `interior` is reserved
        only for non-boundary inputs.
        """
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

        # 1. PEC: waveguide bottom + waveguide top excluding the slot opening
        bottom = on_y0 & (x[:, 0] >= -atol) & (x[:, 0] <= wg_w + atol)
        top_left = on_yh & (x[:, 0] < slot_lo - atol)
        top_right = on_yh & (x[:, 0] > slot_hi + atol)
        labels[bottom | top_left | top_right] = "pec"

        # 2. Radiation:
        #    a) buffer top wall (y = buf_top, slot_lo ≤ x ≤ slot_hi)
        #    b) buffer side walls (x = slot_lo / slot_hi, wg_h ≤ y ≤ buf_top)
        #       — include y = wg_h so the slot-edge corners belong to "radiation"
        #         instead of leaking into "interior".
        on_buf_left = np.isclose(x[:, 0], slot_lo, atol=atol) & (x[:, 1] >= wg_h - atol) & (x[:, 1] <= buf_top + atol)
        on_buf_right = np.isclose(x[:, 0], slot_hi, atol=atol) & (x[:, 1] >= wg_h - atol) & (x[:, 1] <= buf_top + atol)
        on_buf_top = on_yt & in_buf_x
        labels[on_buf_left | on_buf_right | on_buf_top] = "radiation"

        # 3. Ports — paint LAST so they win at the inlet/outlet corners that
        #    PEC would otherwise have claimed.
        labels[on_x0 & in_wg_y] = "port_in"
        labels[on_xL & in_wg_y] = "port_out"

        return labels
