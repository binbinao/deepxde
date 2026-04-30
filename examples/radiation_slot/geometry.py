"""Geometry, uniform FDFD grid, and boundary marker for the radiation-slot demo."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import deepxde as dde


@dataclass
class RadiationSlotGeometry:
    """All lengths in cm. 2D domain = waveguide ∪ buffer (above the slot opening).

    NOTE: ``waveguide_width`` here is the length along the wave propagation
    direction (x in 2D). ``waveguide_height`` is the TE10 cutoff dimension
    (physical b).
    """
    waveguide_width: float = 4.0
    waveguide_height: float = 0.51
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
