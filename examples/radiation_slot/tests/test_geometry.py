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
