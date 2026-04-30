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


def test_fdfd_grid_shape_and_mask():
    g = RadiationSlotGeometry()
    X, Y, mask = g.fdfd_grid(mesh_size=0.05)
    assert X.shape == Y.shape == mask.shape
    assert X.shape[1] == int(round(g.waveguide_width / 0.05)) + 1
    assert X.shape[0] == int(round((g.waveguide_height + g.buffer_height) / 0.05)) + 1
    area_grid = mask.sum() * 0.05 * 0.05
    area_true = g.waveguide_width * g.waveguide_height + g.slot_length * g.buffer_height
    # Allow ~10% tolerance: discrete cells include slot edges + grid does not
    # divide waveguide_height (0.51) evenly at mesh_size 0.05.
    assert area_grid == pytest.approx(area_true, rel=0.10)


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
    # Every random boundary point must land in one of the four boundary
    # categories — none should be classified as 'interior'. This is the real
    # disjointness/coverage check (a single label cell can never simultaneously
    # equal two distinct strings, so no per-cell intersection is needed).
    assert (labels != "interior").all()


def test_boundary_marker_corners_resolve_to_ports():
    """Inlet/outlet corners must be classified as ports (not PEC)."""
    g = RadiationSlotGeometry()
    corners = np.array([
        [0.0, 0.0], [0.0, g.waveguide_height],
        [g.waveguide_width, 0.0], [g.waveguide_width, g.waveguide_height],
    ])
    labels = g.boundary_marker(corners).tolist()
    assert labels == ["port_in", "port_in", "port_out", "port_out"]


def test_boundary_marker_slot_edge_corners_are_radiation():
    """The slot edge corners (slot_lo, wg_h) and (slot_hi, wg_h) must be
    classified as 'radiation' — they bound the open buffer, not metal."""
    g = RadiationSlotGeometry()
    slot_lo, slot_hi = g.slot_x_range()
    edges = np.array([[slot_lo, g.waveguide_height], [slot_hi, g.waveguide_height]])
    assert g.boundary_marker(edges).tolist() == ["radiation", "radiation"]
