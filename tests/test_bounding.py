import numpy as np
import pytest

from pyshape import analyze_mesh, min_volume_box, oriented_box, pca_box

pytest.importorskip("scipy")


def _rotation(seed):
    """Deterministic proper rotation matrix from a QR decomposition."""
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def _box_points(a, b, c):
    corners = np.array([[x, y, z] for x in (0, a) for y in (0, b) for z in (0, c)], float)
    return corners


TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


@pytest.mark.parametrize("seed", range(5))
def test_min_volume_box_of_rotated_cuboid_is_exact(seed):
    points = _box_points(1.0, 2.0, 3.0) @ _rotation(seed).T
    box = min_volume_box(points)
    assert (box.short, box.intermediate, box.long) == pytest.approx((1, 2, 3), rel=1e-9)
    assert box.volume == pytest.approx(6.0, rel=1e-9)
    np.testing.assert_allclose(box.directions @ box.directions.T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_min_volume_box_of_regular_tetrahedron_is_enclosing_cube(seed):
    # The optimum (the cube on whose corners the tetrahedron sits) is neither
    # facet- nor edge-aligned; it needs the edge-pair candidates.
    box = min_volume_box(TETRA @ _rotation(seed).T)
    assert (box.short, box.intermediate, box.long) == pytest.approx((2, 2, 2), rel=1e-9)


def test_pca_box_is_not_unique_for_isotropic_shapes():
    spans = []
    for seed in range(5):
        box = pca_box(_box_points(1.0, 1.0, 1.0) @ _rotation(seed).T)
        spans.append(box.long)
    # Documented limitation: the cube vertex covariance is isotropic.
    assert np.ptp(spans) > 1e-3


@pytest.mark.parametrize("seed", range(3))
def test_pca_box_matches_known_axes_for_anisotropic_shape(seed):
    # Symmetric anisotropic point set: principal directions are the box edges.
    box = pca_box(_box_points(1.0, 2.0, 3.0) @ _rotation(seed).T)
    assert (box.short, box.intermediate, box.long) == pytest.approx((1, 2, 3), rel=1e-9)


def test_box_contains_points_and_center():
    points = np.random.default_rng(3).normal(size=(300, 3)) * [1, 2, 3]
    for method in ("pca", "min_volume"):
        box = oriented_box(points, method)
        local = (points - box.center) @ box.directions.T
        half = np.array([box.short, box.intermediate, box.long]) / 2
        assert np.all(np.abs(local) <= half + 1e-9)
    assert min_volume_box(points).volume <= pca_box(points).volume + 1e-12


def test_unknown_axes_method():
    with pytest.raises(ValueError, match="Unknown axes method"):
        oriented_box(TETRA, "feret")


def test_analysis_with_min_volume_axes(cube_mesh):
    nodes, faces = cube_mesh
    rotated = nodes @ _rotation(7).T
    result = analyze_mesh(rotated, faces, axes="min_volume")
    assert result.axes_method == "min_volume"
    values = result.values
    assert (values["axis_S"], values["axis_I"], values["axis_L"]) == pytest.approx((1, 1, 1))
    assert values["sphericity_krumbein"] == pytest.approx(1)
    assert values["zingg_SI"] == pytest.approx(1)
    assert values["angelidakis_flatness"] == pytest.approx(0, abs=1e-12)
    assert next(m.group for m in result.metrics if m.key == "axis_S") == "Axes (minimum-volume box)"
    assert analyze_mesh(rotated, faces).axes_method == "pca"
    with pytest.raises(ValueError, match="Unknown axes method"):
        analyze_mesh(nodes, faces, axes="feret")


def test_min_volume_axes_need_scipy(cube_mesh, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    with pytest.raises(ImportError, match=r"pyshape\[analysis\]"):
        analyze_mesh(*cube_mesh, axes="min_volume")
