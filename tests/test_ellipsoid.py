"""Principal dimensions from the inertia-equivalent and the fitted ellipsoid (#6)."""
import numpy as np
import pytest

from pyshape import analyze_mesh
from pyshape.bounding import AXES_METHODS
from pyshape.ellipsoid import (
    ELLIPSOID_METHODS, fit_ellipsoid, fitted_ellipsoid, inertia_ellipsoid, vertex_areas,
)


def icosphere(subdivisions):
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(v, float) / np.linalg.norm(v) for v in (
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
        (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1))]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    nodes, faces = subdivide(np.array(verts), np.array(faces), subdivisions, project=True)
    return nodes, faces


def subdivide(nodes, faces, levels, project=False):
    """Midpoint subdivision; ``project`` puts new vertices on the unit sphere."""
    verts = [np.asarray(v, float) for v in nodes]
    faces = [tuple(f) for f in faces]
    for _ in range(levels):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (verts[a] + verts[b]) / 2
                verts.append(m / np.linalg.norm(m) if project else m)
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def rotation(a, b, c):
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


CUBE = (np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], float),
        np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                  [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]]))
TETRA = (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
         np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]))


def test_methods_are_registered():
    assert set(ELLIPSOID_METHODS) <= set(AXES_METHODS)
    assert AXES_METHODS[:2] == ("pca", "min_volume")  # defaults and order unchanged


def test_analytic_ellipsoid_recovered_and_converges():
    radii = np.array([3.0, 2.0, 1.0])
    move = rotation(0.3, 1.0, -0.4), np.array([5.0, 1.0, -2.0])
    errors = []
    for level in (2, 3, 4):
        sphere, faces = icosphere(level)
        nodes = (sphere * radii) @ move[0].T + move[1]
        fitted = fitted_ellipsoid(nodes, faces)
        # Vertices lie exactly on the ellipsoid: the fit is exact.
        assert np.allclose([fitted.a, fitted.b, fitted.c], radii, rtol=1e-10)
        assert np.allclose(fitted.center, move[1], atol=1e-10)
        inertia = inertia_ellipsoid(nodes, faces)
        errors.append(np.abs(np.array([inertia.a, inertia.b, inertia.c]) / radii - 1).max())
        assert np.allclose(inertia.center, move[1], atol=1e-10)
        # The long axis lies along the rotated x axis.
        assert abs(abs(inertia.directions[2] @ move[0][:, 0]) - 1) < 1e-6
    assert errors[-1] < 1e-3
    # Second-order convergence: each refinement divides the error by about 4.
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


@pytest.mark.parametrize("method", ELLIPSOID_METHODS)
def test_sphere_has_equal_dimensions(method):
    nodes, faces = icosphere(3)
    values = analyze_mesh(nodes * 2.5, faces, axes=method).values
    assert values["axis_S"] == pytest.approx(values["axis_L"], rel=1e-12)
    assert values["axis_I"] == pytest.approx(values["axis_L"], rel=1e-12)
    assert values["zingg_SI"] == pytest.approx(1.0) and values["zingg_IL"] == pytest.approx(1.0)


@pytest.mark.parametrize("mesh", [CUBE, TETRA], ids=["cube", "tetrahedron"])
def test_platonic_solids_have_equal_dimensions_without_scatter(mesh):
    nodes, faces = mesh
    rng = np.random.default_rng(3)
    for method, (n, f) in (("inertia_ellipsoid", (nodes, faces)),
                           ("fitted_ellipsoid", subdivide(nodes, faces, 3))):
        lengths = []
        for _ in range(5):
            q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
            values = analyze_mesh(n @ q.T + rng.normal(size=3), f, axes=method).values
            lengths.append([values["axis_S"], values["axis_I"], values["axis_L"]])
        lengths = np.array(lengths)
        # No scatter over rotations (unlike the PCA box).
        assert np.ptp(lengths, axis=0).max() < 1e-9 * lengths.max(), method
        # Equal lengths: exact for the inertia ellipsoid (isotropic inertia); for the
        # fit, up to the asymmetry of the triangulation of the subdivided surface.
        tolerance = 1e-9 if method == "inertia_ellipsoid" else 1e-2
        assert np.ptp(lengths[0]) < tolerance * lengths.max(), method


def test_cuboid_inertia_ellipsoid_has_box_ratios():
    nodes, faces = CUBE
    nodes = nodes * [4.0, 2.0, 1.0]
    values = analyze_mesh(nodes, faces, axes="inertia_ellipsoid").values
    factor = np.sqrt(5 / 3)
    assert [values["axis_S"], values["axis_I"], values["axis_L"]] == pytest.approx(
        [factor * 1, factor * 2, factor * 4], rel=1e-12)
    assert values["zingg_SI"] == pytest.approx(0.5) and values["zingg_IL"] == pytest.approx(0.5)


@pytest.mark.parametrize("method", ELLIPSOID_METHODS)
def test_rotation_and_translation_invariance(method):
    sphere, faces = icosphere(3)
    x, y, z = sphere.T
    nodes = sphere * (1 + 0.15 * np.sin(3 * x) * np.cos(2 * y) + 0.1 * z ** 3)[:, None] * [1.5, 1, 0.7]
    base = analyze_mesh(nodes, faces, axes=method).values
    moved = analyze_mesh(nodes @ rotation(0.7, -0.2, 2.1).T + [10, -3, 4], faces, axes=method).values
    for key in ("axis_S", "axis_I", "axis_L", "sphericity_krumbein", "zingg_SI", "zingg_IL"):
        assert moved[key] == pytest.approx(base[key], rel=1e-9), key


def test_fit_is_area_weighted_by_default():
    sphere, faces = icosphere(3)
    nodes = sphere * [3.0, 2.0, 1.0]
    weights = vertex_areas(nodes, faces)
    assert weights.sum() == pytest.approx(
        0.5 * np.linalg.norm(np.cross(nodes[faces[:, 1]] - nodes[faces[:, 0]],
                                      nodes[faces[:, 2]] - nodes[faces[:, 0]]), axis=1).sum())
    # On exact ellipsoid points, weighted and unweighted (SHAPE) fits agree.
    unweighted = fitted_ellipsoid(nodes, faces, weighting="none")
    assert [unweighted.a, unweighted.b, unweighted.c] == pytest.approx([3, 2, 1], rel=1e-10)
    with pytest.raises(ValueError, match="weighting"):
        fitted_ellipsoid(nodes, faces, weighting="volume")


def test_degenerate_fits_are_reported():
    with pytest.raises(ValueError, match="at least nine points"):
        fit_ellipsoid(CUBE[0])
    plane = np.column_stack([np.random.default_rng(0).uniform(size=(50, 2)), np.zeros(50)])
    with pytest.raises(ValueError, match="do not determine a quadric"):
        fit_ellipsoid(plane)
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 24, endpoint=False), np.linspace(-1, 1, 9))
    hyperboloid = np.column_stack([(np.cosh(v) * np.cos(u)).ravel(),
                                   (np.cosh(v) * np.sin(u)).ravel(), np.sinh(v).ravel()])
    with pytest.raises(ValueError, match="not an ellipsoid"):
        fit_ellipsoid(hyperboloid)


def test_unavailable_dimensions_are_explicit_errors():
    # Too few vertices to fit; open surface without volume for the inertia ellipsoid.
    fitted = analyze_mesh(*CUBE, axes="fitted_ellipsoid")
    errors = {m.key: m.error for m in fitted.metrics}
    for key in ("axis_S", "axis_L", "sphericity_krumbein", "zingg_SI", "angelidakis_flatness"):
        assert fitted.values[key] is None and errors[key].startswith("ellipsoid fit failed"), key
    assert fitted.values["volume"] == pytest.approx(1.0)  # other results are unaffected
    open_surface = analyze_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], axes="inertia_ellipsoid")
    error = {m.key: m.error for m in open_surface.metrics}["axis_S"]
    assert open_surface.values["axis_S"] is None and "inertia-equivalent ellipsoid" in error


@pytest.mark.parametrize("method, source", [
    ("inertia_ellipsoid", "inertia-equivalent ellipsoid"), ("fitted_ellipsoid", "fitted ellipsoid")])
def test_groups_and_labels_name_the_method(method, source):
    nodes, faces = icosphere(2)
    result = analyze_mesh(nodes * [2, 1.5, 1], faces, axes=method)
    groups = {m.key: m.group for m in result.metrics}
    assert groups["axis_S"] == f"Principal dimensions ({source})"
    assert groups["zingg_SI"] == f"Form from S, I, L ({source})"
    assert result.axes_method == method
