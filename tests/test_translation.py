"""No result depends on where the particle is (issue #32)."""
from pathlib import Path

import numpy as np
import pytest

from pyshape import (
    analyze_mesh, fitted_ellipsoid, inertia_ellipsoid, load_geometry, min_volume_box, pca_box,
    prepare_mesh,
)

SYNTHETIC = Path(__file__).resolve().parents[1] / "examples" / "assets" / "synthetic"
DIRECTION = np.array([0.6, -0.48, 0.64])  # unit vector
INDICES = ("sphericity_krumbein", "zingg_SI", "zingg_IL", "potticary_flatness",
           "potticary_elongation", "kongfonseca_flatness", "kongfonseca_elongation",
           "angelidakis_flatness", "angelidakis_elongation")


def mesh(name):
    geometry = load_geometry(SYNTHETIC / f"{name}.stl")
    return prepare_mesh(geometry.nodes, geometry.faces)


@pytest.mark.parametrize("axes", ["pca", "min_volume", "inertia_ellipsoid", "fitted_ellipsoid"])
@pytest.mark.parametrize("name", ["rough_particle", "ellipsoid_3_2_1"])
def test_results_do_not_depend_on_position(name, axes):
    if axes == "min_volume":
        pytest.importorskip("scipy")
    nodes, faces = mesh(name)
    reference = analyze_mesh(nodes, faces, axes=axes).values
    size = float(np.ptp(nodes, axis=0).max())
    for offset in (1e5,):
        moved = analyze_mesh(nodes + DIRECTION * offset * size, faces, axes=axes).values
        for key, value in reference.items():
            if value is None:
                assert moved[key] is None, key
            elif key.startswith("centroid_"):
                assert moved[key] == pytest.approx(
                    value + DIRECTION["xyz".index(key[-1])] * offset * size, rel=1e-12)
            elif key in INDICES:
                assert moved[key] == pytest.approx(value, abs=1e-9), key
            else:
                assert moved[key] == pytest.approx(value, rel=1e-9), key


def test_shapes_are_centred_on_the_particle():
    pytest.importorskip("scipy")
    nodes, faces = mesh("ellipsoid_3_2_1")  # semi-axes 3, 2, 1, centred at (1, -2, 0.5)
    centre = np.array([1.0, -2.0, 0.5])
    shapes = (pca_box(nodes), min_volume_box(nodes), inertia_ellipsoid(nodes, faces),
              fitted_ellipsoid(nodes, faces))
    for shape in shapes:
        np.testing.assert_allclose(shape.center, centre, atol=2e-3)
    shift = DIRECTION * 1e4
    for shape, moved in zip(shapes, (pca_box(nodes + shift), min_volume_box(nodes + shift),
                                     inertia_ellipsoid(nodes + shift, faces),
                                     fitted_ellipsoid(nodes + shift, faces))):
        np.testing.assert_allclose(moved.center - shift, shape.center, atol=1e-9)


def test_inertia_ellipsoid_is_centred_on_the_barycentre():
    nodes, faces = mesh("rough_particle")
    values = analyze_mesh(nodes, faces).values
    barycentre = [values[f"centroid_{axis}"] for axis in "xyz"]
    np.testing.assert_allclose(inertia_ellipsoid(nodes, faces).center, barycentre, atol=1e-12)
    assert np.linalg.norm(barycentre) > 0.05  # not at the origin


def test_boxes_touch_the_particle_on_every_face():
    pytest.importorskip("scipy")
    nodes, _ = mesh("rough_particle")
    for box in (pca_box(nodes + 50.0), min_volume_box(nodes + 50.0)):
        half = np.array([box.short, box.intermediate, box.long]) / 2
        projected = (nodes + 50.0 - box.center) @ box.directions.T
        np.testing.assert_allclose(projected.min(axis=0), -half, atol=1e-9)
        np.testing.assert_allclose(projected.max(axis=0), half, atol=1e-9)
