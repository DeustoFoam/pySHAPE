from pathlib import Path
import sys

import numpy as np
import pytest

from pyshape import analyze_file, analyze_mesh, io, load_geometry
from pyshape.analysis import prepare_mesh
from pyshape.geometry import surface_area


def two_cubes(cube_mesh, offset, second_scale=1.0):
    nodes, faces = cube_mesh
    # Deliberately reverse the second solid, including one inconsistent face.
    second_faces = faces[:, ::-1].copy()
    second_faces[0] = second_faces[0, ::-1]
    return (
        np.vstack((nodes, nodes * second_scale + offset)),
        np.vstack((faces, second_faces + len(nodes))),
    )


@pytest.mark.parametrize("scale", [1, 1e-6, 1e6])
def test_overlapping_solids_count_material_once(cube_mesh, scale):
    pytest.importorskip("manifold3d")
    nodes, faces = two_cubes(cube_mesh, [0.5, 0, 0])
    nodes = nodes * scale
    before_nodes, before_faces = nodes.copy(), faces.copy()
    result = analyze_mesh(nodes, faces)
    assert result.values["volume"] == pytest.approx(1.5 * scale**3, rel=1e-6, abs=0)
    assert result.values["surface_area"] == pytest.approx(8 * scale**2, rel=1e-6, abs=0)
    assert result.values["centroid_x"] == pytest.approx(0.75 * scale, rel=1e-6, abs=0)
    assert result.values["centroid_y"] == pytest.approx(0.5 * scale, rel=1e-6, abs=0)
    assert result.values["centroid_z"] == pytest.approx(0.5 * scale, rel=1e-6, abs=0)
    assert result.values["sphericity_wadell"] == pytest.approx(
        np.pi**(1 / 3) * (6 * 1.5)**(2 / 3) / 8, rel=1e-6
    )
    np.testing.assert_array_equal(nodes, before_nodes)
    np.testing.assert_array_equal(faces, before_faces)
    vertices, triangles = prepare_mesh(nodes, faces)
    assert surface_area(vertices, triangles) == result.values["surface_area"]


@pytest.mark.parametrize("offset,second_scale,volume,area,centroid_x", [
    ([2, 0, 0], 1, 2, 12, 1.5),       # disjoint solids
    ([1, 0, 0], 1, 2, 10, 1),         # face contact
    ([0, 0, 0], 1, 1, 6, 0.5),        # identical solids
    ([0.25, 0.25, 0.25], 0.5, 1, 6, 0.5),  # inner solid is not a cavity
])
def test_union_component_arrangements(cube_mesh, offset, second_scale, volume, area, centroid_x):
    pytest.importorskip("manifold3d")
    result = analyze_mesh(*two_cubes(cube_mesh, offset, second_scale))
    assert result.values["volume"] == pytest.approx(volume, rel=1e-6)
    assert result.values["surface_area"] == pytest.approx(area, rel=1e-6)
    assert result.values["centroid_x"] == pytest.approx(centroid_x, rel=1e-6)


def test_missing_union_backend_is_not_a_sum_fallback(cube_mesh, monkeypatch):
    monkeypatch.setitem(sys.modules, "manifold3d", None)
    with pytest.raises(ImportError, match=r"Multi-component solid union requires pyshape\[mesh\]"):
        analyze_mesh(*two_cubes(cube_mesh, [0.5, 0, 0]))


def test_two_spheres_against_analytic_union():
    pytest.importorskip("manifold3d")
    trimesh = pytest.importorskip("trimesh")
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1)
    nodes = np.vstack((sphere.vertices, sphere.vertices + [1, 0, 0]))
    faces = np.vstack((sphere.faces, sphere.faces[:, ::-1] + len(sphere.vertices)))
    result = analyze_mesh(nodes, faces)
    # Equal unit spheres with center separation 1: overlap is 5*pi/12.
    assert result.values["volume"] == pytest.approx(9 * np.pi / 4, rel=0.005)
    assert result.values["surface_area"] == pytest.approx(6 * np.pi, rel=0.005)
    assert result.values["centroid_x"] == pytest.approx(0.5, abs=1e-7)


@pytest.mark.parametrize("name,volume,area", [
    ("case-000001.stl", 2.1301211948057196e-13, 1.7279899272760144e-08),
    ("case-000005.stl", 2.492697857129396e-14, 5.095525210797647e-09),
])
@pytest.mark.parametrize("builtin_loader", [False, True])
def test_user_mesh_regression(name, volume, area, builtin_loader, monkeypatch):
    pytest.importorskip("manifold3d")
    path = Path(__file__).resolve().parents[1] / "examples" / "assets" / name
    if not path.exists():
        pytest.skip("Local user-provided mesh is not bundled in the distribution")
    if builtin_loader:
        monkeypatch.setattr(io, "HAS_TRIMESH", False)
    geometry = load_geometry(path)
    result = analyze_file(path)
    assert result.values["volume"] == pytest.approx(volume, rel=2e-6, abs=0)
    assert result.values["surface_area"] == pytest.approx(area, rel=2e-6, abs=0)
    assert not any(m.error for m in result.metrics if m.key != "convexity")
    if name == "case-000005.stl":
        assert result.values["volume"] < 2.501883827587843e-14  # sum of all three shells
        assert result.values["surface_area"] < surface_area(geometry.nodes, geometry.faces)
