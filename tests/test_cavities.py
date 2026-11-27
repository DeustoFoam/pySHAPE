"""Nested shells as cavities (issue #2), on the static synthetic particles."""
from pathlib import Path

import numpy as np
import pytest

from pyshape import analyze_file, analyze_mesh, load_geometry, prepare_mesh, prepare_solid
from pyshape.geometry import closed_surface_inertia_tensor, surface_area
from pyshape.preprocess import _oriented_shells

SYNTHETIC = Path(__file__).resolve().parents[1] / "examples" / "assets" / "synthetic"
CAVITY_KEYS = ("n_cavities", "cavity_volume", "porosity")


def shells(name):
    """Shells of a synthetic file, each oriented outwards, largest first."""
    geometry = load_geometry(SYNTHETIC / f"{name}.stl")
    nodes, oriented, components = _oriented_shells(geometry.nodes, geometry.faces)
    parts = [(nodes, oriented[np.asarray(c)]) for c in components]
    return sorted(parts, key=lambda part: -abs(_volume(*part)))


def _volume(nodes, faces):
    t = nodes[faces]
    return np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6.0


def _centroid(nodes, faces):
    t = nodes[faces]
    signed = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0
    return (signed[:, None] * t.sum(axis=1) / 4.0).sum(axis=0) / signed.sum()


def test_assets_exist():
    names = {p.stem for p in SYNTHETIC.glob("*.stl")}
    assert {"hollow_sphere", "cube_with_spherical_void", "sphere_with_two_cavities",
            "nested_island", "two_spheres_overlapping", "open_ellipsoid", "sphere"} <= names
    assert (SYNTHETIC / "README.md").is_file()


def test_default_fill_is_unchanged_and_reports_cavities_as_unavailable():
    # A sphere inside a sphere (e.g. a DEM clump) is solid by default: the outer sphere.
    pytest.importorskip("manifold3d")  # "fill" merges the shells with Manifold
    result = analyze_file(SYNTHETIC / "hollow_sphere.stl")
    assert result.cavities == "fill"
    assert result.values["volume"] == pytest.approx(_volume(*shells("hollow_sphere")[0]), rel=1e-6)
    errors = {m.key: m.error for m in result.metrics}
    for key in CAVITY_KEYS:
        assert result.values[key] is None and errors[key] == "cavities are filled (cavities='fill')"
    assert result.values["surface_area_outer"] == result.values["surface_area"]


def test_hollow_sphere_exact_mass_properties():
    (n_out, f_out), (n_in, f_in) = shells("hollow_sphere")
    v_out, v_in = _volume(n_out, f_out), _volume(n_in, f_in)
    a_out, a_in = surface_area(n_out, f_out), surface_area(n_in, f_in)
    result = analyze_file(SYNTHETIC / "hollow_sphere.stl", cavities="keep")
    values = result.values
    assert result.cavities == "keep"
    assert values["volume"] == pytest.approx(v_out - v_in, rel=1e-12)
    assert values["surface_area"] == pytest.approx(a_out + a_in, rel=1e-12)
    assert values["surface_area_outer"] == pytest.approx(a_out, rel=1e-12)
    assert values["n_cavities"] == 1
    assert values["cavity_volume"] == pytest.approx(v_in, rel=1e-9)
    assert values["porosity"] == pytest.approx(v_in / v_out, rel=1e-9)
    # Inertia of the material = inertia of the outer solid minus that of the void.
    i_out = closed_surface_inertia_tensor(n_out, f_out, np.zeros(3))[0]
    i_in = closed_surface_inertia_tensor(n_in, f_in, np.zeros(3))[0]
    expected = np.linalg.eigvalsh(i_out - i_in)
    got = [values[f"inertia_I{k}"] for k in (1, 2, 3)]
    assert got == pytest.approx(expected, rel=1e-9)


def test_form_descriptors_use_the_envelope():
    # The cube's PCA directions are not unique, so compare the minimum-volume box and
    # the inertia ellipsoid of the envelope. "fill" merges the shells with Manifold
    # (single precision), hence the tolerance.
    pytest.importorskip("scipy")
    pytest.importorskip("manifold3d")
    path = SYNTHETIC / "cube_with_spherical_void.stl"
    for axes in ("min_volume", "inertia_ellipsoid"):
        fill = analyze_file(path, axes=axes).values
        keep = analyze_file(path, axes=axes, cavities="keep").values
        for key in ("sphericity_wadell", "convexity", "orientation_C", "orientation_F",
                    "orientation_E", "axis_S", "axis_I", "axis_L", "sphericity_krumbein",
                    "zingg_SI", "zingg_IL"):
            assert keep[key] == pytest.approx(fill[key], rel=1e-6, abs=1e-9), (axes, key)
        assert keep["volume"] < fill["volume"]


def test_off_centre_void_moves_the_centroid():
    (n_out, f_out), (n_in, f_in) = shells("cube_with_spherical_void")
    v_out, v_in = _volume(n_out, f_out), _volume(n_in, f_in)
    expected = (v_out * _centroid(n_out, f_out) - v_in * _centroid(n_in, f_in)) / (v_out - v_in)
    values = analyze_file(SYNTHETIC / "cube_with_spherical_void.stl", cavities="keep").values
    got = [values[f"centroid_{axis}"] for axis in "xyz"]
    assert got == pytest.approx(expected, abs=1e-9)
    assert np.sign(got[0]) == -np.sign(_centroid(n_in, f_in)[0])  # away from the void


def test_two_cavities_and_nested_island_follow_the_even_odd_rule():
    two = analyze_file(SYNTHETIC / "sphere_with_two_cavities.stl", cavities="keep").values
    parts = shells("sphere_with_two_cavities")
    assert two["n_cavities"] == 2
    assert two["volume"] == pytest.approx(
        _volume(*parts[0]) - _volume(*parts[1]) - _volume(*parts[2]), rel=1e-12)
    island = analyze_file(SYNTHETIC / "nested_island.stl", cavities="keep").values
    big, middle, small = (_volume(*part) for part in shells("nested_island"))
    assert island["n_cavities"] == 1
    assert island["volume"] == pytest.approx(big - middle + small, rel=1e-12)
    assert island["cavity_volume"] == pytest.approx(middle - small, rel=1e-9)


def test_result_does_not_depend_on_the_winding_in_the_file():
    geometry = load_geometry(SYNTHETIC / "hollow_sphere.stl")
    nodes, faces = geometry.nodes, geometry.faces
    flipped = faces.copy()
    half = len(faces) // 2  # the second shell (the cavity) is the second half of the file
    flipped[half:] = flipped[half:, ::-1]
    a = analyze_mesh(nodes, faces, cavities="keep").values
    b = analyze_mesh(nodes, flipped, cavities="keep").values
    assert a == pytest.approx(b, rel=1e-12, abs=1e-15)


def test_overlapping_solids_are_still_merged_with_keep():
    pytest.importorskip("manifold3d")
    keep = analyze_file(SYNTHETIC / "two_spheres_overlapping.stl", cavities="keep").values
    fill = analyze_file(SYNTHETIC / "two_spheres_overlapping.stl").values
    assert keep["n_cavities"] == 0 and keep["porosity"] == 0
    assert keep["volume"] == pytest.approx(fill["volume"], rel=1e-12)


def test_prepared_solid_flags_cavity_walls_inwards():
    geometry = load_geometry(SYNTHETIC / "hollow_sphere.stl")
    solid = prepare_solid(geometry.nodes, geometry.faces, cavities="keep")
    walls = solid.faces[solid.cavity_faces]
    assert 0 < len(walls) < len(solid.faces)
    assert _volume(solid.nodes, walls) < 0          # oriented towards the cavity
    assert _volume(solid.envelope_nodes, solid.envelope_faces) > 0
    nodes, faces = prepare_mesh(geometry.nodes, geometry.faces, cavities="keep")
    assert np.array_equal(faces, solid.faces)


def test_overlapping_cavities_merge_into_one():
    pytest.importorskip("manifold3d")
    (n_out, f_out), (n_in, f_in) = shells("hollow_sphere")
    # A second void inside the material that crosses the wall of the first one.
    second = (n_in - n_in.mean(axis=0)) * 0.6 + [0.5, 0, 0]
    nodes = np.vstack([n_out, n_in, second])
    faces = np.vstack([f_out, f_in + len(n_out), f_in + len(n_out) + len(n_in)])
    values = analyze_mesh(nodes, faces, cavities="keep").values
    v_first, v_second = _volume(n_in, f_in), _volume(second, f_in)
    assert values["n_cavities"] == 1
    assert max(v_first, v_second) < values["cavity_volume"] < v_first + v_second
    assert values["volume"] == pytest.approx(_volume(n_out, f_out) - values["cavity_volume"],
                                             rel=1e-9)


def test_unknown_mode_and_open_mesh():
    geometry = load_geometry(SYNTHETIC / "hollow_sphere.stl")
    with pytest.raises(ValueError, match="cavities"):
        analyze_mesh(geometry.nodes, geometry.faces, cavities="hollow")
    open_result = analyze_file(SYNTHETIC / "open_ellipsoid.stl", cavities="keep").values
    assert open_result["volume"] is None and open_result["porosity"] is None
    assert open_result["n_cavities"] == 0


def test_batch_and_web_accept_cavities(tmp_path):
    from pyshape import analyze_folder

    target = tmp_path / "hollow_sphere.stl"
    target.write_bytes((SYNTHETIC / "hollow_sphere.stl").read_bytes())
    batch = analyze_folder(tmp_path, cavities="keep", write_csv=False)
    assert batch.items[0].result.values["n_cavities"] == 1
    with pytest.raises(ValueError, match="cavities"):
        analyze_folder(tmp_path, cavities="x")
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from pyshape import web

    with TestClient(web.create_app()) as client:
        response = client.post(
            "/api/analyze", files={"file": ("p.stl", target.read_bytes())},
            data={"cavities": "keep", "include_preview": "true"})
        payload = response.json()
        assert response.status_code == 200 and payload["cavities"] == "keep"
        assert payload["preview"]["cavity_triangles"]
        assert len(payload["preview"]["triangles"]) + len(payload["preview"]["cavity_triangles"]) \
            == payload["preview"]["shown_faces"]
        assert client.post("/api/analyze", files={"file": ("p.stl", target.read_bytes())},
                           data={"cavities": "x"}).status_code == 422
        pytest.importorskip("manifold3d")  # "fill" merges the shells with Manifold
        fill = client.post("/api/analyze", files={"file": ("p.stl", target.read_bytes())},
                           data={"include_preview": "true"}).json()
        assert fill["preview"]["cavity_triangles"] == []
