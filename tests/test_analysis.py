import csv
import io
import json
import subprocess
import sys

import numpy as np
import pytest

from pyshape import AnalysisResult, analyze_file, analyze_mesh, compute_metrics, load_geometry
from pyshape import analysis


def test_import_does_not_load_optional_dependencies():
    subprocess.run(
        [sys.executable, "-c",
         "import sys; import pyshape; "
         "assert not any(name in sys.modules for name in "
         "('tkinter', 'matplotlib', 'fastapi', 'scipy', 'trimesh'))"],
        check=True,
    )


def test_public_api_and_result_shape(tetra_mesh):
    nodes, faces = tetra_mesh
    result = analyze_mesh(nodes, faces)
    assert isinstance(result, AnalysisResult)
    assert result.values["surface_area"] == pytest.approx((3 + np.sqrt(3)) / 2)
    assert result.values["volume"] == pytest.approx(1 / 6)
    assert result.values["centroid_x"] == pytest.approx(0.25)
    assert len(result.metrics) == len(result.values) == 30
    assert set(result.values) == {
        "n_vertices", "n_faces", "surface_area", "volume",
        "centroid_x", "centroid_y", "centroid_z",
        "inertia_I1", "inertia_I2", "inertia_I3", "sphericity_wadell",
        "orientation_C", "orientation_F", "orientation_R",
        "orientation_eig1", "orientation_eig2", "orientation_eig3",
        "axis_S", "axis_I", "axis_L", "sphericity_krumbein",
        "potticary_flatness", "potticary_elongation",
        "kongfonseca_flatness", "kongfonseca_elongation",
        "angelidakis_flatness", "angelidakis_elongation",
        "zingg_SI", "zingg_IL", "convexity",
    }
    assert not any(metric.error for metric in result.metrics if metric.key != "convexity")
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert set(payload) == {"source_name", "input_units", "scale_factor", "axes_method", "metrics"}
    assert payload["axes_method"] == "pca"
    assert set(payload["metrics"][0]) == {
        "key", "label", "value", "group", "default_selected", "error",
    }


def test_file_workflow(stl_path):
    result = analyze_file(stl_path)
    geometry = load_geometry(stl_path)
    expected = analyze_mesh(geometry.nodes, geometry.faces)
    assert result.values == expected.values
    assert result.source_name == stl_path.name


def test_unit_conversion_and_input_not_mutated(tetra_mesh):
    nodes, faces = tetra_mesh
    original_nodes, original_faces = nodes.copy(), faces.copy()
    raw = analyze_mesh(nodes, faces)
    converted = analyze_mesh(nodes, faces, units="mm")
    assert converted.input_units == "mm"
    assert converted.scale_factor == 0.001
    assert converted.values["surface_area"] == pytest.approx(raw.values["surface_area"] * 1e-6)
    assert converted.values["volume"] == pytest.approx(raw.values["volume"] * 1e-9)
    assert converted.values["axis_L"] == pytest.approx(raw.values["axis_L"] * 1e-3)
    assert converted.values["sphericity_wadell"] == pytest.approx(raw.values["sphericity_wadell"])
    np.testing.assert_array_equal(nodes, original_nodes)
    np.testing.assert_array_equal(faces, original_faces)


@pytest.mark.parametrize("span,scale", [(1, 1), (10, 1), (11, 1e-3), (10000, 1e-3), (10001, 1e-6)])
def test_auto_units_are_opt_in(tetra_mesh, span, scale):
    nodes, faces = tetra_mesh
    assert analyze_mesh(nodes * span, faces).scale_factor == 1
    assert analyze_mesh(nodes * span, faces, units="auto").scale_factor == scale


def test_one_based_faces(tetra_mesh):
    nodes, faces = tetra_mesh
    assert analyze_mesh(nodes, faces + 1).values == analyze_mesh(nodes, faces).values


def test_reversed_winding(tetra_mesh):
    nodes, faces = tetra_mesh
    original = analyze_mesh(nodes, faces)
    reversed_result = analyze_mesh(nodes, faces[:, ::-1])
    assert reversed_result.values["volume"] == pytest.approx(original.values["volume"])
    for axis in "xyz":
        assert reversed_result.values[f"centroid_{axis}"] == pytest.approx(0.25)


def test_translation(tetra_mesh):
    nodes, faces = tetra_mesh
    translated = analyze_mesh(nodes + [3, 4, 5], faces)
    assert translated.values["volume"] == pytest.approx(1 / 6)
    assert translated.values["centroid_x"] == pytest.approx(3.25)
    assert translated.values["centroid_y"] == pytest.approx(4.25)
    assert translated.values["centroid_z"] == pytest.approx(5.25)


def test_open_mesh_volume_is_explicit(tetra_mesh):
    nodes, faces = tetra_mesh
    faces = faces[:-1]
    result = analyze_mesh(nodes, faces)
    assert result.values["surface_area"] > 0
    assert len(result.metrics) == 30
    for key in ["volume", "centroid_x", "centroid_y", "centroid_z", "inertia_I1",
                "sphericity_wadell", "convexity"]:
        metric = next(m for m in result.metrics if m.key == key)
        assert metric.value is None
        assert "requires" in metric.error
    json.dumps(result.to_dict(), allow_nan=False)


def test_inconsistent_winding_is_repaired_without_mutation(tetra_mesh):
    nodes, faces = tetra_mesh
    scrambled = faces.copy()
    scrambled[0] = scrambled[0, ::-1]
    before = scrambled.copy()
    expected = analyze_mesh(nodes, faces)
    repaired = analyze_mesh(nodes, scrambled)
    assert repaired.values == pytest.approx(expected.values)
    np.testing.assert_array_equal(scrambled, before)
    assert analysis._volume_error(analysis.prepare_mesh(nodes, scrambled)[1]) is None


def test_zero_enclosed_volume(tetra_mesh):
    nodes, faces = tetra_mesh
    nodes[:, 2] = 0
    result = analyze_mesh(nodes, faces)
    assert result.values["volume"] is None
    assert next(m.error for m in result.metrics if m.key == "volume")


@pytest.mark.parametrize(
    "nodes,faces,message",
    [
        ([], [], "nodes must have shape"),
        ([[0, 0, 0]] * 3, [], "faces must have nonempty shape"),
        ([[0, 0, 0]] * 3, [[0, 1, 2]], "positive surface area"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, float("nan")]], [[0, 1, 2]], "finite"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1.5, 2]], "integer"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[-1, 1, 2]], "out of range"),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]], "out of range"),
    ],
)
def test_invalid_arrays(nodes, faces, message):
    with pytest.raises(ValueError, match=message):
        compute_metrics(nodes, faces)


def test_invalid_units(tetra_mesh):
    with pytest.raises(ValueError, match="Unknown units"):
        analyze_mesh(*tetra_mesh, units="feet")


def test_missing_scipy_only_affects_convexity(tetra_mesh, monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    result = analyze_mesh(*tetra_mesh)
    assert result.values["volume"] == pytest.approx(1 / 6)
    hull = next(m for m in result.metrics if m.key == "convexity")
    assert hull.value is None
    assert "pyshape[analysis]" in hull.error


@pytest.mark.parametrize("error", [RuntimeError, ImportError])
def test_unexpected_computation_errors_propagate(tetra_mesh, monkeypatch, error):
    def broken(_nodes):
        raise error("unexpected implementation error")

    monkeypatch.setattr(analysis, "convex_hull_volume", broken)
    with pytest.raises(error, match="unexpected implementation error"):
        analyze_mesh(*tetra_mesh)


def test_csv_selection_and_default_export(tetra_mesh, tmp_path):
    result = analyze_mesh(*tetra_mesh)
    result.source_name = 'example,"particle".stl'
    rows = list(csv.reader(io.StringIO(result.to_csv(["surface_area", "volume"]))))
    assert rows[0] == ["parameter", "value"]
    assert rows[1] == ["source_file", result.source_name]
    assert [row[0] for row in rows[2:]] == ["surface_area", "volume"]
    assert "n_vertices" not in result.to_csv()
    path = tmp_path / "results.csv"
    result.write_csv(path, ["volume"])
    assert path.read_text().splitlines()[-1] == "volume,0.16666666666666666"
    with pytest.raises(ValueError, match="Unknown metric"):
        result.to_csv(["typo"])
    with pytest.raises(ValueError, match="at least one"):
        result.to_csv([])
    with pytest.raises(ValueError, match="unavailable"):
        analyze_mesh(tetra_mesh[0], tetra_mesh[1][:-1]).to_csv(["volume"])


def test_inertia_matches_tetrahedral_reference(tetra_mesh):
    from pyshape import volume_centroid_inertia_tensor

    nodes, faces = tetra_mesh
    expected = volume_centroid_inertia_tensor(nodes, [[0, 1, 2, 3]])[3].diagonal()
    values = analyze_mesh(nodes + [2, -1, 5], faces).values
    got = [values[f"inertia_I{i}"] for i in (1, 2, 3)]
    assert got == pytest.approx(expected, rel=1e-12)


def test_inertia_of_cuboid_and_units(cube_mesh):
    nodes, faces = cube_mesh
    a, b, c = 1.0, 2.0, 3.0
    values = analyze_mesh(nodes * [a, b, c], faces, units="mm").values
    mass = a * b * c
    expected = sorted([mass * (b * b + c * c) / 12, mass * (a * a + c * c) / 12,
                       mass * (a * a + b * b) / 12])
    got = [values[f"inertia_I{i}"] for i in (1, 2, 3)]
    assert got == pytest.approx([e * 1e-15 for e in expected], rel=1e-12)


def test_fast_and_general_orientation_paths_agree(cube_mesh, monkeypatch):
    nodes, faces = cube_mesh
    two_shells = (np.vstack((nodes, nodes + 5)), np.vstack((faces, faces + len(nodes))))
    for mesh in ((nodes, faces), (nodes, faces[:, ::-1]), two_shells):
        expected = analysis.prepare_mesh(*mesh) if mesh is not two_shells else None
        fast = analysis._consistent_components(
            len(mesh[1]), *_edge_pairs(mesh[1]))
        assert fast is not None
        if expected is None:
            assert len(fast) == 2
            continue
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "scipy.sparse", None)
            general = analysis.prepare_mesh(*mesh)
        np.testing.assert_array_equal(expected[1], general[1])


def _edge_pairs(faces):
    faces = np.asarray(faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    return edges, np.argsort(inverse.ravel(), kind="stable").reshape(-1, 2)


def test_micrometre_units(tetra_mesh):
    result = analyze_mesh(*tetra_mesh, units="um")
    assert result.scale_factor == 1e-6
    assert result.values["volume"] == pytest.approx(1e-18 / 6)
