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
        "orientation_C", "orientation_F", "orientation_E",
        "orientation_eig1", "orientation_eig2", "orientation_eig3",
        "axis_S", "axis_I", "axis_L", "sphericity_krumbein",
        "potticary_flatness", "potticary_elongation",
        "kongfonseca_flatness", "kongfonseca_elongation",
        "angelidakis_flatness", "angelidakis_elongation",
        "zingg_SI", "zingg_IL", "convexity",
    }
    assert not any(metric.error for metric in result.metrics if metric.key != "convexity")
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert set(payload) == {
        "source_name", "input_units", "scale_factor", "axes_method", "metrics",
        "output_units", "output_scale", "reader", "reader_version", "warnings",
    }
    assert payload["axes_method"] == "pca"
    assert set(payload["metrics"][0]) == {
        "key", "label", "value", "group", "default_selected", "error", "unit",
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
    assert rows[0] == ["parameter", "value", "unit"]
    assert rows[1] == ["source_file", result.source_name, ""]
    assert [row[0] for row in rows[2:]] == ["surface_area", "volume"]
    assert "n_vertices" not in result.to_csv()
    path = tmp_path / "results.csv"
    result.write_csv(path, ["volume"])
    assert path.read_text().splitlines()[-1] == "volume,0.16666666666666666,input_units^3"
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


def test_micrometre_units(tetra_mesh):
    result = analyze_mesh(*tetra_mesh, units="um")
    assert result.scale_factor == 1e-6
    assert result.values["volume"] == pytest.approx(1e-18 / 6)


def test_units_of_dimensional_metrics(tetra_mesh):
    si = analyze_mesh(*tetra_mesh, units="mm")
    units = {metric.key: metric.unit for metric in si.metrics}
    assert units["surface_area"] == "m²" and units["volume"] == "m³"
    assert units["centroid_x"] == units["axis_L"] == "m"
    assert units["inertia_I1"] == "m⁵"
    assert units["sphericity_wadell"] == units["n_faces"] == units["zingg_SI"] == ""
    raw = analyze_mesh(*tetra_mesh)
    assert {m.key: m.unit for m in raw.metrics}["volume"] == "input_units³"
    rows = list(csv.reader(io.StringIO(si.to_csv(["surface_area", "inertia_I1", "convexity"]))))
    assert rows[1:] == [
        ["surface_area", rows[1][1], "m^2"], ["inertia_I1", rows[2][1], "m^5"], ["convexity", rows[3][1], ""],
    ]


@pytest.mark.parametrize("units, expected", [
    ("as-is", "input_units^2"), ("um", "m^2"), ("inch", "m^2"),
])
def test_unit_label(units, expected):
    assert analysis.unit_label("surface_area", units, ascii=True) == expected
    assert analysis.unit_label("convexity", units) == ""


def _scaled_cube(mesh, factor):
    nodes, faces = mesh
    return np.asarray(nodes, dtype=float) * factor, faces


def test_output_units_default_is_metres(cube_mesh):
    default = analyze_mesh(*cube_mesh, units="mm")
    explicit = analyze_mesh(*cube_mesh, units="mm", output_units="m")
    assert default.output_units == "m" and default.output_scale == 1.0
    assert default.values == explicit.values
    assert {m.key: m.unit for m in default.metrics}["volume"] == "m³"


def test_mm_output_equals_metres_scaled(cube_mesh):
    in_m = analyze_mesh(*cube_mesh, units="mm", output_units="m")
    in_mm = analyze_mesh(*cube_mesh, units="mm", output_units="mm")
    assert in_mm.output_units == "mm" and in_mm.output_scale == pytest.approx(1000.0)
    for key, value in in_m.values.items():
        power = analysis.LENGTH_POWERS.get(key, 0)
        if value is None:
            assert in_mm.values[key] is None
        else:
            assert in_mm.values[key] == pytest.approx(value * 1000.0 ** power, rel=1e-12)
    units = {m.key: m.unit for m in in_mm.metrics}
    assert units["axis_L"] == "mm" and units["surface_area"] == "mm²"
    assert units["volume"] == "mm³" and units["inertia_I1"] == "mm⁵"
    assert units["convexity"] == ""
    rows = list(csv.reader(io.StringIO(in_mm.to_csv(["volume", "inertia_I1"]))))
    assert rows[0] == ["parameter", "value", "unit"]  # no source_file row for arrays
    assert [row[2] for row in rows[1:]] == ["mm^3", "mm^5"]
    # a 'same' request with mm input gives the input numbers back for lengths
    same = analyze_mesh(*cube_mesh, units="mm", output_units="same")
    assert same.output_units == "mm" and same.values == in_mm.values


def test_micrometre_and_inch_output(cube_mesh):
    in_m = analyze_mesh(*cube_mesh, units="cm")
    in_um = analyze_mesh(*cube_mesh, units="cm", output_units="um")
    in_in = analyze_mesh(*cube_mesh, units="cm", output_units="inch")
    assert in_um.values["volume"] == pytest.approx(in_m.values["volume"] * 1e18, rel=1e-12)
    assert in_in.values["axis_L"] == pytest.approx(in_m.values["axis_L"] / 0.0254, rel=1e-12)
    assert {m.key: m.unit for m in in_um.metrics}["surface_area"] == "µm²"
    assert in_um.to_csv(["surface_area"]).splitlines()[-1].endswith(",um^2")
    assert {m.key: m.unit for m in in_in.metrics}["axis_L"] == "in"


def test_output_units_ignored_for_as_is(cube_mesh):
    plain = analyze_mesh(*cube_mesh)
    for choice in ("m", "mm", "um", "same"):
        result = analyze_mesh(*cube_mesh, output_units=choice)
        assert result.output_units == "as-is" and result.output_scale == 1.0
        assert result.values == plain.values
        assert {m.key: m.unit for m in result.metrics}["volume"] == "input_units³"


def test_invalid_output_units(cube_mesh):
    with pytest.raises(ValueError, match="Unknown output units"):
        analyze_mesh(*cube_mesh, units="mm", output_units="km")
    with pytest.raises(ValueError, match="Unknown output units"):
        analyze_mesh(*cube_mesh, output_units="km")


def test_auto_units_are_rejected(tetra_mesh):
    with pytest.raises(ValueError, match="Unknown units 'auto'") as error:
        analyze_mesh(*tetra_mesh, units="auto")
    for unit in ("as-is", "um", "mm", "cm", "m", "inch"):
        assert unit in str(error.value)


def test_orientation_indices_cuboid_exact():
    # Cuboid 4 x 2 x 1: f = (8, 4, 2) / 14, so C = 1/4, F = 1/2, E = 1/4
    # (Orosz et al., 2021, Eqs. 3-5); the key of elongation is orientation_E.
    unit = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=float)
    faces = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ], dtype=int)
    result = analyze_mesh(unit * [4.0, 2.0, 1.0], faces)
    values = result.values
    assert "orientation_R" not in values
    assert np.isclose(values["orientation_C"], 0.25)
    assert np.isclose(values["orientation_F"], 0.5)
    assert np.isclose(values["orientation_E"], 0.25)
    labels = {m.key: m.label for m in result.metrics}
    assert labels["orientation_E"] == "Elongation E (surface orientation tensor)"
