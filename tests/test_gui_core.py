from pathlib import Path

import numpy as np

from pyshape.gui_core import (
    Metric,
    compute_metrics,
    make_export_path,
    write_csv,
    axis_triad,
    load_geometry,
)


def make_cube(side: float = 1.0):
    s = side
    nodes = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [1, 5, 6], [1, 6, 2],
        [2, 6, 7], [2, 7, 3],
        [3, 7, 4], [3, 4, 0],
    ], dtype=int)
    return nodes, faces


def test_compute_metrics_cube():
    nodes, faces = make_cube(2.0)
    metrics = compute_metrics(nodes, faces)
    by_key = {m.key: m for m in metrics}
    assert np.isclose(by_key["surface_area"].value, 24.0)
    assert np.isclose(by_key["volume"].value, 8.0)
    assert np.isclose(by_key["sphericity_wadell"].value, (np.pi ** (1 / 3) * (6 * 8) ** (2 / 3)) / 24.0)
    assert by_key["axis_S"].value <= by_key["axis_I"].value <= by_key["axis_L"].value
    assert by_key["convexity"].error is None
    assert np.isclose(by_key["convexity"].value, 1.0, atol=1e-6)


def test_metric_text_formatting():
    ok = Metric("k", "Label", 1.23456789)
    assert ok.text == "1.23457"
    na = Metric("k", "Label", None, error="boom")
    assert na.text == "n/a (boom)"
    missing = Metric("k", "Label", None)
    assert missing.text == "n/a"


def test_make_export_path_default_overwrites(tmp_path):
    src = tmp_path / "sample.stl"
    src.write_text("solid x\nendsolid x\n")
    out = make_export_path(src, "_SHAPE", avoid_overwrite=False)
    assert out == tmp_path / "sample_SHAPE.csv"
    out.write_text("existing")
    out2 = make_export_path(src, "_SHAPE", avoid_overwrite=False)
    assert out2 == out  # still overwrites, no numbering


def test_make_export_path_progressive_numbering(tmp_path):
    src = tmp_path / "sample.stl"
    src.write_text("solid x\nendsolid x\n")
    first = make_export_path(src, "_SHAPE", avoid_overwrite=True)
    assert first == tmp_path / "sample_SHAPE.csv"
    first.write_text("existing")
    second = make_export_path(src, "_SHAPE", avoid_overwrite=True)
    assert second == tmp_path / "sample_SHAPE_1.csv"
    second.write_text("existing")
    third = make_export_path(src, "_SHAPE", avoid_overwrite=True)
    assert third == tmp_path / "sample_SHAPE_2.csv"


def test_write_csv(tmp_path):
    out = tmp_path / "out.csv"
    write_csv(out, [("surface_area", 1.5), ("volume", 2.5)], source_name="sample.stl")
    text = out.read_text()
    lines = text.strip().splitlines()
    assert lines[0] == "parameter,value"
    assert lines[1] == "source_file,sample.stl"
    assert lines[2] == "surface_area,1.5"
    assert lines[3] == "volume,2.5"


def test_axis_triad_scales_with_bbox():
    nodes, _ = make_cube(4.0)
    center, length = axis_triad(nodes)
    assert np.allclose(center, [2.0, 2.0, 2.0])
    assert np.isclose(length, 0.6 * 4.0)


def test_load_geometry_stl():
    repo_root = Path(__file__).resolve().parents[1]
    stl_path = repo_root / "examples" / "assets" / "Tetrahedron_ascii.stl"
    geo = load_geometry(stl_path)
    assert geo.source_format == "stl"
    assert geo.converted_from_native is False
    assert geo.nodes.shape[1] == 3
    assert geo.faces.shape[1] == 3


def test_load_geometry_unsupported_extension(tmp_path):
    bogus = tmp_path / "file.xyz"
    bogus.write_text("nope")
    import pytest
    with pytest.raises(ValueError):
        load_geometry(bogus)
