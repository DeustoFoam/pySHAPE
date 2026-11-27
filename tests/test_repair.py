"""Diagnosis of open meshes, repair_mesh and the pyshape-repair command."""
import builtins
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from pyshape import analyze_file, analyze_mesh, diagnose_mesh, load_stl, repair_mesh, write_stl
from pyshape.repair import (
    DEFAULT_REL_TOL, MeshDiagnosis, RepairReport, _close_pairs, main, open_mesh_warning,
)

SYNTHETIC = Path(__file__).resolve().parents[1] / "examples" / "assets" / "synthetic"

CUBE_NODES = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                       [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
CUBE_FACES = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                       [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def split_cube(gap):
    """Cube whose top face (two triangles) has its own vertices, raised by ``gap``."""
    top = [4, 5, 6, 7]
    nodes = np.vstack([CUBE_NODES, CUBE_NODES[top] + [0, 0, gap]])
    faces = CUBE_FACES.copy()
    remap = dict(zip(top, range(8, 12)))
    faces[2:4] = np.vectorize(remap.get)(faces[2:4])
    return nodes, faces


def test_closed_mesh_is_unchanged():
    nodes, faces, report = repair_mesh(CUBE_NODES, CUBE_FACES)
    assert not report.changed and report.closed
    np.testing.assert_array_equal(nodes, CUBE_NODES)
    np.testing.assert_array_equal(faces, CUBE_FACES)
    assert diagnose_mesh(CUBE_NODES, CUBE_FACES).message() is None


def test_inputs_are_not_modified():
    nodes, faces = split_cube(1e-6)
    before = nodes.copy(), faces.copy()
    repair_mesh(nodes, faces)
    np.testing.assert_array_equal(nodes, before[0])
    np.testing.assert_array_equal(faces, before[1])


def test_stitching_closes_small_gaps():
    nodes, faces = split_cube(1e-6)
    diagnosis = diagnose_mesh(nodes, faces)
    assert diagnosis.n_boundary_edges == 8 and diagnosis.n_near_boundary_vertices == 8
    assert "pyshape-repair can close" in diagnosis.message()
    fixed_nodes, fixed_faces, report = repair_mesh(nodes, faces)
    assert report.closed and report.changed and report.refused is None
    assert report.stitched_vertices == 4 and report.removed_unreferenced_vertices == 4
    assert report.max_displacement == pytest.approx(1e-6) and report.max_displacement <= report.tolerance
    assert len(fixed_nodes) == 8 and len(fixed_faces) == 12
    volume = analyze_mesh(fixed_nodes, fixed_faces).values["volume"]
    assert volume == pytest.approx(1.0, abs=1e-12)


def test_gaps_wider_than_tolerance_stay_open():
    nodes, faces = split_cube(0.1)
    diagnosis = diagnose_mesh(nodes, faces)
    assert diagnosis.n_near_boundary_vertices == 0 and "holes" in diagnosis.message()
    _, _, report = repair_mesh(nodes, faces)
    assert not report.changed and not report.closed
    _, _, report = repair_mesh(nodes, faces, tol=0.2)  # an explicit tolerance closes it
    assert report.closed and report.rel_tol is None


def test_zero_tolerance_disables_stitching():
    _, _, report = repair_mesh(*split_cube(1e-6), tol=0)
    assert report.stitched_vertices == 0 and not report.closed


def test_degenerate_and_duplicate_faces_are_removed():
    faces = np.vstack([CUBE_FACES, CUBE_FACES[3][::-1], [[0, 0, 1]]])
    diagnosis = diagnose_mesh(CUBE_NODES, faces)
    assert diagnosis.n_duplicate_faces == 1 and diagnosis.n_degenerate_faces == 1
    assert "1 duplicate and 1 degenerate faces, which pyshape-repair removes" \
        in diagnosis.message()
    nodes, fixed, report = repair_mesh(CUBE_NODES, faces)
    assert report.removed_duplicate_faces == 1 and report.removed_degenerate_faces == 1
    assert report.closed and len(fixed) == 12
    assert report.surface_area_after == pytest.approx(6.0)


def test_coincident_vertices_are_merged():
    nodes = np.vstack([CUBE_NODES, CUBE_NODES[0]])
    faces = CUBE_FACES.copy()
    faces[0, 0] = 8  # same position as vertex 0, another index
    _, fixed, report = repair_mesh(nodes, faces)
    assert report.merged_coincident_vertices == 1 and report.closed
    assert fixed.max() == 7


def test_vertices_sharing_an_edge_are_never_joined():
    # An open strip whose boundary edges are shorter than the tolerance.
    nodes = np.array([[0, 0, 0], [1e-6, 0, 0], [0, 1, 0], [1e-6, 1, 0]], float)
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    fixed_nodes, fixed_faces, report = repair_mesh(nodes, faces, tol=1e-3)
    assert report.stitched_vertices == 0
    np.testing.assert_array_equal(fixed_faces, faces)


def test_groups_do_not_chain():
    # Three separate triangles whose corners at x ~ 0 are 0.6 tol apart in a row.
    tol = 1e-3
    xs = [0.0, 0.6 * tol, 1.2 * tol]
    nodes, faces = [], []
    for i, x in enumerate(xs):
        nodes += [[x, 0, 0], [x + 1, 5 * i, 0], [x + 1, 5 * i + 1, 0]]
        faces.append([3 * i, 3 * i + 1, 3 * i + 2])
    _, _, report = repair_mesh(np.array(nodes), np.array(faces), tol=tol)
    assert report.stitched_vertices == 1 and report.max_displacement <= tol


def test_stitching_that_creates_nonmanifold_edges_is_refused():
    # Three fins whose free edges almost coincide: joining them would make one
    # edge shared by three triangles.
    nodes, faces = [], []
    for i, angle in enumerate((0.0, 2.0, 4.0)):
        offset = i * 1e-7
        nodes += [[offset, 0, 0], [offset, 1, 0], [np.cos(angle), 0.5, np.sin(angle)]]
        faces.append([3 * i, 3 * i + 1, 3 * i + 2])
    nodes, faces = np.array(nodes), np.array(faces)
    fixed_nodes, fixed_faces, report = repair_mesh(nodes, faces, tol=1e-3)
    assert report.refused and "non-manifold" in report.refused
    assert report.stitched_vertices == 0
    np.testing.assert_array_equal(fixed_nodes, nodes)
    np.testing.assert_array_equal(fixed_faces, faces)


def test_pair_search_without_scipy_matches(monkeypatch):
    pytest.importorskip("scipy")
    points = np.random.default_rng(3).random((400, 3))
    with_scipy = _close_pairs(points, 0.05)
    real_import = builtins.__import__

    def no_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_scipy)
    np.testing.assert_array_equal(_close_pairs(points, 0.05), with_scipy)
    assert len(with_scipy) > 0


@pytest.mark.parametrize("binary", [None, True, False])
def test_write_stl_round_trip(tmp_path, binary):
    nodes = CUBE_NODES * 0.1 if binary is False else CUBE_NODES  # 0.1 is not a float32
    path = tmp_path / "cube.stl"
    write_stl(path, nodes, CUBE_FACES, binary=binary)
    read_nodes, read_faces = load_stl(path)
    np.testing.assert_array_equal(read_nodes[read_faces], nodes[CUBE_FACES])
    is_binary = not path.read_bytes().startswith(b"solid")
    assert is_binary == (binary is not False)


def test_write_stl_chooses_ascii_for_double_precision(tmp_path):
    path = tmp_path / "cube.stl"
    write_stl(path, CUBE_NODES * 0.1, CUBE_FACES)
    assert path.read_bytes().startswith(b"solid")
    read_nodes, read_faces = load_stl(path)
    np.testing.assert_array_equal(read_nodes[read_faces], (CUBE_NODES * 0.1)[CUBE_FACES])


def test_analysis_warns_only_for_open_meshes():
    assert analyze_file(SYNTHETIC / "sphere.stl").warnings == []
    gaps = analyze_file(SYNTHETIC / "sphere_with_gaps.stl")
    assert gaps.values["volume"] is None
    assert len(gaps.warnings) == 1 and "pyshape-repair can close" in gaps.warnings[0]
    hole = analyze_file(SYNTHETIC / "open_ellipsoid.stl")
    assert "holes" in hole.warnings[0] and "pyshape-repair can close" not in hole.warnings[0]
    assert any(row[0] == "warning" for row in hole.metadata_rows())


def test_repaired_sphere_matches_the_closed_sphere():
    from pyshape import load_geometry

    geo = load_geometry(SYNTHETIC / "sphere_with_gaps.stl")
    nodes, faces, report = repair_mesh(geo.nodes, geo.faces)
    assert report.closed and report.max_displacement < 2e-6
    assert report.tolerance == pytest.approx(DEFAULT_REL_TOL * report.median_edge_length)
    result = analyze_mesh(nodes, faces)
    assert result.warnings == []
    # The same icosphere without the gaps, from the generator of the file.
    spec = importlib.util.spec_from_file_location(
        "make_synthetic_assets", SYNTHETIC.parents[1] / "make_synthetic_assets.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    closed = analyze_mesh(*generator.icosphere(3))
    for key in ("volume", "surface_area", "sphericity_wadell"):
        assert result.values[key] == pytest.approx(closed.values[key], rel=1e-5)


def test_command_writes_repaired_file_and_report(tmp_path, capsys):
    source = tmp_path / "gaps.stl"
    source.write_bytes((SYNTHETIC / "sphere_with_gaps.stl").read_bytes())
    assert main([str(source)]) == 0
    out = tmp_path / "gaps_repaired.stl"
    record = json.loads((tmp_path / "gaps_repaired.json").read_text(encoding="utf-8"))
    assert record["input"] == "gaps.stl" and record["report"]["closed"]
    assert len(record["input_sha256"]) == 64 and record["settings"]["rel_tol"] == DEFAULT_REL_TOL
    assert analyze_file(out).warnings == []
    assert source.read_bytes() == (SYNTHETIC / "sphere_with_gaps.stl").read_bytes()
    assert "closed" in capsys.readouterr().out
    # --no-overwrite numbers the next output instead of replacing it.
    assert main([str(source), "--no-overwrite"]) == 0
    assert (tmp_path / "gaps_repaired_1.stl").exists()


def test_command_check_and_unchanged_files_write_nothing(tmp_path, capsys):
    for name in ("sphere_with_gaps.stl", "open_ellipsoid.stl", "sphere.stl"):
        (tmp_path / name).write_bytes((SYNTHETIC / name).read_bytes())
    assert main([str(tmp_path / "sphere_with_gaps.stl"), "--check"]) == 0
    assert main([str(tmp_path / "open_ellipsoid.stl"), str(tmp_path / "sphere.stl")]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "open_ellipsoid.stl", "sphere.stl", "sphere_with_gaps.stl"]
    text = capsys.readouterr().out
    assert "nothing changed" in text and "holes" in text


def test_command_errors(tmp_path, capsys):
    source = tmp_path / "gaps.stl"
    source.write_bytes((SYNTHETIC / "sphere_with_gaps.stl").read_bytes())
    with pytest.raises(SystemExit):
        main([str(source), str(source), "-o", str(tmp_path / "x.stl")])
    assert main([str(source), "-o", str(source)]) == 1  # never replaces the input
    assert main([str(tmp_path / "missing.stl")]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_report_types_and_dicts():
    nodes, faces = split_cube(1e-6)
    diagnosis = diagnose_mesh(nodes, faces)
    *_, report = repair_mesh(nodes, faces)
    assert isinstance(diagnosis, MeshDiagnosis) and isinstance(report, RepairReport)
    assert open_mesh_warning(nodes, faces) == diagnosis.message()
    assert open_mesh_warning(CUBE_NODES, CUBE_FACES) is None
    json.dumps(diagnosis.to_dict()), json.dumps(report.to_dict())
    assert diagnosis.to_dict()["closed"] is False and report.to_dict()["changed"] is True
