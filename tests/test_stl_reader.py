"""STL files: built-in parser first, trimesh only as a recorded fallback (issue #20)."""
import csv
import importlib.util
import re
import io as std_io
import struct
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

from pyshape import analyze_file, analyze_folder, analyze_mesh, io, load_geometry, load_stl
from pyshape.batch import main as batch_main
from pyshape.gui_core import geo_metadata_rows, reader_text

REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "examples" / "assets"
REASON = "binary STL: the header gives 99 triangles (5034 bytes) but the file has 684 bytes"


def binary_stl(triangles, header=b"pySHAPE test"):
    triangles = np.asarray(triangles, dtype="<f4").reshape(-1, 3, 3)
    data = header.ljust(80, b" ")[:80] + struct.pack("<I", len(triangles))
    for tri in triangles:
        data += struct.pack("<12fH", 0, 0, 0, *tri.ravel(), 0)
    return data


def ascii_stl(triangles, name="test"):
    lines = [f"solid {name}"]
    for tri in np.asarray(triangles, dtype=float).reshape(-1, 3, 3):
        lines += [" facet normal 0 0 0", "  outer loop"]
        lines += [f"   vertex {x!r} {y!r} {z!r}" for x, y, z in tri.tolist()]
        lines += ["  endloop", " endfacet"]
    return ("\n".join(lines + [f"endsolid {name}"]) + "\n").encode("latin-1")


@pytest.fixture
def cube_triangles(cube_mesh):
    nodes, faces = cube_mesh
    return nodes[faces]


@pytest.fixture
def float32_triangles():
    """A closed icosahedron with coordinates that are not round numbers."""
    phi = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
                  [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
                  [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]]) * 0.37 + 0.11
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
                  [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
                  [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
                  [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    # float32 values, so that the binary and the ASCII files hold the same numbers
    return v.astype(np.float32).astype(float)[f]


@pytest.fixture
def broken_parser(monkeypatch):
    """Simulate an STL file that the built-in parser rejects.

    No real file is known that the parser rejects and trimesh reads: trimesh
    returns an empty mesh for every malformed binary file tried, and the parser
    now reads 8-bit ASCII names itself. The fallback path is therefore tested
    with a parser that fails on purpose.
    """
    def reject(data):
        raise io.STLParseError(REASON)

    monkeypatch.setattr(io, "_parse_stl", reject)


@pytest.fixture
def no_trimesh_calls(monkeypatch):
    trimesh = pytest.importorskip("trimesh")

    def forbidden(*args, **kwargs):
        raise AssertionError("trimesh must not read STL files the parser accepts")

    monkeypatch.setattr(trimesh, "load", forbidden)


# --- built-in parser -------------------------------------------------------

@pytest.mark.parametrize("name", ["cube_triangles", "float32_triangles"])
def test_ascii_and_binary_versions_give_identical_arrays(tmp_path, request, name):
    triangles = request.getfixturevalue(name)
    (tmp_path / "a.stl").write_bytes(ascii_stl(triangles))
    (tmp_path / "b.stl").write_bytes(binary_stl(triangles))
    nodes_a, faces_a = load_stl(tmp_path / "a.stl", strict=True)
    nodes_b, faces_b = load_stl(tmp_path / "b.stl", strict=True)
    assert np.array_equal(nodes_a, nodes_b) and np.array_equal(faces_a, faces_b)
    assert np.array_equal(nodes_a[faces_a], triangles)  # same triangles, same winding


def test_vertices_are_merged_in_order_of_first_appearance(tmp_path, cube_triangles):
    path = tmp_path / "cube.stl"
    path.write_bytes(binary_stl(cube_triangles))
    nodes, faces = load_stl(path)
    assert len(nodes) == 8 and faces.dtype == np.int32
    first = [tuple(v) for v in cube_triangles.reshape(-1, 3)]
    order = sorted(set(first), key=first.index)
    assert [tuple(v) for v in nodes] == order
    assert faces.max() == 7 and faces[0].tolist() == [0, 1, 2]


def test_binary_header_starting_with_solid_is_read_as_binary(tmp_path, cube_triangles):
    path = tmp_path / "cube.stl"
    path.write_bytes(binary_stl(cube_triangles, header=b"solid exported by a CAD program"))
    geometry = load_geometry(path)
    assert geometry.reader == io.READER_BUILTIN_STL
    assert len(geometry.faces) == 12 and geometry.warnings == []


def test_stl_uses_builtin_parser_even_with_trimesh(stl_path, no_trimesh_calls, tmp_path,
                                                   cube_triangles):
    assert io.HAS_TRIMESH
    binary = _write(tmp_path / "cube.stl", binary_stl(cube_triangles))
    for path in (stl_path, binary):  # ASCII and binary
        geometry = load_geometry(path)
        assert geometry.reader == "builtin-stl"
        assert geometry.reader_version is None and geometry.warnings == []
        result = analyze_file(path)
        assert result.reader == "builtin-stl" and result.warnings == []
        assert ["reader", "builtin-stl", ""] in csv.reader(std_io.StringIO(result.to_csv()))


def test_ascii_variants_are_read(tmp_path, cube_triangles):
    text = ascii_stl(cube_triangles)
    half = text.index(b" facet", len(text) // 2)
    variants = {
        "latin1_name": text.replace(b"solid test", "solid pièce 1".encode("latin-1"), 1),
        "cp1252_name": text.replace(b"solid test", "solid Teil – 1".encode("cp1252"), 1),
        "crlf_upper": text.replace(b"\n", b"\r\n").upper(),
        "bom_tabs": b"\xef\xbb\xbf" + text.replace(b" ", b"\t"),
        "two_solids": text[:half] + b"endsolid a\nsolid b\n" + text[half:],
    }
    expected = load_stl(_write(tmp_path / "plain.stl", text))
    for name, data in variants.items():
        nodes, faces = load_stl(_write(tmp_path / f"{name}.stl", data), strict=True)
        assert np.array_equal(nodes, expected[0]) and np.array_equal(faces, expected[1]), name


def _write(path, data):
    path.write_bytes(data)
    return path


def _malformed(binary):
    n = struct.unpack_from("<I", binary, 80)[0]
    return {
        "count_too_large": (binary[:80] + struct.pack("<I", n + 5) + binary[84:],
                            f"header gives {n + 5} triangles"),
        "count_zero": (binary[:80] + struct.pack("<I", 0) + binary[84:], "header gives 0 triangles"),
        "trailing_bytes": (binary + b"\0\0", f"but the file has {len(binary) + 2} bytes"),
        "truncated": (binary[:-10], f"but the file has {len(binary) - 10} bytes"),
        "header_only": (binary[:40], "fewer than the 84-byte header"),
    }


def test_malformed_binary_gives_the_exact_reason(tmp_path, cube_triangles, monkeypatch):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    for name, (data, reason) in _malformed(binary_stl(cube_triangles)).items():
        with pytest.raises(io.STLParseError, match="Could not parse STL file: binary STL") as info:
            load_geometry(_write(tmp_path / f"{name}.stl", data))
        assert reason in str(info.value), name
        assert "pyshape[mesh]" in str(info.value)


def test_trimesh_empty_result_is_a_failure(tmp_path, cube_triangles):
    trimesh = pytest.importorskip("trimesh")
    for name, (data, reason) in _malformed(binary_stl(cube_triangles)).items():
        path = _write(tmp_path / f"{name}.stl", data)
        with pytest.raises(io.STLParseError) as info:
            load_geometry(path)
        message = str(info.value)
        assert reason in message and f"trimesh {trimesh.__version__} could not read it" in message


@pytest.mark.parametrize("kind", ["binary", "ascii"])
def test_non_finite_coordinates_are_an_error_without_fallback(tmp_path, cube_triangles,
                                                               no_trimesh_calls, kind):
    triangles = cube_triangles.copy()
    triangles[3, 1, 2] = np.nan if kind == "binary" else np.inf
    data = binary_stl(triangles) if kind == "binary" else ascii_stl(triangles)
    path = _write(tmp_path / "bad.stl", data)
    with pytest.raises(ValueError, match="non-finite coordinate .* in triangle 4") as info:
        load_geometry(path)
    assert not isinstance(info.value, io.STLParseError)


@pytest.mark.parametrize("text, reason", [
    ("solid x\nvertex 0 0\nendsolid x", "line 2: a vertex must have exactly three coordinates"),
    ("solid x\nvertex 0 0 zero\nvertex 1 0 0\nvertex 0 1 0\n", "line 2: invalid coordinate"),
    ("solid x\nvertex 0 0 0\nvertex 1 0 0\n", "incomplete triangle"),
    ("solid x\nendsolid x\n", "no triangles found"),
])
def test_malformed_ascii_gives_the_exact_reason(tmp_path, monkeypatch, text, reason):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    with pytest.raises(io.STLParseError, match="ASCII STL") as info:
        load_stl(_write(tmp_path / "bad.stl", text.encode()))
    assert reason in str(info.value)


# --- fallback, warning and strict mode ---------------------------------------

def test_fallback_is_recorded_with_a_warning(stl_path, broken_parser):
    trimesh = pytest.importorskip("trimesh")
    geometry = load_geometry(stl_path)
    assert geometry.reader == "trimesh-fallback"
    assert geometry.reader_version == trimesh.__version__
    assert len(geometry.warnings) == 1 and REASON in geometry.warnings[0]
    result = analyze_file(stl_path)
    assert result.reader == "trimesh-fallback" and result.warnings == geometry.warnings
    assert result.to_dict()["warnings"] == geometry.warnings
    rows = list(csv.reader(std_io.StringIO(result.to_csv())))
    assert rows[1:4] == [["source_file", stl_path.name, ""],
                         ["reader", "trimesh-fallback", ""],
                         ["reader_version", trimesh.__version__, ""]]
    assert rows[4][0] == "warning" and REASON in rows[4][1]
    assert geo_metadata_rows(geometry) == [tuple(row) for row in rows[2:5]]
    assert "fallback" in reader_text(geometry.reader, geometry.reader_version)
    with pytest.warns(io.STLFallbackWarning, match="rejected it"):
        load_stl(stl_path)


def test_fallback_gives_the_same_mesh_as_the_parser(stl_path, monkeypatch):
    pytest.importorskip("trimesh")
    expected = load_geometry(stl_path)
    monkeypatch.setattr(io, "_parse_stl", lambda data: (_ for _ in ()).throw(
        io.STLParseError(REASON)))
    rescued = load_geometry(stl_path)
    assert rescued.reader == "trimesh-fallback"
    assert np.array_equal(rescued.nodes, expected.nodes)
    assert np.array_equal(rescued.faces, expected.faces)


def test_strict_stl_raises_the_parser_error(stl_path, broken_parser):
    pytest.importorskip("trimesh")
    with pytest.raises(io.STLParseError, match="strict STL reading: trimesh not tried"):
        load_geometry(stl_path, strict_stl=True)
    with pytest.raises(io.STLParseError, match=re.escape(REASON)):
        analyze_file(stl_path, strict_stl=True)
    with pytest.raises(io.STLParseError), warnings.catch_warnings():
        warnings.simplefilter("error")
        load_stl(stl_path, strict=True)


def test_without_trimesh_the_parser_error_is_raised(stl_path, broken_parser, monkeypatch):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    with pytest.raises(io.STLParseError, match=re.escape(REASON)) as info:
        load_geometry(stl_path)
    assert "trimesh, which could try another reader, is not installed" in str(info.value)


def test_batch_reports_the_fallback(tmp_path, stl_path, broken_parser, capsys):
    pytest.importorskip("trimesh")
    (tmp_path / "p.stl").write_bytes(stl_path.read_bytes())
    batch = analyze_folder(tmp_path, summary=True)
    (item,) = batch.items
    assert item.ok and item.result.reader == "trimesh-fallback"
    assert REASON in item.result.warnings[0]
    csv_rows = list(csv.reader(std_io.StringIO(item.csv_path.read_text(encoding="utf-8"))))
    assert ["reader", "trimesh-fallback", ""] in csv_rows
    summary = list(csv.reader(std_io.StringIO(batch.summary_path.read_text(encoding="utf-8"))))
    assert summary[0][-2:] == ["reader", "warnings"]
    assert summary[1][-2] == "trimesh-fallback" and REASON in summary[1][-1]
    assert batch_main([str(tmp_path)]) == 0
    assert "warning: STL file read by trimesh" in capsys.readouterr().out
    strict = analyze_folder(tmp_path, strict_stl=True, write_csv=False)
    assert not strict.succeeded and "strict STL reading" in strict.items[0].error
    assert batch_main([str(tmp_path), "--strict-stl"]) == 1
    assert "FAILED: STLParseError" in capsys.readouterr().out


def test_other_formats_record_trimesh_without_warning(tmp_path, cube_mesh):
    trimesh = pytest.importorskip("trimesh")
    nodes, faces = cube_mesh
    path = tmp_path / "cube.obj"
    trimesh.Trimesh(nodes, faces, process=False).export(path)
    result = analyze_file(path)
    assert result.reader == "trimesh" and result.reader_version == trimesh.__version__
    assert result.warnings == []


def test_arrays_have_no_reader():
    result = analyze_mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]))
    assert result.reader is None and result.warnings == []
    assert [row[0] for row in csv.reader(std_io.StringIO(result.to_csv()))][1] != "reader"


# --- agreement with trimesh on the example and validation meshes ------------

def _trimesh_arrays(trimesh, path):
    """The STL reading used before issue #20: trimesh with its processing."""
    mesh = trimesh.load(str(path), process=True, force="mesh")
    mesh.merge_vertices()
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int32)


def _validation_meshes():
    pytest.importorskip("scipy")
    sys.path.insert(0, str(REPO / "validation"))
    try:
        import meshes
    except ImportError:
        pytest.skip("validation meshes not available")
    finally:
        sys.path.pop(0)
    for name in meshes.PLATONIC:
        nodes, faces, _ = meshes.platonic(name, subdivisions=2)
        yield name, nodes, faces
    yield "prolate", *meshes.ellipsoid((2, 1, 1), 3)
    yield "torus", *meshes.torus(3.0, 1.0, 60, 20)
    yield "random", *meshes.random_particle(3, subdivisions=3, roughness=0.3)


def _is_multi_component(faces):
    """True if the triangles form more than one vertex-connected component."""
    parent = np.arange(int(faces.max()) + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in faces:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra
        parent[find(rc)] = ra
    return len({find(i) for i in np.unique(faces)}) > 1


def test_builtin_parser_and_trimesh_agree(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    has_union = importlib.util.find_spec("manifold3d") is not None
    files = sorted(ASSETS.glob("*.stl"))
    for name, nodes, faces in _validation_meshes():
        triangles = nodes[faces]
        files.append(_write(tmp_path / f"{name}_bin.stl", binary_stl(triangles)))
        files.append(_write(tmp_path / f"{name}_ascii.stl",
                            ascii_stl(triangles.astype(np.float32).astype(float))))
    assert len(files) >= 3 + 2 * 8
    for path in files:
        builtin = load_geometry(path)
        assert builtin.reader == "builtin-stl", path.name
        nodes, faces = _trimesh_arrays(trimesh, path)
        assert np.array_equal(builtin.nodes, nodes), path.name
        assert np.array_equal(builtin.faces, faces), path.name
        if not has_union and _is_multi_component(faces):
            continue  # analysing it needs the solid union (manifold3d)
        for axes in ("pca", "min_volume"):
            assert (analyze_mesh(builtin.nodes, builtin.faces, axes=axes).values
                    == analyze_mesh(nodes, faces, axes=axes).values), (path.name, axes)
