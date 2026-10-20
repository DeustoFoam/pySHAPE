import struct

import numpy as np
import pytest

from pyshape import io


def test_builtin_ascii_loader(stl_path, monkeypatch):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    nodes, faces = io.load_stl(stl_path)
    assert nodes.shape == (4, 3)
    assert faces.shape == (4, 3)
    assert np.isfinite(nodes).all()


def test_builtin_binary_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    path = tmp_path / "triangle.stl"
    header = b"binary STL".ljust(80, b"\0") + struct.pack("<I", 1)
    facet = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    path.write_bytes(header + facet)
    nodes, faces = io.load_stl(path)
    assert nodes.shape == (3, 3)
    assert faces.shape == (1, 3)


@pytest.mark.parametrize("text", [
    "solid bad\nvertex nope 0 0\nendsolid bad",
    "vertex 0 0\nvertex 1 0 0\nvertex 0 1 0",
    "vertex nan 0 0\nvertex 1 0 0\nvertex 0 1 0",
    "vertex 0 0 0\nvertex 1 0 0",
])
def test_builtin_loader_rejects_malformed_vertices(tmp_path, monkeypatch, text):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    path = tmp_path / "bad.stl"
    path.write_text(text)
    with pytest.raises(ValueError, match="Could not parse STL"):
        io.load_stl(path)


def test_optional_mesh_loader_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "HAS_TRIMESH", False)
    path = tmp_path / "mesh.obj"
    path.write_text("v 0 0 0")
    with pytest.raises(ImportError, match=r"pyshape\[mesh\]"):
        io.load_geometry(path)


def test_obj_loader(tmp_path):
    pytest.importorskip("trimesh")
    path = tmp_path / "triangle.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    geometry = io.load_geometry(path)
    assert geometry.source_format == "obj"
    assert geometry.converted_from_native
    assert geometry.faces.shape == (1, 3)
