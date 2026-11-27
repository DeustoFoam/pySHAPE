"""Uploads of every supported mesh format to the web API and pages."""
import base64
import csv
import io
import json
import re
import struct
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from pyshape import io as pyshape_io
from pyshape import web

STATIC = Path(web.__file__).with_name("static")


@pytest.fixture
def client():
    with TestClient(web.create_app()) as client:
        yield client


def _mesh_bytes(cube_mesh, file_type):
    trimesh = pytest.importorskip("trimesh")
    nodes, faces = cube_mesh
    return trimesh.Trimesh(nodes, faces, process=False).export(file_type=file_type)


def _values(payload):
    return {m["key"]: m["value"] for m in payload["metrics"]}


def _upload(client, path, name, data, **form):
    return client.post(path, files={"file": (name, data)}, data=form)


@pytest.mark.parametrize("file_type", ["obj", "ply", "off", "glb"])
def test_other_formats_give_the_same_results_as_stl(client, cube_mesh, file_type):
    stl = _upload(client, "/api/analyze", "cube.stl", _mesh_bytes(cube_mesh, "stl"), units="mm")
    other = _upload(client, "/api/analyze", f"cube.{file_type}",
                    _mesh_bytes(cube_mesh, file_type), units="mm")
    assert stl.status_code == 200 and other.status_code == 200, other.text
    assert other.json()["source_name"] == f"cube.{file_type}"
    expected, got = _values(stl.json()), _values(other.json())
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        if value is None:  # unavailable in both (e.g. cavity results with cavities="fill")
            assert got[key] is None, key
        else:
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-15), key
    assert got["volume"] == pytest.approx(1e-9)  # 1 mm^3 in m^3


@pytest.mark.parametrize("file_type", ["obj", "ply"])
def test_export_of_other_formats_matches_stl(client, cube_mesh, file_type):
    form = {"keys": "surface_area,volume,convexity"}
    stl = _upload(client, "/api/export", "cube.stl", _mesh_bytes(cube_mesh, "stl"), **form)
    other = _upload(client, "/api/export", f"cube.{file_type}", _mesh_bytes(cube_mesh, file_type), **form)
    assert other.status_code == 200, other.text
    metadata = {"source_file", "reader", "reader_version"}
    rows = lambda r: [row for row in csv.reader(io.StringIO(r.text)) if row[0] not in metadata]
    assert rows(other) == rows(stl)
    assert ["reader", "trimesh", ""] in csv.reader(io.StringIO(other.text))
    assert ["reader", "builtin-stl", ""] in csv.reader(io.StringIO(stl.text))


def test_uppercase_extension_is_accepted(client, cube_mesh):
    response = _upload(client, "/api/analyze", "CUBE.OBJ", _mesh_bytes(cube_mesh, "obj"))
    assert response.status_code == 200, response.text
    assert response.json()["source_name"] == "CUBE.OBJ"


def test_unsupported_extension_lists_the_accepted_ones(client):
    response = _upload(client, "/api/analyze", "particle.xyz", b"1 2 3")
    assert response.status_code == 415
    for extension in web.WEB_EXTENSIONS:
        assert extension in response.json()["detail"]


def test_unreadable_file_is_a_client_error(client):
    pytest.importorskip("trimesh")
    response = _upload(client, "/api/analyze", "broken.ply", b"ply\nformat nonsense\n")
    assert response.status_code == 422
    assert response.json()["detail"]


def test_non_stl_without_trimesh_names_the_extra(client, cube_mesh, monkeypatch):
    data = _mesh_bytes(cube_mesh, "obj")
    monkeypatch.setattr(pyshape_io, "HAS_TRIMESH", False)
    response = _upload(client, "/api/analyze", "cube.obj", data)
    assert response.status_code == 503
    assert "trimesh" in response.json()["detail"]
    assert "pyshape[mesh]" in response.json()["detail"]
    stl = _upload(client, "/api/analyze", "cube.stl", _mesh_bytes(cube_mesh, "stl"))
    assert stl.status_code == 200  # STL still works with the built-in parser


def test_stl_reader_is_reported(client, cube_mesh):
    response = _upload(client, "/api/analyze", "cube.stl", _mesh_bytes(cube_mesh, "stl"))
    assert response.status_code == 200
    payload = response.json()
    assert payload["reader"] == "builtin-stl"
    assert payload["reader_version"] is None and payload["warnings"] == []


def test_stl_fallback_warning_in_api(client, cube_mesh, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    data = _mesh_bytes(cube_mesh, "stl")
    reason = "binary STL: the header gives 99 triangles"

    def reject(data):
        raise pyshape_io.STLParseError(reason)

    monkeypatch.setattr(pyshape_io, "_parse_stl", reject)  # simulate a file the parser rejects
    response = _upload(client, "/api/analyze", "cube.stl", data, include_csv="true")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["reader"] == "trimesh-fallback"
    assert payload["reader_version"] == trimesh.__version__
    assert len(payload["warnings"]) == 1 and reason in payload["warnings"][0]
    assert _values(payload)["volume"] == pytest.approx(1.0)
    rows = list(csv.reader(io.StringIO(payload["csv"])))
    assert ["reader", "trimesh-fallback", ""] in rows
    assert any(row[0] == "warning" and reason in row[1] for row in rows)
    export = _upload(client, "/api/export", "cube.stl", data)
    assert any(row[0] == "warning" for row in csv.reader(io.StringIO(export.text)))
    monkeypatch.setattr(pyshape_io, "HAS_TRIMESH", False)
    failed = _upload(client, "/api/analyze", "cube.stl", data)
    assert failed.status_code == 422 and reason in failed.json()["detail"]


def test_missing_format_package_is_named(client, monkeypatch):
    pytest.importorskip("trimesh")

    def needs_package(path):
        raise ImportError("missing `pip install pycollada`")

    monkeypatch.setattr(web, "load_geometry", needs_package)
    response = _upload(client, "/api/analyze", "cube.dae", b"<COLLADA/>")
    assert response.status_code == 503
    assert "pycollada" in response.json()["detail"]
    assert "trimesh, which is not installed" not in response.json()["detail"]


def _gltf(cube_mesh, uri):
    nodes, faces = cube_mesh
    blob = nodes.astype(np.float32).tobytes() + faces.astype(np.uint32).tobytes()
    uri = uri or "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
    n_pos = len(nodes) * 12
    document = {
        "asset": {"version": "2.0"}, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}],
        "buffers": [{"uri": uri, "byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": n_pos},
                        {"buffer": 0, "byteOffset": n_pos, "byteLength": len(blob) - n_pos}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(nodes), "type": "VEC3",
             "min": nodes.min(axis=0).tolist(), "max": nodes.max(axis=0).tolist()},
            {"bufferView": 1, "componentType": 5125, "count": faces.size, "type": "SCALAR"},
        ],
    }
    return json.dumps(document).encode()


def test_web_accepts_library_formats_except_text_gltf():
    assert ".gltf" in pyshape_io.SUPPORTED_EXTENSIONS  # the library still reads it
    assert web.WEB_EXTENSIONS == tuple(
        e for e in pyshape_io.SUPPORTED_EXTENSIONS if e != ".gltf")


@pytest.mark.parametrize("uri", [None, "cube.bin", "../secret.bin", "file:///etc/passwd",
                                 "http://127.0.0.1:9/x.bin"])
def test_gltf_is_not_accepted(client, cube_mesh, uri):
    for name in ("cube.gltf", "CUBE.GLTF"):
        response = _upload(client, "/api/analyze", name, _gltf(cube_mesh, uri))
        assert response.status_code == 415
        assert "Unsupported file type '.gltf'" in response.json()["detail"]


def test_glb_with_external_buffer_is_rejected(client):
    document = json.dumps({"asset": {"version": "2.0"},
                           "buffers": [{"uri": "outside.bin", "byteLength": 4}]}).encode()
    document += b" " * (-len(document) % 4)
    glb = b"glTF" + struct.pack("<II", 2, 12 + 8 + len(document))
    glb += struct.pack("<II", len(document), 0x4E4F534A) + document
    response = _upload(client, "/api/analyze", "cube.glb", glb)
    assert response.status_code == 415


@pytest.mark.parametrize("data", [b"{not a glb}", b"glTF" + b"\x00" * 4])
def test_invalid_glb_is_a_client_error(client, data):
    response = _upload(client, "/api/analyze", "broken.glb", data)
    assert response.status_code == 422


def test_pages_accept_every_supported_extension():
    expected = list(web.WEB_EXTENSIONS)
    assert ".gltf" not in expected
    for page in ("index.html", "batch.html"):
        accepts = re.findall(r'accept="([^"]+)"', (STATIC / page).read_text(encoding="utf-8"))
        assert accepts and all(a.split(",") == expected for a in accepts), page
        assert "STL file" not in (STATIC / page).read_text(encoding="utf-8")
    script = (STATIC / "batch.js").read_text(encoding="utf-8")
    listed = re.search(r"const EXTENSIONS = \[([^\]]+)\]", script).group(1)
    assert [e.strip().strip('"') for e in listed.split(",")] == expected
    assert "stl" not in script.lower().replace('".stl"', "")
