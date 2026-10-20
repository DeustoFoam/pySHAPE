import csv
import io
import sys

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient

from pyshape import analyze_file
from pyshape import web


@pytest.fixture
def client():
    with TestClient(web.create_app()) as client:
        yield client


def test_static_app_and_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert "Calculate results" in client.get("/").text
    for name in ["app.js", "style.css"]:
        response = client.get(f"/static/{name}")
        assert response.status_code == 200
        assert response.text
    assert client.get("/docs").status_code == 200
    assert "/api/analyze" in client.get("/openapi.json").json()["paths"]


def test_api_matches_library(client, stl_path):
    response = client.post(
        "/api/analyze",
        files={"file": (stl_path.name, stl_path.read_bytes())},
        data={"units": "mm"},
    )
    assert response.status_code == 200
    assert response.json() == analyze_file(stl_path, units="mm").to_dict()
    assert "preview" not in response.json()


def test_preview_is_bounded(client, stl_path, monkeypatch):
    monkeypatch.setattr(web, "MAX_PREVIEW_FACES", 2)
    response = client.post(
        "/api/analyze",
        files={"file": ("../../particle.stl", stl_path.read_bytes())},
        data={"include_preview": "true"},
    )
    result = response.json()
    assert result["source_name"] == "particle.stl"
    assert result["preview"]["shown_faces"] == 2
    assert result["preview"]["total_faces"] == 4
    assert len(result["preview"]["triangles"]) == 2
    assert len(result["preview"]["triangles"][0]) == 3
    assert next(m["value"] for m in result["metrics"] if m["key"] == "n_faces") == 4


def test_csv_matches_library(client, stl_path):
    response = client.post(
        "/api/export",
        files={"file": (stl_path.name, stl_path.read_bytes())},
        data={"keys": "surface_area,volume"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    expected = analyze_file(stl_path).to_csv(["surface_area", "volume"])
    assert list(csv.reader(io.StringIO(response.text))) == list(csv.reader(io.StringIO(expected)))


@pytest.mark.parametrize("path", ["/api/analyze", "/api/export"])
@pytest.mark.parametrize("filename,data,code", [
    ("bad.obj", b"not STL", 415),
    ("empty.stl", b"", 422),
    ("bad.stl", b"not STL", 422),
])
def test_invalid_uploads(client, path, filename, data, code):
    response = client.post(path, files={"file": (filename, data)})
    assert response.status_code == code
    assert response.json()["detail"]


def test_invalid_units_and_selection(client, stl_path):
    files = {"file": (stl_path.name, stl_path.read_bytes())}
    assert client.post("/api/analyze", files=files, data={"units": "invalid"}).status_code == 422
    response = client.post("/api/export", files=files, data={"keys": "invalid"})
    assert response.status_code == 422
    assert "Unknown metric" in response.json()["detail"]
    assert client.post("/api/analyze").status_code == 422


def test_upload_limit(client, stl_path, monkeypatch):
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 16)
    response = client.post("/api/analyze", files={"file": ("mesh.stl", stl_path.read_bytes())})
    assert response.status_code == 413


def test_open_surface_has_explicit_errors(client):
    triangle = b"solid t\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendsolid t\n"
    response = client.post("/api/analyze", files={"file": ("open.stl", triangle)})
    assert response.status_code == 200
    volume = next(m for m in response.json()["metrics"] if m["key"] == "volume")
    assert volume["value"] is None
    assert "closed" in volume["error"]


def test_multisolid_api_and_preview_use_union(client, cube_mesh, monkeypatch):
    pytest.importorskip("manifold3d")
    import numpy as np
    import trimesh

    nodes, faces = cube_mesh
    mesh = trimesh.Trimesh(
        vertices=np.vstack((nodes, nodes + [0.5, 0, 0])),
        faces=np.vstack((faces, faces[:, ::-1] + len(nodes))),
        process=False,
    )
    files = {"file": ("multisolid.stl", mesh.export(file_type="stl"))}
    response = client.post("/api/analyze", files=files, data={"include_preview": "true"})
    assert response.status_code == 200
    payload = response.json()
    values = {m["key"]: m["value"] for m in payload["metrics"]}
    assert values["volume"] == pytest.approx(1.5, rel=1e-6)
    assert values["surface_area"] == pytest.approx(8, rel=1e-6)
    assert payload["preview"]["total_faces"] == values["n_faces"]
    exported = client.post("/api/export", files=files, data={"keys": "volume,surface_area"})
    assert exported.status_code == 200
    rows = dict(csv.reader(io.StringIO(exported.text)))
    assert float(rows["volume"]) == pytest.approx(1.5, rel=1e-6)
    assert float(rows["surface_area"]) == pytest.approx(8, rel=1e-6)
    monkeypatch.setitem(sys.modules, "manifold3d", None)
    for endpoint in ("/api/analyze", "/api/export"):
        unavailable = client.post(endpoint, files=files)
        assert unavailable.status_code == 503
        assert "pyshape[mesh]" in unavailable.json()["detail"]
