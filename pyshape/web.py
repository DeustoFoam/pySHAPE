"""Optional local HTTP API and HTML viewer; install ``pyshape[web]``."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import AnalysisResult, analyze_mesh, prepare_mesh, scale_nodes
from .io import load_geometry

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_PREVIEW_FACES = 5000


def _analyze_upload(
    file: UploadFile, units: str,
) -> tuple[AnalysisResult, np.ndarray, np.ndarray]:
    name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    # Only self-contained STL uploads are accepted; other loaders may resolve
    # companion files. The Python API still supports every existing format.
    if Path(name).suffix.lower() != ".stl":
        raise HTTPException(415, "The web interface accepts STL files only.")
    with TemporaryDirectory(prefix="pyshape-") as directory:
        path = Path(directory) / "mesh.stl"
        size = 0
        with path.open("wb") as stream:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "STL upload exceeds the 32 MiB limit.")
                stream.write(chunk)
        if size == 0:
            raise HTTPException(422, "The uploaded STL file is empty.")
        try:
            geometry = load_geometry(path)
            result = analyze_mesh(geometry.nodes, geometry.faces, units=units)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        result.source_name = name
        nodes, _ = scale_nodes(geometry.nodes, units)
        return result, nodes, geometry.faces


def _preview(nodes: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    nodes, faces = prepare_mesh(nodes, faces)
    selected = np.linspace(0, len(faces) - 1, min(len(faces), MAX_PREVIEW_FACES), dtype=int)
    mins, maxs = nodes.min(axis=0), nodes.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    triangles = ((nodes[faces[selected]] - center) / span).tolist()
    return {
        "triangles": triangles,
        "total_faces": len(faces),
        "shown_faces": len(selected),
    }


def create_app() -> FastAPI:
    """Create an ASGI app; no server is started when importing this module."""
    application = FastAPI(title="pySHAPE", version="0.1.0")
    static = Path(__file__).with_name("static")
    application.mount("/static", StaticFiles(directory=static), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/api/analyze")
    def analyze(
        file: UploadFile = File(...),
        units: str = Form("as-is"),
        include_preview: bool = Form(False),
    ) -> dict[str, object]:
        """Upload STL and receive metadata and metric records, with optional preview."""
        result, nodes, faces = _analyze_upload(file, units)
        payload = result.to_dict()
        if include_preview:
            payload["preview"] = _preview(nodes, faces)
        return payload

    @application.post("/api/export")
    def export(
        file: UploadFile = File(...),
        units: str = Form("as-is"),
        keys: Optional[str] = Form(None),
    ) -> Response:
        """Analyze STL and download CSV; keys is a comma-separated metric selection."""
        result, _, _ = _analyze_upload(file, units)
        try:
            text = result.to_csv(None if keys is None else keys.split(","))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return Response(
            text,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="results.csv"'},
        )

    return application


app = create_app()


def main() -> None:
    """Serve locally by default; remote deployment needs separate hardening."""
    import uvicorn

    parser = argparse.ArgumentParser(description="pySHAPE local API and browser viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("pyshape.web:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
