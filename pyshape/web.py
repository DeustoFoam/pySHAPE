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

from . import __version__
from .analysis import AnalysisResult, analyze_mesh, prepare_mesh, scale_nodes
from .io import load_geometry

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_PREVIEW_FACES = 5000


def _analyze_upload(
    file: UploadFile, units: str, axes: str = "pca",
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
            result = analyze_mesh(geometry.nodes, geometry.faces, units=units, axes=axes)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        result.source_name = name
        nodes, _ = scale_nodes(geometry.nodes, units)
        return result, nodes, geometry.faces


def _cluster_vertices(nodes: np.ndarray, faces: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex-clustering simplification on a cells^3 grid (display only)."""
    lo = nodes.min(axis=0)
    size = float(np.ptp(nodes, axis=0).max()) / cells
    keys = np.floor((nodes - lo) / size).astype(np.int64)
    _, cluster = np.unique(keys, axis=0, return_inverse=True)
    cluster = cluster.ravel()
    counts = np.bincount(cluster)
    centers = np.stack([np.bincount(cluster, weights=nodes[:, k]) for k in range(3)], axis=1)
    centers /= counts[:, None]
    mapped = cluster[faces]
    keep = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2]) & (mapped[:, 0] != mapped[:, 2])
    mapped = mapped[keep]
    _, unique = np.unique(np.sort(mapped, axis=1), axis=0, return_index=True)
    return centers, mapped[np.sort(unique)]


def _preview(nodes: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    """Triangles for the browser preview, simplified when the mesh is large.

    Large meshes are simplified by vertex clustering (finest grid that keeps at
    most MAX_PREVIEW_FACES faces), which keeps the surface closed; evenly
    sampled faces are the fallback. Calculations always use the full mesh.
    """
    nodes, faces = prepare_mesh(nodes, faces)
    shown_nodes, shown_faces = nodes, faces
    if len(faces) > MAX_PREVIEW_FACES:
        best = None
        low, high = 2, 1024
        while low <= high:
            cells = (low + high) // 2
            candidate = _cluster_vertices(nodes, faces, cells)
            if 0 < len(candidate[1]) <= MAX_PREVIEW_FACES:
                best, low = candidate, cells + 1
            else:
                high = cells - 1
        if best is not None:
            shown_nodes, shown_faces = best
        else:
            selected = np.linspace(0, len(faces) - 1, MAX_PREVIEW_FACES, dtype=int)
            shown_faces = faces[selected]
    mins, maxs = nodes.min(axis=0), nodes.max(axis=0)
    center = (mins + maxs) / 2.0
    span = float((maxs - mins).max())
    triangles = ((shown_nodes[shown_faces] - center) / span).tolist()
    return {
        "triangles": triangles,
        "total_faces": len(faces),
        "shown_faces": len(shown_faces),
    }


def create_app() -> FastAPI:
    """Create an ASGI app; no server is started when importing this module."""
    application = FastAPI(title="pySHAPE", version=__version__)
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
        axes: str = Form("pca"),
    ) -> dict[str, object]:
        """Upload STL and receive metadata and metric records, with optional preview.

        ``axes`` is ``pca`` (default) or ``min_volume`` for the S, I, L dimensions.
        """
        result, nodes, faces = _analyze_upload(file, units, axes)
        payload = result.to_dict()
        if include_preview:
            payload["preview"] = _preview(nodes, faces)
        return payload

    @application.post("/api/export")
    def export(
        file: UploadFile = File(...),
        units: str = Form("as-is"),
        keys: Optional[str] = Form(None),
        axes: str = Form("pca"),
    ) -> Response:
        """Analyze STL and download CSV; keys is a comma-separated metric selection."""
        result, _, _ = _analyze_upload(file, units, axes)
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
