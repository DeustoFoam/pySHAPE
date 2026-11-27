"""Optional local HTTP API and HTML viewer; install ``pyshape[web]``."""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .analysis import (
    AXES_LABELS, LENGTH_POWERS, OUTPUT_UNITS, UNIT_SCALES,
    AnalysisResult, analyze_mesh, attach_source, compute_metrics, scale_nodes,
)
from . import descriptors
from . import io as io_module
from .io import NATIVE_EXTENSIONS, SUPPORTED_EXTENSIONS, load_geometry
from .preprocess import prepare_mesh

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MAX_PREVIEW_FACES = 5000
# The web accepts the library formats except the text .gltf, which usually
# keeps its buffers in separate files (the self-contained .glb is accepted).
WEB_EXTENSIONS = tuple(e for e in SUPPORTED_EXTENSIONS if e != ".gltf")
ACCEPTED = ", ".join(WEB_EXTENSIONS)  # for messages and /docs


# --- Documentation-only models: they describe the responses in /docs and
# /openapi.json; the endpoints keep returning the library's own dictionaries.

class MetricRecord(BaseModel):
    key: str = Field(description="Stable identifier, used in `keys` and in CSV files.")
    label: str = Field(description="Human-readable name.")
    value: Optional[float] = Field(description="Result, or null when unavailable (see `error`).")
    group: str = Field(description="Group shown in the interfaces, by what the descriptor depends on, e.g. \"Size and mass properties\" or \"Form from S, I, L (PCA box)\".")
    default_selected: bool = Field(description="Exported when no `keys` are given.")
    error: Optional[str] = Field(description="Why the value is unavailable, e.g. open mesh.")
    unit: str = Field(description="Unit of the value, empty for dimensionless results.")


class Preview(BaseModel):
    triangles: list[list[list[float]]] = Field(
        description="Triangles for display, centred and scaled to unit size; "
        f"simplified when the mesh has more than {MAX_PREVIEW_FACES} faces.")
    total_faces: int = Field(description="Faces of the analysed mesh (after the solid union).")
    shown_faces: int = Field(description="Faces in `triangles`.")


class AnalysisResponse(BaseModel):
    metrics: list[MetricRecord] = Field(description="All 30 results, in a fixed order.")
    source_name: Optional[str] = Field(description="Uploaded file name, without folders.")
    input_units: str = Field(description="Units of the coordinates in the file.")
    scale_factor: float = Field(description="Metres per input unit (1 for as-is).")
    axes_method: str = Field(description="How S, I and L were measured: pca, min_volume, inertia_ellipsoid or fitted_ellipsoid.")
    output_units: str = Field(description="Length unit of the results, or as-is.")
    output_scale: float = Field(description="Factor applied to lengths after conversion to metres.")
    reader: Optional[str] = Field(description="Reader of the file: `builtin-stl` (STL, built-in "
                                  "parser), `trimesh` (other formats) or `trimesh-fallback` (an "
                                  "STL file rejected by the built-in parser and read by trimesh).")
    reader_version: Optional[str] = Field(description="trimesh version when trimesh read the "
                                          "file, otherwise null.")
    warnings: list[str] = Field(description="Notes about the input, e.g. why an STL file was "
                                "read by the trimesh fallback. Empty for most files.")
    preview: Optional[Preview] = Field(None, description="Only with `include_preview=true`.")
    csv: Optional[str] = Field(None, description="Only with `include_csv=true`: the CSV text "
                               "that `/api/export` returns for the same request.")


class ErrorMessage(BaseModel):
    detail: str = Field(description="What went wrong.")


class HelpEntry(BaseModel):
    short: str = Field(description="One line, shown as a tooltip.")
    description: str = Field(description="A few sentences on what it is and how to read it.")
    formula: str = Field(description="Definition in plain text, empty when there is none.")
    depends_on: str = Field(description="What it needs, e.g. a closed mesh or the bounding box.")
    references: list[str] = Field(description="Keys of `references`.")


class Reference(BaseModel):
    citation: str = Field(description="Authors, year and source.")
    url: str = Field(description="DOI link, empty when not given.")


class DescriptorHelp(BaseModel):
    metrics: dict[str, HelpEntry] = Field(description="One entry per metric key.")
    controls: dict[str, HelpEntry] = Field(description="One entry per control of the "
                                           "interfaces, e.g. `input_units`, `axes`.")
    axes_methods: dict[str, HelpEntry] = Field(description="One entry per value of `axes`.")
    groups: dict[str, str] = Field(description="Optional line per result group; the part "
                                   "of a group name before ` (` is looked up too.")
    references: dict[str, Reference] = Field(description="Cited works, by key.")


class Health(BaseModel):
    status: str = Field(description="Always `ok` when the server is running.", examples=["ok"])


def _error(description: str) -> dict[str, object]:
    return {"model": ErrorMessage, "description": description}


UPLOAD_ERRORS = {
    413: _error(f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MiB."),
    415: _error(f"The file extension is not one of {ACCEPTED}, or a `.glb` file refers "
                "to external files (its buffers and images must be embedded)."),
    422: _error("Invalid mesh, or unknown `units`, `axes`, `output_units` or metric key. "
                "For a missing `file` or a malformed field, `detail` is instead a list of "
                "validation errors."),
    503: _error("A package is missing: trimesh, to read formats other than STL, or the "
                "solid-union packages, for a mesh with several closed shells. Both come "
                "with `pyshape[mesh]`."),
}

_POWER_TEXT = {1: "length", 2: "length²", 3: "length³", 5: "length⁵"}


def _metric_table() -> str:
    """Markdown table of every metric key, built from a reference tetrahedron."""
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    rows = ["| Key | Result | Group | Unit | Exported by default | Description |",
            "| --- | --- | --- | --- | --- | --- |"]
    for metric in compute_metrics(nodes, faces):
        unit = _POWER_TEXT.get(LENGTH_POWERS.get(metric.key, 0), "–")
        rows.append(f"| `{metric.key}` | {metric.label} | {metric.group} | {unit} | "
                    f"{'yes' if metric.default_selected else 'no'} | "
                    f"{(descriptors.metric_help(metric.key) or {}).get('short', '')} |")
    return "\n".join(rows)


def _choices(values) -> dict[str, object]:
    return {"json_schema_extra": {"enum": list(values)}}


UNITS_DOC = ("Units of the coordinates in the file. `as-is` keeps them unchanged, and the "
             "results are then in the file's own units.")
AXES_DOC = ("How the principal dimensions S, I, L are measured: `pca` (box aligned with the "
            "principal directions, fast), `min_volume` (minimum-volume box, "
            "orientation-independent, slower), `inertia_ellipsoid` (full axes of the "
            "ellipsoid with the same volume and principal moments of inertia) or "
            "`fitted_ellipsoid` (area-weighted least-squares ellipsoid through the surface "
            "vertices). Ellipsoid axes are not enclosing dimensions.")
OUTPUT_DOC = ("Length unit of the results: areas in its square, volumes in its cube, "
              "inertia per unit density in its fifth power. `same` follows `units`. "
              "Ignored when `units` is `as-is`.")
KEYS_DOC = ("Comma-separated metric keys to export, e.g. `volume,convexity` (see the table "
            "of metric keys above). Default: the keys exported by default. An unknown or "
            "unavailable key gives 422.")
FILE_DOC = (f"Mesh file, up to {MAX_UPLOAD_BYTES // 2**20} MiB: STL (ASCII or binary, read by "
            "pySHAPE's built-in parser, no extra package needed), or OBJ, PLY, OFF, GLB, 3MF or "
            f"DAE (read by trimesh). The reader is chosen by the extension ({ACCEPTED}, any "
            "case) and returned as `reader`.")

API_DESCRIPTION = f"""
Three-dimensional particle shape analysis: the same results as the pySHAPE Python
library (`pyshape.analyze_file`), for programs written in any language.

**How to use it.** Send a mesh file as `multipart/form-data` to `/api/analyze` (JSON
records) or `/api/export` (CSV). Every request is independent; uploads are processed
in temporary files that are deleted afterwards, and nothing is stored on the server.

```
curl -F "file=@particle.stl" -F "units=mm" http://127.0.0.1:8000/api/analyze
curl -F "file=@particle.stl" -F "units=mm" -F "keys=volume,convexity" \\
     http://127.0.0.1:8000/api/export -o particle.csv
```

**Formats and limits.** {ACCEPTED} (any case), up to {MAX_UPLOAD_BYTES // 2**20} MiB.
STL is the native format: it is always read by pySHAPE's built-in parser, which needs
no extra package. If the parser rejects an STL file and trimesh is installed, trimesh
is tried: the response then has `reader: "trimesh-fallback"` and a `warnings` entry
with the parser's reason. The other formats are read by trimesh (`reader: "trimesh"`).
Each upload must be a
single self-contained file: the text glTF format (`.gltf`), whose buffers are usually
separate files, is not accepted (export `.glb`), and references of OBJ or DAE files
to other files (materials, textures) are not available. Every mesh is then prepared in the same way (orientation of closed
shells, solid union). Volume-based results need a closed mesh; for open meshes they
are returned with `value: null` and an `error` instead of failing the request.

**Security.** The server is meant for local use. It has no authentication; do not expose
it to untrusted networks. Uploads other than STL are parsed by trimesh (version 5 or
later, which refuses references outside the upload's temporary folder).

The browser interface is at [/](/) (single particle) and [/batch](/batch) (a folder of
particles). Hover over a result or a control there for a one-line description, and
click a result's name for its definition, formula and references. The same texts are
served as JSON by `GET /api/descriptors`.

### Metric keys

{{table}}

The labels and group of `axis_S`, `axis_I` and `axis_L` name the method chosen with
`axes` (shown here for `pca`). Units follow `units` and `output_units`.
"""


def _glb_external_references(path: Path) -> list[str]:
    """URIs of buffers and images that a GLB file loads from other files.

    Embedded data (the binary chunk, or ``data:`` URIs) is allowed; anything
    else would need files that are not part of the upload.
    """
    data = path.read_bytes()
    if data[:4] != b"glTF" or len(data) < 20:  # 12-byte header, then the JSON chunk
        raise ValueError("The file is not a binary glTF (.glb) file.")
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A:  # "JSON"
        raise ValueError("The GLB file has no JSON chunk.")
    try:
        document = json.loads(data[20:20 + length])
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"The JSON chunk of the GLB file is not valid: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("The JSON chunk of the GLB file is not an object.")
    uris = []
    for group in ("buffers", "images"):
        for item in document.get(group) or []:
            uri = item.get("uri") if isinstance(item, dict) else None
            if isinstance(uri, str) and not uri.startswith("data:"):
                uris.append(uri)
    return uris


def _analyze_upload(
    file: UploadFile, units: str, axes: str = "pca", output_units: str = "m",
) -> tuple[AnalysisResult, np.ndarray, np.ndarray]:
    name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    # The reader is chosen by the extension (io.load_geometry); every mesh is
    # then prepared in the same way by analyze_mesh (prepare_mesh).
    extension = Path(name).suffix.lower()
    if extension not in WEB_EXTENSIONS:
        raise HTTPException(
            415, f"Unsupported file type '{extension or name}'. Accepted: {ACCEPTED}.")
    with TemporaryDirectory(prefix="pyshape-") as directory:
        # Keep the extension so that load_geometry picks the right reader. The
        # folder holds only this file, so references to other files cannot resolve.
        path = Path(directory) / f"mesh{extension}"
        size = 0
        with path.open("wb") as stream:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"Upload exceeds the {MAX_UPLOAD_BYTES // 2**20} MiB limit.")
                stream.write(chunk)
        if size == 0:
            raise HTTPException(422, "The uploaded file is empty.")
        if extension == ".glb":
            try:
                external = _glb_external_references(path)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if external:
                raise HTTPException(
                    415, f"The {extension} file refers to external files ({', '.join(external)}), "
                    "which cannot be uploaded with it. Export the .glb with embedded buffers "
                    "and images.")
        try:
            geometry = load_geometry(path)
        except ImportError as exc:
            if not io_module.HAS_TRIMESH:
                raise HTTPException(
                    503, f"Reading {extension} files needs trimesh, which is not installed on "
                    "the server. Install it with: pip install \"pyshape[mesh]\"") from exc
            # trimesh is there, but its reader for this format needs another package
            # (e.g. pycollada for .dae).
            raise HTTPException(
                503, f"Reading {extension} files needs a package that is not installed on "
                f"the server: {exc}") from exc
        except ValueError as exc:  # includes io.STLParseError
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:  # third-party readers raise many exception types
            if extension in NATIVE_EXTENSIONS:
                raise
            raise HTTPException(
                422, f"Could not read the {extension} file: {type(exc).__name__}: {exc}") from exc
        try:
            result = analyze_mesh(
                geometry.nodes, geometry.faces, units=units, axes=axes, output_units=output_units,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ImportError as exc:
            raise HTTPException(503, str(exc)) from exc
        attach_source(result, geometry, name)  # file name, reader and load warnings
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
    application = FastAPI(
        title="pySHAPE API",
        version=__version__,
        description=API_DESCRIPTION.replace("{table}", _metric_table()),
        license_info={"name": "GPL-3.0", "url": "https://www.gnu.org/licenses/gpl-3.0.html"},
        openapi_tags=[
            {"name": "analysis", "description": "Analyse an uploaded mesh file."},
            {"name": "help", "description": "Descriptions of the results and controls."},
            {"name": "service", "description": "Server status."},
        ],
    )
    static = Path(__file__).with_name("static")
    application.mount("/static", StaticFiles(directory=static), name="static")

    @application.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @application.get("/batch", include_in_schema=False)
    def batch() -> FileResponse:
        return FileResponse(static / "batch.html")

    @application.get(
        "/api/health", tags=["service"], summary="Check that the server is running",
        response_model=Health,
    )
    def health() -> dict[str, str]:
        """Returns `{"status": "ok"}`; use it to wait for the server to start."""
        return {"status": "ok"}

    @application.get(
        "/api/descriptors", tags=["help"],
        summary="Descriptions of the results and controls",
        response_model=DescriptorHelp,
    )
    def descriptor_help() -> dict[str, object]:
        """Returns the help texts of the interfaces: for every metric key and control a
        one-line `short` text, a `description`, the `formula`, what it `depends_on` and
        its `references`. Help is keyed by metric key, never by label or group."""
        return descriptors.as_dict()

    @application.post(
        "/api/analyze", tags=["analysis"], summary="Analyse a mesh file (JSON results)",
        responses={200: {"model": AnalysisResponse, "description": "All results of the particle."},
                   **UPLOAD_ERRORS},
    )
    def analyze(
        file: UploadFile = File(..., description=FILE_DOC),
        units: str = Form("as-is", description=UNITS_DOC, **_choices(UNIT_SCALES)),
        include_preview: bool = Form(False, description="Add `preview`: simplified "
                                     "triangles of the analysed surface, for display."),
        axes: str = Form("pca", description=AXES_DOC, **_choices(AXES_LABELS)),
        output_units: str = Form("m", description=OUTPUT_DOC,
                                 **_choices(["same", *OUTPUT_UNITS])),
        include_csv: bool = Form(False, description="Add `csv`: the CSV text that "
                                 "`/api/export` would return, so one request gives both."),
        keys: Optional[str] = Form(None, description="With `include_csv`: " + KEYS_DOC),
    ) -> dict[str, object]:
        """Returns every result of the particle as a record with key, label, group,
        value, unit and, when unavailable, the reason, together with the units and
        method used. The records match `pyshape.analyze_file(...).to_dict()`."""
        result, nodes, faces = _analyze_upload(file, units, axes, output_units)
        payload = result.to_dict()
        if include_preview:
            payload["preview"] = _preview(nodes, faces)
        if include_csv:
            try:
                payload["csv"] = result.to_csv(None if keys is None else keys.split(","))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        return payload

    @application.post(
        "/api/export", tags=["analysis"], summary="Analyse a mesh file (CSV download)",
        response_class=Response,
        responses={
            200: {"description": "CSV with columns parameter, value, unit. The first rows "
                                 "name the source file and its reader (`reader`, and "
                                 "`reader_version` when trimesh read it), then one `warning` "
                                 "row per warning, then the results.",
                  "content": {"text/csv": {"schema": {"type": "string"}, "example":
                      "parameter,value,unit\nsource_file,particle.stl,\nreader,builtin-stl,\n"
                      "volume,0.1666666666666667,mm^3\nconvexity,1.0,\n"}}},
            **UPLOAD_ERRORS,
        },
    )
    def export(
        file: UploadFile = File(..., description=FILE_DOC),
        units: str = Form("as-is", description=UNITS_DOC, **_choices(UNIT_SCALES)),
        keys: Optional[str] = Form(None, description=KEYS_DOC),
        axes: str = Form("pca", description=AXES_DOC, **_choices(AXES_LABELS)),
        output_units: str = Form("m", description=OUTPUT_DOC,
                                 **_choices(["same", *OUTPUT_UNITS])),
    ) -> Response:
        """Returns the selected results as a CSV file (`results.csv`), as written by
        `AnalysisResult.to_csv`. Units use plain ASCII (`mm^2`, `um`)."""
        result, _, _ = _analyze_upload(file, units, axes, output_units)
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
