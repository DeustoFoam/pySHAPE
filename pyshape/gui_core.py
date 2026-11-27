"""Backward-compatible imports for the original desktop viewer helpers."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .analysis import (
    Metric,
    compute_metrics,
    convex_hull_volume,
    estimate_axes,
    mesh_volume_centroid,
)
from .export import make_export_path, write_csv
from .io import (
    LoadedGeometry,
    NATIVE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    TRIMESH_EXTENSIONS,
    load_geometry,
)


_UNIT_NAMES = {"um": "µm", "mm": "mm", "cm": "cm", "m": "m", "inch": "inch"}


def units_summary(units: str, scale: float, output_units: str = "m") -> str:
    """One-line description of how input coordinates map to result units.

    ``output_units`` is the effective output unit (see ``resolve_output_units``).
    Kept here, free of Tk/Matplotlib, so that it can be tested without the GUI extra.
    """
    if units == "as-is":
        return "Input: as-is → results in input units"
    result = _UNIT_NAMES.get(output_units, output_units)
    return f"Input: {_UNIT_NAMES.get(units, units)} → results in {result}"


_READER_NAMES = {
    "builtin-stl": "built-in STL parser",
    "trimesh": "trimesh",
    "trimesh-fallback": "trimesh (fallback: the built-in STL parser rejected the file)",
}


def reader_text(reader, version=None) -> str:
    """Human-readable name of the reader of a file, with the trimesh version."""
    text = _READER_NAMES.get(reader, reader or "unknown")
    if version:
        text = text.replace("trimesh", f"trimesh {version}", 1)
    return text


def geo_metadata_rows(geo: LoadedGeometry) -> list[tuple[str, str, str]]:
    """CSV metadata rows of a loaded file, as in ``AnalysisResult.to_csv``."""
    rows = []
    if geo.reader is not None:
        rows.append(("reader", geo.reader, ""))
    if geo.reader_version is not None:
        rows.append(("reader_version", geo.reader_version, ""))
    rows.extend(("warning", text, "") for text in geo.warnings)
    return rows


def axis_triad(nodes: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the center and arm length of a bounding-box-scaled XYZ triad."""
    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float((maxs - mins).max())
    return center, 0.6 * extent if extent > 0 else 1.0


# Preview colours (RGB in 0..1): the outer side of the surface is light blue,
# the inner side (a triangle seen from behind) a darker shade of the same blue,
# so that the inside of the surface showing through an opening or a
# transparent surface is not confused with the outside. Orange is reserved for
# nested shells (cavities). The browser preview uses the same colours.
OUTSIDE_RGB = (0.69, 0.77, 0.87)   # light steel blue
INSIDE_RGB = (0.36, 0.43, 0.53)    # darker shade of the same blue
CAVITY_RGB = (0.93, 0.62, 0.32)    # orange: cavity walls (nested shells)
CAVITY_INSIDE_RGB = (0.55, 0.36, 0.18)  # darker orange: back of a cavity wall
PREVIEW_LIGHT = (0.15, -1.1, 0.95)


def view_direction(elev: float, azim: float) -> np.ndarray:
    """Unit vector from the scene towards a Matplotlib 3D camera (degrees)."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def face_normals(nodes: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normals of the triangles (zero for degenerate ones), from their winding."""
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces)
    normals = np.cross(nodes[faces[:, 1]] - nodes[faces[:, 0]], nodes[faces[:, 2]] - nodes[faces[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    return np.divide(normals, lengths[:, None], out=np.zeros_like(normals),
                     where=lengths[:, None] > 0)


def preview_face_colors(
    normals: np.ndarray, view: np.ndarray, opacity: float = 1.0,
    cavity: Optional[np.ndarray] = None,
) -> np.ndarray:
    """RGBA face colours for the desktop preview.

    Triangles whose normal points towards the camera show the outer side
    (light blue); the others show the inner side (darker blue). Faces flagged
    in ``cavity`` (cavity walls) are orange, darker from behind. Both are shaded by a fixed
    light (Lambert, two-sided), and ``opacity`` (0 to 1) sets their alpha.
    """
    light = np.asarray(PREVIEW_LIGHT) / np.linalg.norm(PREVIEW_LIGHT)
    front = normals @ np.asarray(view, dtype=float) >= 0
    side = np.where(front, 1.0, -1.0)
    intensity = 0.3 + 0.7 * np.clip(side * (normals @ light), 0.0, 1.0)
    base = np.where(front[:, None], np.asarray(OUTSIDE_RGB)[None, :], np.asarray(INSIDE_RGB)[None, :])
    if cavity is not None and np.any(cavity):
        # Cavity walls (``cavity`` flags, oriented towards the cavity) are orange.
        cavity_base = np.where(front[:, None], np.asarray(CAVITY_RGB)[None, :],
                               np.asarray(CAVITY_INSIDE_RGB)[None, :])
        base = np.where(np.asarray(cavity, bool)[:, None], cavity_base, base)
    rgb = np.clip(base * intensity[:, None], 0.0, 1.0)
    alpha = np.full((len(normals), 1), float(np.clip(opacity, 0.0, 1.0)))
    return np.hstack([rgb, alpha])
