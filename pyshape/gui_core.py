"""Backward-compatible imports for the original desktop viewer helpers."""
from __future__ import annotations

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
