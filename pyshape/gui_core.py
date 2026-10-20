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


def axis_triad(nodes: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the center and arm length of a bounding-box-scaled XYZ triad."""
    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float((maxs - mins).max())
    return center, 0.6 * extent if extent > 0 else 1.0
