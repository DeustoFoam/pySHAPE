"""Headless triangle-mesh analysis used by every pySHAPE interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike

from .export import csv_text
from .form import (
    form_parameters_kong_and_fonseca,
    form_parameters_potticary_et_al,
    form_parameters_zingg,
    sphericity_krumbein,
    sphericity_wadell,
    surface_orientation_tensor,
)
from .geometry import surface_area
from .io import load_geometry

UNIT_SCALES = {"as-is": 1.0, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "inch": 0.0254}


@dataclass
class Metric:
    """One descriptor; unavailable values carry an explicit error."""

    key: str
    label: str
    value: Optional[float]
    group: str = "General"
    default_selected: bool = True
    error: Optional[str] = None

    @property
    def text(self) -> str:
        if self.error is not None:
            return f"n/a ({self.error})"
        return "n/a" if self.value is None else f"{self.value:.6g}"


@dataclass
class AnalysisResult:
    """Analysis metadata and descriptors, without UI or HTTP dependencies."""

    metrics: list[Metric]
    source_name: Optional[str] = None
    input_units: str = "as-is"
    scale_factor: float = 1.0

    @property
    def values(self) -> dict[str, Optional[float]]:
        """Metric keys mapped to values; inspect ``metrics`` for error details."""
        return {metric.key: metric.value for metric in self.metrics}

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible metadata and full metric records."""
        return asdict(self)

    def to_csv(self, keys: Optional[Iterable[str]] = None) -> str:
        """Export available results using stable metric keys.

        With no selection, export the default-selected available metrics.
        Explicit unknown or unavailable keys raise ValueError.
        """
        by_key = {metric.key: metric for metric in self.metrics}
        if keys is None:
            selected = [
                metric for metric in self.metrics
                if metric.default_selected and metric.value is not None
            ]
        else:
            selected = []
            for key in keys:
                if key not in by_key:
                    raise ValueError(f"Unknown metric: {key}")
                metric = by_key[key]
                if metric.value is None:
                    raise ValueError(f"Metric '{key}' is unavailable: {metric.error}")
                selected.append(metric)
        if not selected:
            raise ValueError("Select at least one available metric")
        return csv_text(((m.key, m.value) for m in selected), self.source_name)

    def write_csv(self, path: str | Path, keys: Optional[Iterable[str]] = None) -> None:
        """Write selected results as UTF-8 CSV, overwriting the destination."""
        text = self.to_csv(keys)
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)


def validate_mesh(nodes: ArrayLike, faces: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a nonempty, finite triangular mesh without mutating inputs.

    MATLAB-style one-based faces are recognized only when their minimum is 1
    and maximum equals the vertex count, matching the existing low-level API.
    """
    vertices = np.asarray(nodes, dtype=float)
    indices = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("nodes must have shape (N, 3) with at least three vertices")
    if not np.isfinite(vertices).all():
        raise ValueError("nodes must contain only finite coordinates")
    if indices.ndim != 2 or indices.shape[1] != 3 or len(indices) == 0:
        raise ValueError("faces must have nonempty shape (M, 3)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if indices.min() == 1 and indices.max() == len(vertices):
        indices = indices - 1
    if indices.min() < 0 or indices.max() >= len(vertices):
        raise ValueError("face indices out of range for nodes array")
    indices = indices.astype(np.intp, copy=False)
    area = surface_area(vertices, indices)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("mesh must have finite, positive surface area")
    return vertices, indices


def scale_nodes(nodes: np.ndarray, units: str) -> tuple[np.ndarray, float]:
    """Convert explicit input units to metres, or preserve coordinates as-is.

    ``auto`` retains the desktop viewer's legacy size heuristic; it is opt-in.
    """
    if units == "auto":
        span = float(np.ptp(nodes, axis=0).max())
        scale = 1e-6 if span > 10000 else 1e-3 if span > 10 else 1.0
    elif units in UNIT_SCALES:
        scale = UNIT_SCALES[units]
    else:
        raise ValueError(f"Unknown units '{units}'. Choose auto or {', '.join(UNIT_SCALES)}")
    return nodes * scale, scale


def mesh_volume_centroid(nodes: np.ndarray, faces: np.ndarray) -> tuple[float, np.ndarray]:
    """Signed volume and centroid; requires a closed, consistently wound mesh."""
    origin = nodes.mean(axis=0)
    v0, v1, v2 = (nodes[faces[:, i]] - origin for i in range(3))
    tet_signed_vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)) / 6.0
    volume = float(tet_signed_vol.sum())
    if not np.isfinite(volume) or volume == 0:
        raise ValueError("mesh has zero or non-finite enclosed volume")
    centroid = origin + (tet_signed_vol[:, None] * (v0 + v1 + v2) / 4.0).sum(axis=0) / volume
    if not np.isfinite(centroid).all():
        raise ValueError("mesh has a non-finite centroid")
    return volume, centroid


def estimate_axes(nodes: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """PCA-estimated short/intermediate/long spans and corresponding directions."""
    centered = nodes - nodes.mean(axis=0)
    directions = np.linalg.svd(centered, full_matrices=False)[2]
    spans = np.ptp(centered @ directions.T, axis=0)
    order = np.argsort(spans)
    short, intermediate, long = (float(spans[i]) for i in order)
    return short, intermediate, long, directions[order]


def convex_hull_volume(nodes: np.ndarray) -> float:
    """Compute convex-hull volume with optional SciPy."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(nodes).volume)
    except QhullError as exc:
        raise ValueError("Cannot construct a three-dimensional convex hull") from exc


def _volume_error(faces: np.ndarray) -> Optional[str]:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2):
        return "volume requires a closed, manifold triangle mesh"
    orientation = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(np.bincount(inverse, weights=orientation) != 0):
        return "volume requires consistently wound faces"
    return None


def prepare_mesh(nodes: ArrayLike, faces: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Orient closed shells outward and combine them as a solid union.

    Open/non-manifold meshes are not filled or repaired. Multiple closed shells
    represent solid components, not cavities, and need the ``mesh`` extra.
    Input arrays are never modified.
    """
    nodes, faces = validate_mesh(nodes, faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2):
        return nodes, faces

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in faces]
    for first, second in np.argsort(inverse).reshape(-1, 2):
        i, j = int(first % len(faces)), int(second % len(faces))
        same_direction = bool(np.array_equal(edges[first], edges[second]))
        adjacency[i].append((j, same_direction))
        adjacency[j].append((i, same_direction))

    flips = np.full(len(faces), -1, dtype=np.int8)
    components: list[list[int]] = []
    for root in range(len(faces)):
        if flips[root] != -1:
            continue
        flips[root] = 0
        stack = [root]
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, same_direction in adjacency[current]:
                flip = int(flips[current]) ^ int(same_direction)
                if flips[neighbor] == -1:
                    flips[neighbor] = flip
                    stack.append(neighbor)
                elif flips[neighbor] != flip:
                    raise ValueError("mesh surface is not orientable")
        components.append(component)

    oriented = faces.copy()
    mask = flips == 1
    oriented[mask] = oriented[mask, ::-1]
    for component in components:
        shell = oriented[component]
        vertices = nodes[shell]
        vertices = vertices - vertices.mean(axis=(0, 1))
        signed_volume = np.einsum(
            "ij,ij->i", vertices[:, 0], np.cross(vertices[:, 1], vertices[:, 2])
        ).sum() / 6.0
        if signed_volume < 0:
            oriented[component] = shell[:, ::-1]
    if len(components) == 1:
        return nodes, oriented
    return _union_shells(nodes, oriented, components)


def _union_shells(
    nodes: np.ndarray, faces: np.ndarray, components: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import_module("manifold3d")
        import trimesh
    except ModuleNotFoundError as exc:
        if exc.name not in {"manifold3d", "trimesh"}:
            raise
        raise ImportError(
            "Multi-component solid union requires pyshape[mesh]; "
            "install with: pip install 'pyshape[mesh]'"
        ) from exc

    # Manifold uses float32 coordinates. Normalize before boolean operations so
    # micron-scale meshes retain the same precision as metre-scale meshes.
    origin = nodes.min(axis=0) + np.ptp(nodes, axis=0) / 2
    scale = float(np.ptp(nodes, axis=0).max())
    normalized = (nodes - origin) / scale
    shells = [
        trimesh.Trimesh(vertices=normalized, faces=faces[component], process=False)
        for component in components
    ]
    union = trimesh.boolean.union(shells, engine="manifold")
    if union.is_empty:
        raise ValueError("Solid union produced an empty mesh")
    vertices, indices = validate_mesh(
        np.asarray(union.vertices) * scale + origin, np.asarray(union.faces)
    )
    error = _volume_error(indices)
    if error is not None:
        raise ValueError(f"Solid union failed: {error}")
    return vertices, indices


def compute_metrics(nodes: ArrayLike, faces: ArrayLike) -> list[Metric]:
    """Calculate mesh descriptors with a stable set of keys.

    Closed shells are oriented and combined as a solid union. Open meshes retain
    surface and form descriptors but expose volume-dependent metrics as errors.
    Missing SciPy makes only convexity unavailable. Multiple closed shells need
    the ``mesh`` extra; overlap is counted once and internal surfaces excluded.
    """
    nodes, faces = prepare_mesh(nodes, faces)
    metrics: list[Metric] = []

    def add(
        key: str, label: str, value: Optional[float], group: str,
        default_selected: bool = True, error: Optional[str] = None,
    ) -> None:
        if value is not None:
            value = float(value)
            if not np.isfinite(value):
                value, error = None, "calculation produced a non-finite value"
        metrics.append(Metric(key, label, value, group, default_selected, error))

    add("n_vertices", "Vertex count", len(nodes), "Mesh", False)
    add("n_faces", "Face count", len(faces), "Mesh", False)
    area = surface_area(nodes, faces)
    add("surface_area", "Surface area", area, "Geometry")
    volume_error = _volume_error(faces)
    volume = None
    centroid = None
    if volume_error is None:
        try:
            signed_volume, centroid = mesh_volume_centroid(nodes, faces)
            volume = abs(signed_volume)
        except ValueError as exc:
            volume_error = str(exc)
    add("volume", "Volume (mesh)", volume, "Geometry", error=volume_error)
    for i, axis in enumerate("xyz"):
        add(f"centroid_{axis}", f"Centroid {axis.upper()}",
            float(centroid[i]) if centroid is not None else None,
            "Geometry", False, volume_error)
    add("sphericity_wadell", "Sphericity (Wadell)",
        sphericity_wadell(volume, area) if volume is not None else None,
        "Sphericity", error=volume_error)

    compactness, flakiness, rodness, eigenvalues, _ = surface_orientation_tensor(nodes, faces)
    for key, label, value in (
        ("C", "compactness", compactness), ("F", "flakiness", flakiness),
        ("R", "rodness", rodness),
    ):
        add(f"orientation_{key}", f"Orientation {label} ({key})", value, "Orientation")
    for i, value in enumerate(eigenvalues, 1):
        add(f"orientation_eig{i}", f"Orientation eigenvalue {i}", value, "Orientation", False)

    short, intermediate, long, _ = estimate_axes(nodes)
    for axis, label, value in (
        ("S", "Short", short), ("I", "Intermediate", intermediate), ("L", "Long", long),
    ):
        add(f"axis_{axis}", f"{label} axis {axis} (PCA)", value, "Axes (PCA estimate)", False)
    add("sphericity_krumbein", "Sphericity (Krumbein)",
        sphericity_krumbein(short, intermediate, long), "Sphericity")
    for prefix, label, function in (
        ("potticary", "Potticary et al.", form_parameters_potticary_et_al),
        ("kongfonseca", "Kong & Fonseca", form_parameters_kong_and_fonseca),
    ):
        flatness, elongation = function(short, intermediate, long)
        add(f"{prefix}_flatness", f"Flatness ({label})", flatness, "Form parameters")
        add(f"{prefix}_elongation", f"Elongation ({label})", elongation, "Form parameters")
    si, il = form_parameters_zingg(short, intermediate, long)
    add("zingg_SI", "Zingg ratio S/I", si, "Form parameters")
    add("zingg_IL", "Zingg ratio I/L", il, "Form parameters")

    convexity = None
    hull_error = volume_error
    if volume is not None:
        try:
            hull_volume = convex_hull_volume(nodes)
            if hull_volume <= 0:
                raise ValueError("convex hull volume is zero")
            convexity = volume / hull_volume
        except ModuleNotFoundError as exc:
            if exc.name not in {"scipy", "scipy.spatial"}:
                raise
            hull_error = "requires scipy; install pyshape[analysis]"
        except ValueError as exc:
            hull_error = str(exc)
    add("convexity", "Convexity (V / V_convex_hull)", convexity,
        "Form parameters", error=hull_error)
    return metrics


def analyze_mesh(
    nodes: ArrayLike, faces: ArrayLike, *, units: str = "as-is",
) -> AnalysisResult:
    """Analyze arrays; explicit physical units convert lengths to metres."""
    vertices, indices = validate_mesh(nodes, faces)
    vertices, scale = scale_nodes(vertices, units)
    return AnalysisResult(compute_metrics(vertices, indices), input_units=units, scale_factor=scale)


def analyze_file(path: str | Path, *, units: str = "as-is") -> AnalysisResult:
    """Load and analyze a supported mesh file; no GUI or server is needed."""
    geometry = load_geometry(path)
    result = analyze_mesh(geometry.nodes, geometry.faces, units=units)
    result.source_name = geometry.source_path.name
    return result
