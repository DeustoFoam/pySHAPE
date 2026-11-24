"""Headless triangle-mesh analysis used by every pySHAPE interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike

from .bounding import AXES_METHODS, oriented_box
from .export import csv_text
from .form import (
    form_parameters_kong_and_fonseca,
    form_parameters_potticary_et_al,
    form_parameters_angelidakis_et_al,
    form_parameters_zingg,
    sphericity_krumbein,
    sphericity_wadell,
    surface_orientation_tensor,
)
from .geometry import closed_surface_inertia_tensor, surface_area
from .io import load_geometry
from .preprocess import _volume_error, prepare_mesh, validate_mesh  # noqa: F401 (re-exported)

UNIT_SCALES = {"as-is": 1.0, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "inch": 0.0254}
AXES_LABELS = {"pca": ("PCA", "Axes (PCA estimate)"),
               "min_volume": ("min. volume box", "Axes (minimum-volume box)")}

# Power of length of each dimensional metric; all other metrics are dimensionless
# (counts, ratios, indices). Inertia is per unit density, hence length^5.
LENGTH_POWERS = {
    "surface_area": 2, "volume": 3,
    "centroid_x": 1, "centroid_y": 1, "centroid_z": 1,
    "inertia_I1": 5, "inertia_I2": 5, "inertia_I3": 5,
    "axis_S": 1, "axis_I": 1, "axis_L": 1,
}
_SUPERSCRIPTS = {1: "", 2: "²", 3: "³", 5: "⁵"}

# Length units for results: metres per unit. "same" means "same as the input units".
OUTPUT_UNITS = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "cm": 1e-2, "inch": 0.0254}
_UNIT_SYMBOLS = {"m": "m", "mm": "mm", "um": "µm", "cm": "cm", "inch": "in"}
_ASCII_SYMBOLS = {"m": "m", "mm": "mm", "um": "um", "cm": "cm", "inch": "in"}


def resolve_output_units(input_units: str, output_units: str) -> str:
    """Effective output unit: ``"as-is"`` for as-is input, else a key of OUTPUT_UNITS.

    ``output_units="same"`` follows the input units.
    """
    if output_units != "same" and output_units not in OUTPUT_UNITS:
        raise ValueError(
            f"Unknown output units '{output_units}'. Choose same or {', '.join(OUTPUT_UNITS)}"
        )
    if input_units == "as-is":
        return "as-is"
    if output_units != "same":
        return output_units
    return input_units


def unit_label(key: str, units: str, *, ascii: bool = False, output_units: str = "m") -> str:
    """Unit of a metric for the given input and output units ("" if dimensionless).

    Explicit input units give results in ``output_units`` (default m:
    m, m², m³, m⁵); ``as-is`` keeps the coordinates of the input file, so
    results are in ``input_units`` (no space, safe in CSV files) whatever
    ``output_units`` is.
    ``ascii=True`` writes powers as ``^2`` and µm as ``um`` (used for CSV files).
    """
    power = LENGTH_POWERS.get(key, 0)
    if power == 0:
        return ""
    if units == "as-is" or output_units == "as-is":
        base = "input_units"
    else:
        base = (_ASCII_SYMBOLS if ascii else _UNIT_SYMBOLS)[output_units]
    if power == 1:
        return base
    return f"{base}^{power}" if ascii else f"{base}{_SUPERSCRIPTS[power]}"


def apply_output_units(
    metrics: list["Metric"], input_units: str, output_units: str = "m",
) -> tuple[str, float]:
    """Rescale dimensional metrics (computed in metres) and set their units, in place.

    Returns the effective output unit and the length multiplier applied
    (1 for metres and for as-is input). A length scales with k, an area
    with k², a volume with k³ and an inertia per unit density with k⁵;
    dimensionless metrics are untouched.
    """
    effective = resolve_output_units(input_units, output_units)
    factor = 1.0 if effective == "as-is" else 1.0 / OUTPUT_UNITS[effective]
    for metric in metrics:
        power = LENGTH_POWERS.get(metric.key, 0)
        if power and metric.value is not None and factor != 1.0:
            metric.value = metric.value * factor ** power
        metric.unit = unit_label(metric.key, input_units, output_units=effective)
    return effective, factor


@dataclass
class Metric:
    """One descriptor; unavailable values carry an explicit error."""

    key: str
    label: str
    value: Optional[float]
    group: str = "General"
    default_selected: bool = True
    error: Optional[str] = None
    unit: str = ""

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
    axes_method: str = "pca"
    output_units: str = "m"
    output_scale: float = 1.0

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
        return csv_text(
            ((m.key, m.value,
              unit_label(m.key, self.input_units, ascii=True, output_units=self.output_units))
             for m in selected),
            self.source_name,
        )

    def write_csv(self, path: str | Path, keys: Optional[Iterable[str]] = None) -> None:
        """Write selected results as UTF-8 CSV, overwriting the destination."""
        text = self.to_csv(keys)
        with Path(path).open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)


def scale_nodes(nodes: np.ndarray, units: str) -> tuple[np.ndarray, float]:
    """Convert explicit input units to metres, or preserve coordinates as-is."""
    if units not in UNIT_SCALES:
        raise ValueError(f"Unknown units '{units}'. Choose one of: {', '.join(UNIT_SCALES)}")
    scale = UNIT_SCALES[units]
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


def compute_metrics(nodes: ArrayLike, faces: ArrayLike, *, axes: str = "pca") -> list[Metric]:
    """Calculate mesh descriptors with a stable set of keys.

    Closed shells are oriented and combined as a solid union. Open meshes retain
    surface and form descriptors but expose volume-dependent metrics as errors.
    Missing SciPy makes only convexity unavailable with the default PCA axes.
    Multiple closed shells need the ``mesh`` extra; overlap is counted once and
    internal surfaces excluded.

    ``axes`` selects how the short/intermediate/long dimensions S, I, L used by
    the axis-based form indices are measured: ``"pca"`` (principal directions
    of the vertices; default) or ``"min_volume"`` (minimum-volume oriented
    bounding box, which requires SciPy). See :mod:`pyshape.bounding`.
    """
    if axes not in AXES_METHODS:
        raise ValueError(f"Unknown axes method '{axes}'. Choose one of: {', '.join(AXES_METHODS)}")
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
    moments = (
        closed_surface_inertia_tensor(nodes, faces, centroid)[1]
        if volume is not None else [None, None, None]
    )
    for i, value in enumerate(moments, 1):
        add(f"inertia_I{i}", f"Principal inertia I{i} (unit density)", value,
            "Inertia (unit density)", False, volume_error)
    add("sphericity_wadell", "Sphericity (Wadell)",
        sphericity_wadell(volume, area) if volume is not None else None,
        "Sphericity", error=volume_error)

    compactness, flakiness, elongation, eigenvalues, _ = surface_orientation_tensor(nodes, faces)
    for key, label, value in (
        ("C", "Compactness", compactness), ("F", "Flakiness", flakiness),
        ("E", "Elongation", elongation),
    ):
        add(f"orientation_{key}", f"{label} {key} (surface orientation tensor)", value, "Orientation")
    for i, value in enumerate(eigenvalues, 1):
        add(f"orientation_eig{i}", f"Orientation eigenvalue {i}", value, "Orientation", False)

    try:
        box = oriented_box(nodes, axes)
    except ModuleNotFoundError as exc:
        if exc.name not in {"scipy", "scipy.spatial"}:
            raise
        raise ImportError(
            "axes='min_volume' requires scipy; install with: pip install 'pyshape[analysis]'"
        ) from exc
    short, intermediate, long = box.short, box.intermediate, box.long
    method_label, axes_group = AXES_LABELS[axes]
    for axis, label, value in (
        ("S", "Short", short), ("I", "Intermediate", intermediate), ("L", "Long", long),
    ):
        add(f"axis_{axis}", f"{label} axis {axis} ({method_label})", value, axes_group, False)
    add("sphericity_krumbein", "Sphericity (Krumbein)",
        sphericity_krumbein(short, intermediate, long), "Sphericity")
    for prefix, label, function in (
        ("potticary", "Potticary et al.", form_parameters_potticary_et_al),
        ("kongfonseca", "Kong & Fonseca", form_parameters_kong_and_fonseca),
        ("angelidakis", "Angelidakis et al.", form_parameters_angelidakis_et_al),
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
    nodes: ArrayLike, faces: ArrayLike, *, units: str = "as-is", axes: str = "pca",
    output_units: str = "m",
) -> AnalysisResult:
    """Analyze arrays; explicit physical units convert lengths to metres.

    ``axes`` is ``"pca"`` (default) or ``"min_volume"``; see :func:`compute_metrics`.
    ``output_units`` (``"m"`` by default, or ``"mm"``, ``"um"``, ``"cm"``,
    ``"inch"``, ``"same"``) sets the length unit of the dimensional results;
    it is ignored for ``units="as-is"``, whose results stay in input units.
    """
    resolve_output_units(units if units == "as-is" else "m", output_units)  # validate early
    vertices, indices = validate_mesh(nodes, faces)
    vertices, scale = scale_nodes(vertices, units)
    metrics = compute_metrics(vertices, indices, axes=axes)
    effective, factor = apply_output_units(metrics, units, output_units)
    return AnalysisResult(
        metrics, input_units=units, scale_factor=scale, axes_method=axes,
        output_units=effective, output_scale=factor,
    )


def analyze_file(
    path: str | Path, *, units: str = "as-is", axes: str = "pca", output_units: str = "m",
) -> AnalysisResult:
    """Load and analyze a supported mesh file; no GUI or server is needed."""
    geometry = load_geometry(path)
    result = analyze_mesh(
        geometry.nodes, geometry.faces, units=units, axes=axes, output_units=output_units,
    )
    result.source_name = geometry.source_path.name
    return result
