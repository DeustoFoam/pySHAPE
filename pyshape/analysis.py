"""Headless triangle-mesh analysis used by every pySHAPE interface."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike

from .bounding import AXES_METHODS, oriented_box
from .ellipsoid import fitted_ellipsoid, inertia_ellipsoid_from_moments
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
from .io import LoadedGeometry, load_geometry
from .preprocess import (  # noqa: F401 (prepare_mesh, validate_mesh re-exported)
    CAVITY_MODES, _volume_error, prepare_mesh, prepare_solid, validate_mesh,
)

UNIT_SCALES = {"as-is": 1.0, "um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "inch": 0.0254}
# Result groups, ordered by what each descriptor depends on (issue #7).
GROUP_MESH = "Mesh"
GROUP_SIZE = "Size and mass properties"
GROUP_SURFACE_FORM = "Form, surface-based (independent of the box)"
# Per axes method: short label used in metric labels, name of the source of
# S, I, L, and the group of S, I, L.
AXES_SOURCES = {"pca": "PCA box", "min_volume": "minimum-volume box",
                "inertia_ellipsoid": "inertia-equivalent ellipsoid",
                "fitted_ellipsoid": "fitted ellipsoid"}
AXES_LABELS = {method: (short, f"Principal dimensions ({AXES_SOURCES[method]})")
               for method, short in (("pca", "PCA"), ("min_volume", "min. volume box"),
                                     ("inertia_ellipsoid", "inertia ellipsoid"),
                                     ("fitted_ellipsoid", "fitted ellipsoid"))}


def result_groups(axes: str = "pca") -> list[str]:
    """Result groups in display order for an axes method."""
    return [GROUP_MESH, GROUP_SIZE, GROUP_SURFACE_FORM, AXES_LABELS[axes][1],
            f"Form from S, I, L ({AXES_SOURCES[axes]})"]

# Power of length of each dimensional metric; all other metrics are dimensionless
# (counts, ratios, indices). Inertia is per unit density, hence length^5.
LENGTH_POWERS = {
    "surface_area": 2, "surface_area_outer": 2, "volume": 3, "cavity_volume": 3,
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
    """Analysis metadata and descriptors, without UI or HTTP dependencies.

    For a file, ``reader`` names the reader that loaded it (``"builtin-stl"``,
    ``"trimesh"`` or ``"trimesh-fallback"``, see :func:`pyshape.load_geometry`),
    ``reader_version`` the trimesh version when trimesh was used, and
    ``warnings`` lists notes about the input, e.g. an STL file read by the
    trimesh fallback. All three are empty for arrays. ``cavities`` records how
    nested closed shells were treated (``"fill"`` or ``"keep"``).
    """

    metrics: list[Metric]
    source_name: Optional[str] = None
    input_units: str = "as-is"
    scale_factor: float = 1.0
    axes_method: str = "pca"
    output_units: str = "m"
    output_scale: float = 1.0
    reader: Optional[str] = None
    reader_version: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    cavities: str = "fill"

    def metadata_rows(self) -> list[tuple[str, str, str]]:
        """CSV rows after ``source_file``: reader, its version and each warning."""
        rows = []
        if self.reader is not None:
            rows.append(("reader", self.reader, ""))
        if self.reader_version is not None:
            rows.append(("reader_version", self.reader_version, ""))
        rows.extend(("warning", text, "") for text in self.warnings)
        return rows

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
        Explicit unknown or unavailable keys raise ValueError. For a file, the
        rows ``source_file``, ``reader``, ``reader_version`` (when trimesh was
        used) and ``warning`` (one per warning) come before the results.
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
            self.metadata_rows(),
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


def compute_metrics(
    nodes: ArrayLike, faces: ArrayLike, *, axes: str = "pca", cavities: str = "fill",
) -> list[Metric]:
    """Calculate mesh descriptors with a stable set of keys.

    The mesh is first prepared by :func:`pyshape.prepare_solid`: closed shells
    are oriented and, with the default ``cavities="fill"``, combined as a solid
    union (overlap counted once, internal surfaces excluded; several closed
    shells to merge need the ``mesh`` extra). Open meshes retain surface and
    form descriptors but expose volume-dependent metrics as errors. Missing
    SciPy makes only convexity unavailable with the default PCA axes. The 34
    metrics are grouped by what they depend on (:func:`result_groups`).

    ``axes`` selects how the short/intermediate/long dimensions S, I, L used by
    the axis-based form indices are measured: ``"pca"`` (principal directions
    of the vertices; default), ``"min_volume"`` (minimum-volume oriented
    bounding box, which requires SciPy; see :mod:`pyshape.bounding`),
    ``"inertia_ellipsoid"`` (full axes of the ellipsoid with the same volume
    and principal moments; needs a closed surface) or ``"fitted_ellipsoid"``
    (area-weighted least-squares ellipsoid through the surface vertices); see
    :mod:`pyshape.ellipsoid`. Ellipsoid axes are not enclosing dimensions. If
    they cannot be computed, S, I, L and the indices derived from them carry
    an error.

    ``cavities`` sets how nested closed shells are treated (see
    :func:`pyshape.prepare_solid`): ``"fill"`` (default) treats every shell as
    solid; ``"keep"`` applies the even-odd rule, so that a shell inside another
    is a cavity. With cavities kept, the mass properties (volume, centroid,
    inertia) and ``surface_area`` refer to the material (cavity walls
    included), while the form descriptors (Wadell sphericity, convexity,
    surface orientation tensor, principal dimensions and the indices from
    them) refer to the outer envelope, so that internal voids do not change
    the particle's form. ``n_cavities``, ``cavity_volume`` and ``porosity``
    (cavity volume / envelope volume) are reported only with ``"keep"``.
    """
    if axes not in AXES_METHODS:
        raise ValueError(f"Unknown axes method '{axes}'. Choose one of: {', '.join(AXES_METHODS)}")
    solid = prepare_solid(nodes, faces, cavities=cavities)
    nodes, faces = solid.nodes, solid.faces
    separate_envelope = solid.n_cavities is not None and solid.n_cavities > 0
    env_nodes, env_faces = (solid.envelope_nodes, solid.envelope_faces) if separate_envelope \
        else (nodes, faces)
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

    add("n_vertices", "Vertex count", len(nodes), GROUP_MESH, False)
    add("n_faces", "Face count", len(faces), GROUP_MESH, False)
    area = surface_area(nodes, faces)
    add("surface_area", "Surface area", area, GROUP_SIZE)
    outer_area = surface_area(env_nodes, env_faces) if separate_envelope else area
    add("surface_area_outer", "Surface area of the envelope (outer surface)", outer_area,
        GROUP_SIZE, False)
    volume_error = _volume_error(faces)
    volume = None
    centroid = None
    if volume_error is None:
        try:
            signed_volume, centroid = mesh_volume_centroid(nodes, faces)
            volume = abs(signed_volume)
        except ValueError as exc:
            volume_error = str(exc)
    add("volume", "Volume (mesh)", volume, GROUP_SIZE, error=volume_error)
    for i, axis in enumerate("xyz"):
        add(f"centroid_{axis}", f"Centroid {axis.upper()}",
            float(centroid[i]) if centroid is not None else None,
            GROUP_SIZE, False, volume_error)
    moments, principal_axes = (
        closed_surface_inertia_tensor(nodes, faces, centroid)[1:]
        if volume is not None else ([None, None, None], None)
    )
    for i, value in enumerate(moments, 1):
        add(f"inertia_I{i}", f"Principal inertia I{i} (unit density)", value,
            GROUP_SIZE, False, volume_error)

    # Envelope (outer surface, cavities filled): the form descriptors use it.
    env_volume, env_centroid, env_moments, env_axes = volume, centroid, moments, principal_axes
    envelope_error = volume_error
    if separate_envelope and volume is not None:
        try:
            env_volume, env_centroid = mesh_volume_centroid(env_nodes, env_faces)
            env_volume = abs(env_volume)
            env_moments, env_axes = closed_surface_inertia_tensor(
                env_nodes, env_faces, env_centroid)[1:]
        except ValueError as exc:  # reported on the envelope-based results, not raised
            env_volume, envelope_error = None, f"envelope: {exc}"
    filled = "cavities are filled (cavities='fill')"
    if solid.n_cavities is None:
        cavity_rows = [(None, filled)] * 3
    elif volume is None or env_volume is None:
        cavity_rows = [(solid.n_cavities, None), (None, envelope_error), (None, envelope_error)]
    else:
        void = env_volume - volume
        cavity_rows = [(solid.n_cavities, None), (void, None), (void / env_volume, None)]
    for (key, label), (value, error) in zip((
        ("n_cavities", "Number of cavities"),
        ("cavity_volume", "Cavity volume"),
        ("porosity", "Porosity (cavity volume / envelope volume)"),
    ), cavity_rows):
        add(key, label, value, GROUP_SIZE, False, error)

    add("sphericity_wadell", "Sphericity (Wadell)",
        sphericity_wadell(env_volume, outer_area) if env_volume is not None else None,
        GROUP_SURFACE_FORM, error=envelope_error)
    convexity = None
    hull_error = envelope_error
    if env_volume is not None:
        try:
            hull_volume = convex_hull_volume(env_nodes)
            if hull_volume <= 0:
                raise ValueError("convex hull volume is zero")
            convexity = env_volume / hull_volume
        except ModuleNotFoundError as exc:
            if exc.name not in {"scipy", "scipy.spatial"}:
                raise
            hull_error = "requires scipy; install pyshape[analysis]"
        except ValueError as exc:
            hull_error = str(exc)
    add("convexity", "Convexity (V / V_convex_hull)", convexity,
        GROUP_SURFACE_FORM, error=hull_error)
    compactness, flakiness, elongation, eigenvalues, _ = surface_orientation_tensor(env_nodes, env_faces)
    for key, label, value in (
        ("C", "Compactness", compactness), ("F", "Flakiness", flakiness),
        ("E", "Elongation", elongation),
    ):
        add(f"orientation_{key}", f"{label} {key} (surface orientation tensor)", value,
            GROUP_SURFACE_FORM)
    for i, value in enumerate(eigenvalues, 1):
        add(f"orientation_eig{i}", f"Eigenvalue f{i} (surface orientation tensor)", value,
            GROUP_SURFACE_FORM, False)

    dims_error = None
    if axes == "inertia_ellipsoid":
        if env_volume is None:
            dims_error = f"inertia-equivalent ellipsoid unavailable: {envelope_error}"
        else:
            try:
                dims = inertia_ellipsoid_from_moments(env_volume, env_moments, env_axes, env_centroid)
            except ValueError as exc:
                dims_error = f"inertia-equivalent ellipsoid unavailable: {exc}"
    elif axes == "fitted_ellipsoid":
        try:
            dims = fitted_ellipsoid(env_nodes, env_faces)
        except ValueError as exc:
            dims_error = f"ellipsoid fit failed: {exc}"
    else:
        try:
            dims = oriented_box(env_nodes, axes)
        except ModuleNotFoundError as exc:
            if exc.name not in {"scipy", "scipy.spatial"}:
                raise
            raise ImportError(
                "axes='min_volume' requires scipy; install with: pip install 'pyshape[analysis]'"
            ) from exc
    if dims_error is None:
        short, intermediate, long = dims.short, dims.intermediate, dims.long
    else:
        short = intermediate = long = None
    method_label, axes_group = AXES_LABELS[axes]
    for axis, label, value in (
        ("S", "Short", short), ("I", "Intermediate", intermediate), ("L", "Long", long),
    ):
        add(f"axis_{axis}", f"{label} axis {axis} ({method_label})", value, axes_group, False,
            dims_error)
    dims_group = f"Form from S, I, L ({AXES_SOURCES[axes]})"

    def from_dims(function):
        return (None, None) if dims_error is not None else function(short, intermediate, long)

    add("sphericity_krumbein", "Sphericity (Krumbein)",
        None if dims_error else sphericity_krumbein(short, intermediate, long), dims_group,
        error=dims_error)
    for prefix, label, function in (
        ("potticary", "Potticary et al.", form_parameters_potticary_et_al),
        ("kongfonseca", "Kong & Fonseca", form_parameters_kong_and_fonseca),
        ("angelidakis", "Angelidakis et al.", form_parameters_angelidakis_et_al),
    ):
        flatness, elongation = from_dims(function)
        add(f"{prefix}_flatness", f"Flatness ({label})", flatness, dims_group, error=dims_error)
        add(f"{prefix}_elongation", f"Elongation ({label})", elongation, dims_group,
            error=dims_error)
    si, il = from_dims(form_parameters_zingg)
    add("zingg_SI", "Zingg ratio S/I", si, dims_group, error=dims_error)
    add("zingg_IL", "Zingg ratio I/L", il, dims_group, error=dims_error)
    return metrics


def analyze_mesh(
    nodes: ArrayLike, faces: ArrayLike, *, units: str = "as-is", axes: str = "pca",
    output_units: str = "m", cavities: str = "fill",
) -> AnalysisResult:
    """Analyze arrays; explicit physical units convert lengths to metres.

    ``axes`` selects the principal dimensions and ``cavities`` (``"fill"``,
    default, or ``"keep"``) how nested shells are treated; see
    :func:`compute_metrics`.
    ``output_units`` (``"m"`` by default, or ``"mm"``, ``"um"``, ``"cm"``,
    ``"inch"``, ``"same"``) sets the length unit of the dimensional results;
    it is ignored for ``units="as-is"``, whose results stay in input units.
    """
    resolve_output_units(units if units == "as-is" else "m", output_units)  # validate early
    vertices, indices = validate_mesh(nodes, faces)
    vertices, scale = scale_nodes(vertices, units)
    metrics = compute_metrics(vertices, indices, axes=axes, cavities=cavities)
    effective, factor = apply_output_units(metrics, units, output_units)
    return AnalysisResult(
        metrics, input_units=units, scale_factor=scale, axes_method=axes,
        output_units=effective, output_scale=factor, cavities=cavities,
    )


def analyze_file(
    path: str | Path, *, units: str = "as-is", axes: str = "pca", output_units: str = "m",
    strict_stl: bool = False, cavities: str = "fill",
) -> AnalysisResult:
    """Load and analyze a supported mesh file; no GUI or server is needed.

    The reader is chosen by the extension (see :func:`pyshape.load_geometry`):
    STL files are read by the built-in parser, with trimesh as a fallback when
    the parser rejects a file, unless ``strict_stl=True``. The result records
    the reader, and a fallback adds a warning. Every mesh is then prepared in
    the same way (:func:`pyshape.prepare_mesh`).
    """
    geometry = load_geometry(path, strict_stl=strict_stl)
    result = analyze_mesh(
        geometry.nodes, geometry.faces, units=units, axes=axes, output_units=output_units,
        cavities=cavities,
    )
    return attach_source(result, geometry)


def attach_source(
    result: AnalysisResult, geometry: LoadedGeometry, name: Optional[str] = None,
) -> AnalysisResult:
    """Copy the file name (or ``name``), the reader and the load warnings onto ``result``."""
    result.source_name = geometry.source_path.name if name is None else name
    result.reader = geometry.reader
    result.reader_version = geometry.reader_version
    result.warnings = [*geometry.warnings, *result.warnings]
    return result
