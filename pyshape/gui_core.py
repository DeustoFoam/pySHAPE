"""Non-UI logic backing the pySHAPE GUI (:mod:`pyshape.gui`).

Kept separate from the Tkinter widget code so it can be exercised by unit
tests without a display, and so the GUI module stays focused on layout and
event wiring.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .io import HAS_TRIMESH, load_stl, trimesh
from .geometry import surface_area as _surface_area
from .form import (
    sphericity_wadell,
    sphericity_krumbein,
    surface_orientation_tensor,
    form_parameters_kong_and_fonseca,
    form_parameters_potticary_et_al,
    form_parameters_zingg,
)

# File extensions that can be loaded directly (".stl") or, when trimesh is
# installed, transparently unified into a triangle mesh ("converted to STL").
NATIVE_EXTENSIONS = (".stl",)
TRIMESH_EXTENSIONS = (".obj", ".ply", ".off", ".glb", ".gltf", ".3mf", ".dae")
SUPPORTED_EXTENSIONS = NATIVE_EXTENSIONS + TRIMESH_EXTENSIONS


@dataclass
class Metric:
    """A single computed result parameter, selectable for CSV export."""

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
        if self.value is None:
            return "n/a"
        return f"{self.value:.6g}"


@dataclass
class LoadedGeometry:
    nodes: np.ndarray
    faces: np.ndarray
    source_path: Path
    source_format: str
    converted_from_native: bool


def load_geometry(path: "str | Path") -> LoadedGeometry:
    """Load a mesh file, unifying any supported format into a triangle mesh.

    ``.stl`` files are read with :func:`pyshape.load_stl`. Other supported
    formats (OBJ, PLY, OFF, glTF/GLB, 3MF, DAE) are loaded with ``trimesh``
    and converted into the same (nodes, faces) triangle-mesh representation
    that an STL would provide ("converted to STL" in spirit, since the
    resulting data is handled identically downstream).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()

    if ext in NATIVE_EXTENSIONS:
        nodes, faces = load_stl(p)
        return LoadedGeometry(nodes, faces, p, "stl", converted_from_native=False)

    if ext not in TRIMESH_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    if not HAS_TRIMESH:
        raise ValueError(
            f"Loading '{ext}' files requires the optional 'trimesh' package "
            "(pip install trimesh)."
        )
    mesh = trimesh.load(str(p), process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh.merge_vertices()
    nodes = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    return LoadedGeometry(nodes, faces, p, ext.lstrip("."), converted_from_native=True)


def mesh_volume_centroid(nodes: np.ndarray, faces: np.ndarray) -> Tuple[float, np.ndarray]:
    """Signed volume and centroid of a closed triangle mesh via the divergence theorem."""
    v0 = nodes[faces[:, 0]]
    v1 = nodes[faces[:, 1]]
    v2 = nodes[faces[:, 2]]
    cross = np.cross(v1, v2)
    tet_signed_vol = np.einsum("ij,ij->i", v0, cross) / 6.0
    volume = float(tet_signed_vol.sum())
    tet_centroid = (v0 + v1 + v2) / 4.0
    if abs(volume) > 0:
        centroid = (tet_signed_vol[:, None] * tet_centroid).sum(axis=0) / volume
    else:
        centroid = nodes.mean(axis=0)
    return volume, centroid


def estimate_axes(nodes: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """Estimate the Krumbein/Zingg short/intermediate/long axes (S, I, L) via PCA.

    Returns (S, I, L, directions) where ``directions`` is a (3, 3) array whose
    rows are the unit vectors for S, I, L in that order.
    """
    X = nodes - nodes.mean(axis=0)
    Vt = np.linalg.svd(X, full_matrices=False)[2]
    coords = X @ Vt.T
    spans = coords.max(axis=0) - coords.min(axis=0)
    order = np.argsort(spans)  # ascending: S, I, L
    S_len, I_len, L_len = (float(spans[i]) for i in order)
    directions = Vt[order]
    return S_len, I_len, L_len, directions


def convex_hull_volume(nodes: np.ndarray) -> float:
    from scipy.spatial import ConvexHull  # noqa: WPS433 (optional dependency)

    return float(ConvexHull(nodes).volume)


def compute_metrics(nodes: np.ndarray, faces: np.ndarray) -> List[Metric]:
    """Compute the full set of GUI result parameters for a triangle mesh."""
    metrics: List[Metric] = []

    def add(key, label, value, group, default_selected=True, error=None):
        metrics.append(Metric(key, label, value, group, default_selected, error))

    add("n_vertices", "Vertex count", float(nodes.shape[0]), "Mesh", default_selected=False)
    add("n_faces", "Face count", float(faces.shape[0]), "Mesh", default_selected=False)

    area = None
    try:
        area = _surface_area(nodes, faces)
        add("surface_area", "Surface area", area, "Geometry")
    except Exception as exc:  # pragma: no cover - defensive
        add("surface_area", "Surface area", None, "Geometry", error=str(exc))

    volume = None
    try:
        signed_volume, centroid = mesh_volume_centroid(nodes, faces)
        volume = abs(signed_volume)
        add("volume", "Volume (mesh)", volume, "Geometry")
        add("centroid_x", "Centroid X", float(centroid[0]), "Geometry", default_selected=False)
        add("centroid_y", "Centroid Y", float(centroid[1]), "Geometry", default_selected=False)
        add("centroid_z", "Centroid Z", float(centroid[2]), "Geometry", default_selected=False)
    except Exception as exc:  # pragma: no cover - defensive
        add("volume", "Volume (mesh)", None, "Geometry", error=str(exc))

    if area is not None and volume is not None:
        try:
            phi = sphericity_wadell(volume, area)
            add("sphericity_wadell", "Sphericity (Wadell)", phi, "Sphericity")
        except Exception as exc:
            add("sphericity_wadell", "Sphericity (Wadell)", None, "Sphericity", error=str(exc))

    try:
        C, F, R, eigvals, _ = surface_orientation_tensor(nodes, faces)
        add("orientation_C", "Orientation compactness (C)", C, "Orientation")
        add("orientation_F", "Orientation flakiness (F)", F, "Orientation")
        add("orientation_R", "Orientation rodness (R)", R, "Orientation")
        add("orientation_eig1", "Orientation eigenvalue 1", float(eigvals[0]), "Orientation", default_selected=False)
        add("orientation_eig2", "Orientation eigenvalue 2", float(eigvals[1]), "Orientation", default_selected=False)
        add("orientation_eig3", "Orientation eigenvalue 3", float(eigvals[2]), "Orientation", default_selected=False)
    except Exception as exc:
        add("orientation_C", "Orientation compactness (C)", None, "Orientation", error=str(exc))

    try:
        S_len, I_len, L_len, _ = estimate_axes(nodes)
        add("axis_S", "Short axis S (PCA)", S_len, "Axes (PCA estimate)", default_selected=False)
        add("axis_I", "Intermediate axis I (PCA)", I_len, "Axes (PCA estimate)", default_selected=False)
        add("axis_L", "Long axis L (PCA)", L_len, "Axes (PCA estimate)", default_selected=False)

        spK = sphericity_krumbein(S_len, I_len, L_len)
        add("sphericity_krumbein", "Sphericity (Krumbein)", spK, "Sphericity")

        flP, elP = form_parameters_potticary_et_al(S_len, I_len, L_len)
        add("potticary_flatness", "Flatness (Potticary et al.)", flP, "Form parameters")
        add("potticary_elongation", "Elongation (Potticary et al.)", elP, "Form parameters")

        flK, elK = form_parameters_kong_and_fonseca(S_len, I_len, L_len)
        add("kongfonseca_flatness", "Flatness (Kong & Fonseca)", flK, "Form parameters")
        add("kongfonseca_elongation", "Elongation (Kong & Fonseca)", elK, "Form parameters")

        SIZ, ILZ = form_parameters_zingg(S_len, I_len, L_len)
        add("zingg_SI", "Zingg ratio S/I", SIZ, "Form parameters")
        add("zingg_IL", "Zingg ratio I/L", ILZ, "Form parameters")
    except Exception as exc:
        add("axis_S", "Short axis S (PCA)", None, "Axes (PCA estimate)", error=str(exc))

    if volume is not None:
        try:
            Vch = convex_hull_volume(nodes)
            if Vch > 0:
                add("convexity", "Convexity (V / V_convex_hull)", volume / Vch, "Form parameters")
            else:
                add("convexity", "Convexity (V / V_convex_hull)", None, "Form parameters",
                    error="convex hull volume is zero")
        except ModuleNotFoundError:
            add("convexity", "Convexity (V / V_convex_hull)", None, "Form parameters",
                error="requires scipy")
        except Exception as exc:
            add("convexity", "Convexity (V / V_convex_hull)", None, "Form parameters", error=str(exc))

    return metrics


def axis_triad(nodes: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (center, arm_length) for an XYZ axes triad sized to the mesh bbox."""
    mins = nodes.min(axis=0)
    maxs = nodes.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float((maxs - mins).max())
    length = 0.6 * extent if extent > 0 else 1.0
    return center, length


def make_export_path(input_path: "str | Path", suffix: str, avoid_overwrite: bool) -> Path:
    """Compute the CSV output path for ``input_path``.

    By default the output sits next to the input file, named
    ``<stem><suffix>.csv`` (e.g. ``sample_SHAPE.csv``). When
    ``avoid_overwrite`` is True and that file already exists, a progressive
    ``_1``, ``_2``, ... counter is inserted before the extension instead of
    overwriting the existing file.
    """
    input_path = Path(input_path)
    base_name = f"{input_path.stem}{suffix}"
    candidate = input_path.parent / f"{base_name}.csv"
    if not avoid_overwrite or not candidate.exists():
        return candidate
    n = 1
    while True:
        candidate = input_path.parent / f"{base_name}_{n}.csv"
        if not candidate.exists():
            return candidate
        n += 1


def write_csv(path: "str | Path", rows: Iterable[Tuple[str, object]], source_name: Optional[str] = None) -> None:
    """Write (parameter, value) pairs to a CSV file."""
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        if source_name is not None:
            writer.writerow(["source_file", source_name])
        for name, value in rows:
            writer.writerow([name, value])
