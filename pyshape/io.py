from __future__ import annotations
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Tuple
import struct

import numpy as np

HAS_TRIMESH = find_spec("trimesh") is not None


def _get_trimesh():
    if not HAS_TRIMESH:
        return None
    import trimesh

    return trimesh


NATIVE_EXTENSIONS = (".stl",)
TRIMESH_EXTENSIONS = (".obj", ".ply", ".off", ".glb", ".gltf", ".3mf", ".dae")
SUPPORTED_EXTENSIONS = NATIVE_EXTENSIONS + TRIMESH_EXTENSIONS


@dataclass
class LoadedGeometry:
    """A triangle mesh and its source metadata; coordinates are not rescaled."""

    nodes: np.ndarray
    faces: np.ndarray
    source_path: Path
    source_format: str
    converted_from_native: bool


def load_geometry(path: str | Path) -> LoadedGeometry:
    """Load STL, or another supported format with the ``mesh`` extra.

    Some formats need additional trimesh format-specific dependencies.
    Loader errors propagate to the caller.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    if ext in NATIVE_EXTENSIONS:
        nodes, faces = load_stl(p)
    elif ext in TRIMESH_EXTENSIONS:
        trimesh = _get_trimesh()
        if trimesh is None:
            raise ImportError(f"Loading '{ext}' requires pyshape[mesh].")
        mesh = trimesh.load(str(p), process=True, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        mesh.merge_vertices()
        nodes = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return LoadedGeometry(nodes, faces, p, ext.lstrip("."), ext not in NATIVE_EXTENSIONS)


def _unique_rows_tol(points: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Unique rows with rounding tolerance; returns (unique_rows, inverse_idx)."""
    a = np.asarray(points, dtype=float)
    ar = np.round(a, decimals=decimals)
    # View as a structured array to use np.unique on rows
    view = np.ascontiguousarray(ar).view([("f%d" % i, ar.dtype) for i in range(ar.shape[1])])
    unique, idx, inv = np.unique(view, return_index=True, return_inverse=True)
    nodes = a[idx]
    return nodes, inv


def _parse_stl_binary(data: bytes) -> np.ndarray:
    """Return triangles array (T, 3, 3) from binary STL bytes."""
    if len(data) < 84:
        raise ValueError("Binary STL too small")
    n_tri = struct.unpack_from('<I', data, offset=80)[0]
    expect = 84 + 50 * n_tri
    if len(data) != expect:
        raise ValueError("Binary STL size does not match triangle count")
    # Define a structured dtype for one facet: normal(3f), vertices(3x3f), attr(u2)
    facet_dtype = np.dtype([
        ('normal', '<f4', (3,)),
        ('v', '<f4', (3, 3)),
        ('attr', '<u2'),
    ])
    recs = np.frombuffer(data, dtype=facet_dtype, count=n_tri, offset=84)
    triangles = recs['v'].astype(np.float64)
    return triangles


def _parse_stl_ascii(text: str) -> np.ndarray:
    """Return triangles array (T, 3, 3) from ASCII STL text."""
    tris = []
    cur = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith('vertex'):
            parts = line.split()
            # Format: vertex x y z
            if len(parts) != 4:
                raise ValueError("STL vertex must contain exactly three coordinates")
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            if not np.isfinite([x, y, z]).all():
                raise ValueError("STL coordinates must be finite")
            cur.append((x, y, z))
            if len(cur) == 3:
                tris.append(cur)
                cur = []
    if cur:
        raise ValueError("Incomplete triangle in ASCII STL")
    if not tris:
        raise ValueError("No triangles found in ASCII STL")
    return np.asarray(tris, dtype=np.float64)


def load_stl(path: str | Path, merge_vertices: bool = True, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an STL file and return (nodes, faces).

    Uses trimesh when available (recommended for compatibility), otherwise
    falls back to a built-in ASCII/binary STL parser.

    Parameters
    ----------
    path : str or Path
        Path to an STL file.
    merge_vertices : bool, default True
        If True, merge duplicate vertices.
    decimals : int, default 12
        Rounding precision used when merging vertices in the fallback parser.

    Returns
    -------
    nodes : (N, 3) float64
    faces : (M, 3) int32
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"STL not found: {p}")

    # Preferred: trimesh loader
    trimesh = _get_trimesh()
    if trimesh is not None:
        mesh = trimesh.load(str(p), process=True, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            # Combine scene geometries
            mesh = mesh.dump(concatenate=True)
        if merge_vertices:
            mesh.merge_vertices()
        nodes = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        return nodes, faces

    # Fallback: built-in ASCII/binary parser
    data = p.read_bytes()
    triangles = None
    if len(data) >= 84:
        try:
            triangles = _parse_stl_binary(data)
        except ValueError:
            triangles = None
    if triangles is None:
        try:
            text = data.decode('utf-8')
            triangles = _parse_stl_ascii(text)
        except (UnicodeDecodeError, ValueError) as e:
            raise ValueError(f"Could not parse STL file: {e}") from e

    verts = triangles.reshape(-1, 3)
    if merge_vertices:
        nodes, inv = _unique_rows_tol(verts, decimals=decimals)
        faces = inv.reshape(-1, 3).astype(np.int32)
    else:
        nodes = verts.astype(np.float64)
        faces = np.arange(nodes.shape[0], dtype=np.int32).reshape(-1, 3)
    return nodes, faces
