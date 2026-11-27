"""Mesh file loading.

STL files are always read by the built-in ASCII/binary parser of this module.
trimesh (``pyshape[mesh]``) reads the other formats and, when the built-in
parser rejects an STL file, is tried as a fallback; the result then records the
reader and carries a warning with the parser's reason.
"""
from __future__ import annotations

import re
import struct
import warnings
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Tuple

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

#: Values of ``LoadedGeometry.reader`` and ``AnalysisResult.reader``.
READER_BUILTIN_STL = "builtin-stl"
READER_TRIMESH = "trimesh"
READER_TRIMESH_FALLBACK = "trimesh-fallback"


class STLParseError(ValueError):
    """The built-in parser cannot read an STL file; the message gives the reason."""


class STLFallbackWarning(UserWarning):
    """An STL file rejected by the built-in parser was read by trimesh."""


class _NonFiniteSTL(ValueError):
    """A NaN or infinite coordinate: an error, never retried with trimesh."""


@dataclass
class LoadedGeometry:
    """A triangle mesh and its source metadata; coordinates are not rescaled.

    ``reader`` is ``"builtin-stl"``, ``"trimesh"`` (formats other than STL) or
    ``"trimesh-fallback"`` (an STL file rejected by the built-in parser);
    ``reader_version`` is the trimesh version when trimesh was used, and
    ``warnings`` explains a fallback.
    """

    nodes: np.ndarray
    faces: np.ndarray
    source_path: Path
    source_format: str
    converted_from_native: bool
    reader: Optional[str] = None
    reader_version: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


def load_geometry(path: str | Path, *, strict_stl: bool = False) -> LoadedGeometry:
    """Load a supported mesh file; the reader is chosen by the extension.

    ``.stl`` is read by the built-in parser. If the parser rejects the file and
    trimesh is installed, trimesh is tried and the geometry records
    ``reader="trimesh-fallback"`` and a warning naming the parser's reason;
    ``strict_stl=True`` disables this fallback and raises the parser's error.
    The other formats need the ``mesh`` extra (trimesh), and some of them
    further format-specific packages. Loader errors propagate to the caller.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    ext = p.suffix.lower()
    notes: list[str] = []
    if ext in NATIVE_EXTENSIONS:
        nodes, faces, reader, version, note = _load_stl(p, True, 12, strict_stl)
        if note:
            notes.append(note)
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
        reader, version = READER_TRIMESH, trimesh.__version__
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return LoadedGeometry(
        nodes, faces, p, ext.lstrip("."), ext not in NATIVE_EXTENSIONS,
        reader=reader, reader_version=version, warnings=notes,
    )


def _unique_rows_tol(points: np.ndarray, decimals: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Unique rows after rounding to ``decimals``, in order of first appearance.

    Returns ``(unique_rows, inverse_idx)``; each unique row keeps the original
    (unrounded) coordinates of its first occurrence. The order of first
    appearance is the one trimesh uses, so both readers give the same arrays.
    """
    a = np.asarray(points, dtype=float)
    ar = np.round(a, decimals=decimals)
    # View as a structured array to use np.unique on rows
    view = np.ascontiguousarray(ar).view([("f%d" % i, ar.dtype) for i in range(ar.shape[1])])
    _, idx, inv = np.unique(view, return_index=True, return_inverse=True)
    order = np.argsort(idx)  # unique rows sorted by their first occurrence
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return a[idx[order]], rank[np.asarray(inv).reshape(-1)]


_FACET = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def _check_finite(triangles: np.ndarray, kind: str) -> np.ndarray:
    bad = ~np.isfinite(triangles).all(axis=(1, 2))
    if bad.any():
        raise _NonFiniteSTL(
            f"{kind} STL: non-finite coordinate (NaN or infinity) in triangle "
            f"{int(np.argmax(bad)) + 1}"
        )
    return triangles


def _parse_stl_binary(data: bytes) -> np.ndarray:
    """Return triangles array (T, 3, 3) from binary STL bytes.

    The file size must be exactly 84 + 50 * (triangle count in the header).
    """
    if len(data) < 84:
        raise STLParseError(
            f"binary STL: the file has {len(data)} bytes, fewer than the 84-byte header"
        )
    n_tri = struct.unpack_from("<I", data, offset=80)[0]
    expect = 84 + 50 * n_tri
    if len(data) != expect:
        raise STLParseError(
            f"binary STL: the header gives {n_tri} triangles ({expect} bytes) "
            f"but the file has {len(data)} bytes"
        )
    if n_tri == 0:
        raise STLParseError("binary STL: the header gives 0 triangles")
    recs = np.frombuffer(data, dtype=_FACET, count=n_tri, offset=84)
    return _check_finite(recs["v"].astype(np.float64), "binary")


def _parse_stl_ascii(text: str) -> np.ndarray:
    """Return triangles array (T, 3, 3) from ASCII STL text.

    Only ``vertex`` lines are read, in groups of three: normals, ``outer
    loop`` and ``solid`` lines (several solids are concatenated) are ignored.
    """
    tris = []
    cur = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.lower().startswith("vertex"):
            continue
        parts = line.split()
        # Format: vertex x y z
        if len(parts) != 4:
            raise STLParseError(
                f"ASCII STL, line {number}: a vertex must have exactly three coordinates"
            )
        try:
            cur.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except ValueError:
            raise STLParseError(f"ASCII STL, line {number}: invalid coordinate") from None
        if len(cur) == 3:
            tris.append(cur)
            cur = []
    if cur:
        raise STLParseError("ASCII STL: incomplete triangle at the end of the file")
    if not tris:
        raise STLParseError("ASCII STL: no triangles found")
    return _check_finite(np.asarray(tris, dtype=np.float64), "ASCII")


_ASCII_VERTEX = re.compile(rb"^[ \t]*vertex[ \t]", re.IGNORECASE | re.MULTILINE)


def _parse_stl(data: bytes) -> np.ndarray:
    """Triangles (T, 3, 3) of an STL file, binary or ASCII.

    A file whose size is exactly 84 + 50 * (count in the header) is binary,
    even if its header starts with ``solid``; anything else is read as ASCII.
    The text is decoded byte by byte (Latin-1), so names in any 8-bit encoding
    are accepted. Raises STLParseError with the reason, naming the binary or the
    ASCII reading depending on what the file looks like.
    """
    try:
        return _parse_stl_binary(data)
    except STLParseError as exc:
        binary_reason = exc
    try:
        return _parse_stl_ascii(data.decode("latin-1"))
    except STLParseError as exc:
        ascii_reason = exc
    head = data.lstrip(b"\xef\xbb\xbf \t\r\n")[:5].lower()
    looks_ascii = _ASCII_VERTEX.search(data) is not None or (
        head == b"solid" and b"\0" not in data
    )
    raise ascii_reason if looks_ascii else binary_reason


def _merge(triangles: np.ndarray, merge_vertices: bool, decimals: int) -> Tuple[np.ndarray, np.ndarray]:
    verts = triangles.reshape(-1, 3)
    if merge_vertices:
        nodes, inv = _unique_rows_tol(verts, decimals=decimals)
        return nodes, inv.reshape(-1, 3).astype(np.int32)
    return verts.astype(np.float64), np.arange(verts.shape[0], dtype=np.int32).reshape(-1, 3)


def _trimesh_stl_triangles(trimesh, path: Path) -> np.ndarray:
    """Triangles of an STL file read by trimesh, without its processing."""
    mesh = trimesh.load(str(path), file_type="stl", process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces, dtype=np.int64)]
    if triangles.size == 0:
        raise ValueError("no triangles read")
    return _check_finite(triangles.reshape(-1, 3, 3), "trimesh")


def _load_stl(path: Path, merge_vertices: bool, decimals: int, strict: bool):
    """(nodes, faces, reader, reader_version, warning or None) of an STL file."""
    try:
        triangles = _parse_stl(path.read_bytes())
    except _NonFiniteSTL as exc:
        raise ValueError(f"Could not parse STL file: {exc}") from exc
    except STLParseError as exc:
        reason = str(exc)
        if strict:
            raise STLParseError(
                f"Could not parse STL file: {reason} (strict STL reading: trimesh not tried)"
            ) from exc
        trimesh = _get_trimesh()
        if trimesh is None:
            raise STLParseError(
                f"Could not parse STL file: {reason}. trimesh, which could try another "
                "reader, is not installed (pip install \"pyshape[mesh]\")."
            ) from exc
        try:
            triangles = _trimesh_stl_triangles(trimesh, path)
        except Exception as fallback_exc:  # trimesh raises many exception types
            raise STLParseError(
                f"Could not parse STL file: {reason} (trimesh {trimesh.__version__} "
                f"could not read it either: {fallback_exc})"
            ) from exc
        nodes, faces = _merge(triangles, merge_vertices, decimals)
        note = (f"STL file read by trimesh {trimesh.__version__}: the built-in parser "
                f"rejected it ({reason}).")
        return nodes, faces, READER_TRIMESH_FALLBACK, trimesh.__version__, note
    nodes, faces = _merge(triangles, merge_vertices, decimals)
    return nodes, faces, READER_BUILTIN_STL, None, None


def load_stl(
    path: str | Path, merge_vertices: bool = True, decimals: int = 12, *, strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an STL file and return (nodes, faces).

    The file is read by the built-in ASCII/binary parser. If the parser
    rejects it and trimesh is installed, trimesh is tried (without its own
    processing; vertices are then merged as below) and an
    :class:`STLFallbackWarning` names the parser's reason. Use
    :func:`load_geometry` to get the reader and the warning as data.

    Parameters
    ----------
    path : str or Path
        Path to an STL file.
    merge_vertices : bool, default True
        If True, merge duplicate vertices, kept in order of first appearance.
    decimals : int, default 12
        Coordinates are rounded to this many decimals to find duplicates; the
        merged vertices keep their original coordinates.
    strict : bool, default False
        If True, never use trimesh: raise the parser's error instead.

    Returns
    -------
    nodes : (N, 3) float64
    faces : (M, 3) int32

    Raises
    ------
    STLParseError
        The file cannot be read (a subclass of ValueError).
    ValueError
        A coordinate is NaN or infinite (trimesh is not tried).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"STL not found: {p}")
    nodes, faces, _, _, note = _load_stl(p, merge_vertices, decimals, strict)
    if note:
        warnings.warn(note, STLFallbackWarning, stacklevel=2)
    return nodes, faces
