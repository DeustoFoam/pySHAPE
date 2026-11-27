"""Diagnosis of open meshes and a conservative repair step (``pyshape-repair``).

pySHAPE analyses the surface it is given: :func:`pyshape.analyze_file` never
changes the geometry, and an open surface makes the volume-based results
unavailable. This module helps with one common cause of open surfaces, small
gaps left by export or rounding, without weakening that rule:

* :func:`diagnose_mesh` counts the boundary and non-manifold edges and the
  boundary vertices that lie within a small tolerance of another boundary
  vertex. When a mesh is open, the analysis adds its :meth:`MeshDiagnosis.message`
  to the result warnings, saying whether the openings look like such gaps or
  like holes.
* :func:`repair_mesh` closes such gaps (stitching) and removes degenerate and
  duplicate triangles. It returns a new mesh and a :class:`RepairReport`.
* ``pyshape-repair`` (:func:`main`) applies it to files and writes the repaired
  mesh as a new file, with the report as JSON next to it, so that the repaired
  geometry can be inspected, kept and analysed like any other file.

Holes are not filled, non-manifold edges are not split, and T-junctions (a
vertex lying on another triangle's edge) are not resolved: these change the
geometry by more than a bounded amount and are left to dedicated tools such as
MeshLab, PyMeshFix or Blender's 3D-Print toolbox.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .geometry import surface_area
from .preprocess import validate_mesh

__all__ = [
    "DEFAULT_REL_TOL", "MeshDiagnosis", "RepairReport", "diagnose_mesh", "open_mesh_warning",
    "repair_mesh",
]

#: Default stitching tolerance, as a fraction of the median edge length.
DEFAULT_REL_TOL = 1e-3

_REPAIR_TOOLS = "MeshLab, PyMeshFix or Blender's 3D-Print toolbox"


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #

@dataclass
class MeshDiagnosis:
    """What :func:`diagnose_mesh` found; nothing is changed.

    ``n_near_boundary_vertices`` counts the boundary vertices (on an edge used
    by one triangle) that lie within ``tolerance`` of another boundary vertex
    with which they share no edge: the vertices that :func:`repair_mesh` would
    try to join. ``rel_tol`` is ``None`` when an absolute tolerance was given.
    Degenerate faces repeat a vertex index; duplicate faces use the same three
    vertices as an earlier face; zero-area faces have three distinct vertices
    but no area (they are counted, not removed).
    """

    n_vertices: int
    n_faces: int
    n_boundary_edges: int
    n_nonmanifold_edges: int
    n_boundary_vertices: int
    n_near_boundary_vertices: int
    n_degenerate_faces: int
    n_duplicate_faces: int
    n_zero_area_faces: int
    median_edge_length: float
    tolerance: float
    rel_tol: Optional[float]

    @property
    def closed(self) -> bool:
        """Every edge is shared by exactly two triangles."""
        return self.n_boundary_edges == 0 and self.n_nonmanifold_edges == 0

    def to_dict(self) -> dict[str, object]:
        """The fields and ``closed``, JSON-serialisable."""
        return {**asdict(self), "closed": self.closed}

    def message(self) -> Optional[str]:
        """One warning text for an open or non-manifold mesh; ``None`` if closed."""
        if self.closed:
            return None
        edges = _count(self.n_boundary_edges, "boundary edge")
        if self.n_nonmanifold_edges:
            edges += (f" and {_count(self.n_nonmanifold_edges, 'non-manifold edge')} (shared "
                      "by more than two triangles)")
        parts = [f"Open surface: {edges}."]
        within = f"within {self.tolerance:.3g}"
        if self.rel_tol is not None:
            within += f" ({self.rel_tol:g} x the median edge length)"
        near, total = self.n_near_boundary_vertices, self.n_boundary_vertices
        if total and near == total:
            parts.append(
                f"All {total} boundary vertices lie {within} of another boundary vertex: the "
                "openings are probably gaps left by export or rounding, which pyshape-repair "
                "can close (it writes a repaired copy of the file).")
        elif near:
            parts.append(
                f"{near} of the {total} boundary vertices lie {within} of another boundary "
                "vertex: pyshape-repair can close those gaps; the other openings are holes, "
                f"which pySHAPE does not fill (use e.g. {_REPAIR_TOOLS}).")
        elif total:
            parts.append(
                f"No boundary vertex lies {within} of another: the openings are holes rather "
                f"than export gaps; pySHAPE does not fill holes (use e.g. {_REPAIR_TOOLS}).")
        removable = []
        if self.n_duplicate_faces:
            removable.append(f"{self.n_duplicate_faces} duplicate")
        if self.n_degenerate_faces:
            removable.append(f"{self.n_degenerate_faces} degenerate")
        if removable:
            noun = "face" if self.n_duplicate_faces + self.n_degenerate_faces == 1 else "faces"
            parts.append(f"The mesh has {' and '.join(removable)} {noun}, which pyshape-repair "
                         "removes.")
        if self.n_nonmanifold_edges and not self.n_duplicate_faces:
            parts.append("Non-manifold edges are not repaired.")
        return " ".join(parts)


def _count(n: int, word: str) -> str:
    """``"1 edge"``, ``"2 edges"``; ``word`` is singular, regular plural."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _edge_list(faces: np.ndarray) -> np.ndarray:
    return np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)


def _edge_counts(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique edges (as sorted vertex pairs) and the number of triangles using each."""
    if len(faces) == 0:
        return np.empty((0, 2), dtype=np.intp), np.empty(0, dtype=np.intp)
    return np.unique(_edge_list(faces), axis=0, return_counts=True)


def _degenerate(faces: np.ndarray) -> np.ndarray:
    return (faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2]) | (faces[:, 0] == faces[:, 2])


def _duplicates(faces: np.ndarray) -> np.ndarray:
    """Mask of faces whose vertex set already appeared in an earlier face."""
    mask = np.zeros(len(faces), dtype=bool)
    if len(faces):
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        mask[:] = True
        mask[first] = False
    return mask


def _median_edge(nodes: np.ndarray, edges: np.ndarray) -> float:
    edges = edges[edges[:, 0] != edges[:, 1]]
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    lengths = lengths[lengths > 0]
    return float(np.median(lengths)) if len(lengths) else 0.0


def _close_pairs(points: np.ndarray, tol: float) -> np.ndarray:
    """Index pairs ``(i, j)``, ``i < j``, of points at most ``tol`` apart, sorted.

    Uses SciPy's k-d tree when it is installed and a grid of cells of side
    ``tol`` otherwise; both return the same pairs.
    """
    if len(points) < 2 or not tol > 0:
        return np.empty((0, 2), dtype=np.intp)
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        pairs = np.asarray(cKDTree(points).query_pairs(tol, output_type="ndarray"), dtype=np.intp)
    else:
        cells = np.floor((points - points.min(axis=0)) / tol).astype(np.int64)
        buckets: dict[tuple, list[int]] = {}
        for index, cell in enumerate(map(tuple, cells)):
            buckets.setdefault(cell, []).append(index)
        found = []
        offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
        for (x, y, z), members in buckets.items():
            near = [j for dx, dy, dz in offsets for j in buckets.get((x + dx, y + dy, z + dz), ())]
            near = np.asarray(near, dtype=np.intp)
            for i in members:
                others = near[near > i]
                dist = np.linalg.norm(points[others] - points[i], axis=1)
                found.extend((i, int(j)) for j in others[dist <= tol])
        pairs = np.asarray(found, dtype=np.intp).reshape(-1, 2)
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.intp)
    pairs = np.sort(pairs, axis=1)
    return pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]


def _stitch_candidates(
    nodes: np.ndarray, boundary: np.ndarray, edges: np.ndarray, tol: float,
) -> np.ndarray:
    """Pairs of boundary vertices (global indices) within ``tol`` that share no edge."""
    pairs = boundary[_close_pairs(nodes[boundary], tol)]
    if len(pairs) == 0:
        return pairs
    n = len(nodes)
    connected = np.isin(pairs[:, 0].astype(np.int64) * n + pairs[:, 1],
                        edges[:, 0].astype(np.int64) * n + edges[:, 1])
    return pairs[~connected]


def _tolerance(nodes: np.ndarray, edges: np.ndarray, tol: Optional[float],
               rel_tol: float) -> tuple[float, float]:
    median = _median_edge(nodes, edges)
    if tol is not None:
        if not (np.isfinite(tol) and tol >= 0):
            raise ValueError("tol must be a finite, non-negative length")
        return float(tol), median
    if not (np.isfinite(rel_tol) and rel_tol >= 0):
        raise ValueError("rel_tol must be a finite, non-negative number")
    return rel_tol * median, median


def diagnose_mesh(
    nodes: ArrayLike, faces: ArrayLike, *, tol: Optional[float] = None,
    rel_tol: float = DEFAULT_REL_TOL,
) -> MeshDiagnosis:
    """Describe the defects of a triangle mesh without changing it.

    Parameters
    ----------
    nodes, faces : array_like
        Vertices (N, 3) and triangles (M, 3), as for :func:`pyshape.analyze_mesh`.
    tol : float, optional
        Absolute distance, in the units of ``nodes``, within which two boundary
        vertices count as a gap that stitching could close.
    rel_tol : float
        Used when ``tol`` is not given: the tolerance is ``rel_tol`` times the
        median edge length (default :data:`DEFAULT_REL_TOL`).

    Returns
    -------
    MeshDiagnosis
        Counts of boundary, non-manifold, degenerate and duplicate elements;
        :meth:`MeshDiagnosis.message` gives the warning text used by the
        analysis for open meshes.
    """
    vertices, indices = validate_mesh(nodes, faces)
    degenerate = _degenerate(indices)
    proper = indices[~degenerate]
    duplicate = _duplicates(proper)
    tri = vertices[proper[~duplicate]]
    doubled = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edges, counts = _edge_counts(indices)
    tolerance, median = _tolerance(vertices, edges, tol, rel_tol)
    boundary = np.unique(edges[counts == 1])
    candidates = _stitch_candidates(vertices, boundary, edges, tolerance)
    return MeshDiagnosis(
        n_vertices=len(vertices), n_faces=len(indices),
        n_boundary_edges=int(np.count_nonzero(counts == 1)),
        n_nonmanifold_edges=int(np.count_nonzero(counts > 2)),
        n_boundary_vertices=len(boundary),
        n_near_boundary_vertices=len(np.unique(candidates)),
        n_degenerate_faces=int(np.count_nonzero(degenerate)),
        n_duplicate_faces=int(np.count_nonzero(duplicate)),
        n_zero_area_faces=int(np.count_nonzero(doubled == 0)),
        median_edge_length=median, tolerance=tolerance,
        rel_tol=None if tol is not None else rel_tol,
    )


def open_mesh_warning(nodes: ArrayLike, faces: ArrayLike) -> Optional[str]:
    """The diagnosis warning for an open or non-manifold mesh, else ``None``."""
    return diagnose_mesh(nodes, faces).message()


# --------------------------------------------------------------------------- #
# Repair
# --------------------------------------------------------------------------- #

@dataclass
class RepairReport:
    """What :func:`repair_mesh` changed, in the order the steps are applied.

    Distances are in the units of the input coordinates. ``stitched_vertices``
    counts the boundary vertices moved onto another boundary vertex, by at most
    ``max_displacement`` (never more than ``tolerance``).
    ``removed_collapsed_faces`` are triangles that lost their area because two
    of their vertices were joined. ``refused`` gives the reason when stitching
    was not applied (the other steps still are).
    """

    tolerance: float
    rel_tol: Optional[float]
    median_edge_length: float
    n_vertices_before: int
    n_faces_before: int
    merged_coincident_vertices: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    stitched_vertices: int
    max_displacement: float
    removed_collapsed_faces: int
    removed_unreferenced_vertices: int
    n_vertices_after: int
    n_faces_after: int
    boundary_edges_before: int
    boundary_edges_after: int
    nonmanifold_edges_before: int
    nonmanifold_edges_after: int
    surface_area_before: float
    surface_area_after: float
    refused: Optional[str] = None

    @property
    def closed(self) -> bool:
        """The repaired mesh has no boundary or non-manifold edge."""
        return self.boundary_edges_after == 0 and self.nonmanifold_edges_after == 0

    @property
    def changed(self) -> bool:
        """The repaired mesh differs from the input."""
        return bool(self.merged_coincident_vertices or self.removed_degenerate_faces
                    or self.removed_duplicate_faces or self.stitched_vertices
                    or self.removed_unreferenced_vertices)

    def to_dict(self) -> dict[str, object]:
        """The fields, ``closed`` and ``changed``, JSON-serialisable (as in the report file)."""
        return {**asdict(self), "closed": self.closed, "changed": self.changed}

    def summary(self) -> list[str]:
        """Human-readable lines, as printed by ``pyshape-repair``."""
        tol = f"{self.tolerance:.3g}"
        if self.rel_tol is not None:
            tol += f" ({self.rel_tol:g} x median edge length {self.median_edge_length:.3g})"
        lines = [
            f"before: {self.n_vertices_before} vertices, {self.n_faces_before} faces, "
            f"{self.boundary_edges_before} boundary and {self.nonmanifold_edges_before} "
            "non-manifold edges",
            f"tolerance: {tol}",
        ]
        steps = [
            (self.merged_coincident_vertices, "coincident vertices merged"),
            (self.removed_degenerate_faces, "degenerate faces removed"),
            (self.removed_duplicate_faces, "duplicate faces removed"),
            (self.stitched_vertices, f"boundary vertices stitched (largest move "
                                     f"{self.max_displacement:.3g})"),
            (self.removed_collapsed_faces, "faces collapsed by stitching removed"),
            (self.removed_unreferenced_vertices, "unreferenced vertices removed"),
        ]
        lines += [f"{count} {text}" for count, text in steps if count]
        if self.refused:
            lines.append(f"stitching refused: {self.refused}")
        change = self.surface_area_after - self.surface_area_before
        rel = change / self.surface_area_before if self.surface_area_before else 0.0
        lines.append(
            f"after: {self.n_vertices_after} vertices, {self.n_faces_after} faces, "
            f"{self.boundary_edges_after} boundary and {self.nonmanifold_edges_after} "
            f"non-manifold edges ({'closed' if self.closed else 'still open'}); surface area "
            f"change {change:.3g} ({rel:.2g} relative)")
        return lines


def _merge_coincident(nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge exactly coincident vertices, keeping the order of first appearance."""
    _, first, inverse = np.unique(nodes, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return nodes[first[order]], rank[np.asarray(inverse).reshape(-1)][faces]


def _compact(nodes: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.zeros(len(nodes), dtype=bool)
    used[faces.ravel()] = True
    new_index = np.cumsum(used) - 1
    return nodes[used], new_index[faces]


def _stitch(nodes: np.ndarray, faces: np.ndarray, tol: float) -> tuple[np.ndarray, float, int]:
    """Map each boundary vertex to the one it is joined to; returns (target, max move, moved).

    Vertices are visited in index order. An unassigned vertex becomes the
    representative of a group and takes, nearest first, the unassigned
    boundary vertices within ``tol`` of it that share no edge with the group,
    so a group never spreads beyond ``tol`` from its representative (no
    chaining) and no edge is collapsed. Joined vertices move onto the
    representative.
    """
    edges, counts = _edge_counts(faces)
    target = np.arange(len(nodes))
    boundary = np.unique(edges[counts == 1])
    pairs = _stitch_candidates(nodes, boundary, edges, tol)
    if len(pairs) == 0:
        return target, 0.0, 0
    neighbours: dict[int, list[int]] = {}
    for i, j in pairs.tolist():
        neighbours.setdefault(i, []).append(j)
        neighbours.setdefault(j, []).append(i)
    adjacent: dict[int, set[int]] = {}
    for i, j in edges.tolist():
        adjacent.setdefault(i, set()).add(j)
        adjacent.setdefault(j, set()).add(i)
    assigned = np.zeros(len(nodes), dtype=bool)
    largest, moved = 0.0, 0
    for v in sorted(neighbours):
        if assigned[v]:
            continue
        assigned[v] = True
        group_adjacent = set(adjacent.get(v, ()))
        candidates = neighbours[v]
        distances = np.linalg.norm(nodes[candidates] - nodes[v], axis=1)
        for w, distance in sorted(zip(candidates, distances.tolist()), key=lambda c: (c[1], c[0])):
            if assigned[w] or w in group_adjacent:
                continue
            assigned[w] = True
            target[w] = v
            group_adjacent |= adjacent.get(w, set())
            largest, moved = max(largest, distance), moved + 1
    return target, largest, moved


def repair_mesh(
    nodes: ArrayLike, faces: ArrayLike, *, tol: Optional[float] = None,
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[np.ndarray, np.ndarray, RepairReport]:
    """Close small gaps and remove degenerate and duplicate triangles.

    Steps, each counted in the report:

    1. Exactly coincident vertices are merged.
    2. Degenerate triangles (a repeated vertex index) and duplicate triangles
       (the same three vertices as an earlier triangle, in any order) are
       removed. This changes neither the area nor the volume.
    3. Stitching: boundary vertices (on an edge used by one triangle) within
       the tolerance of another boundary vertex are joined to it. The
       tolerance is ``tol`` or, by default, ``rel_tol`` times the median edge
       length. Only boundary vertices move, by at most the tolerance; two
       vertices joined by an edge are never joined, and groups do not chain
       (see :func:`_stitch`). Triangles left without area are removed.
       Stitching is refused, and the mesh is returned after step 2 only, if it
       would create a non-manifold edge, a duplicate triangle or more
       boundary edges.
    4. Vertices no longer used by any triangle are removed.

    Holes wider than the tolerance stay open; holes are never filled,
    non-manifold edges are not split and the winding is not changed
    (:func:`pyshape.prepare_mesh` orients the shells during the analysis).
    The volume changes by about the tolerance times the area of the stitched
    region; the report gives the largest displacement and the area before and
    after.

    Parameters
    ----------
    nodes, faces : array_like
        Vertices (N, 3) and triangles (M, 3); one-based faces are recognised as
        in :func:`pyshape.validate_mesh`. The inputs are not modified.
    tol : float, optional
        Absolute stitching tolerance in the units of ``nodes``; ``0`` disables
        stitching.
    rel_tol : float
        Stitching tolerance as a fraction of the median edge length, used when
        ``tol`` is not given (default :data:`DEFAULT_REL_TOL`).

    Returns
    -------
    nodes : (N', 3) ndarray of float
    faces : (M', 3) ndarray of int
        The repaired mesh (copies; equal to the validated input when nothing
        was changed).
    report : RepairReport
    """
    vertices, indices = validate_mesh(nodes, faces)
    vertices, indices = vertices.copy(), indices.copy()
    n_vertices, n_faces = len(vertices), len(indices)
    edges, counts = _edge_counts(indices)
    boundary_before = int(np.count_nonzero(counts == 1))
    nonmanifold_before = int(np.count_nonzero(counts > 2))
    area_before = surface_area(vertices, indices)

    vertices, indices = _merge_coincident(vertices, indices)
    merged = n_vertices - len(vertices)
    degenerate = _degenerate(indices)
    indices = indices[~degenerate]
    duplicate = _duplicates(indices)
    indices = indices[~duplicate]

    edges, counts = _edge_counts(indices)
    tolerance, median = _tolerance(vertices, edges, tol, rel_tol)
    clean_boundary = int(np.count_nonzero(counts == 1))
    clean_nonmanifold = int(np.count_nonzero(counts > 2))
    target, largest, stitched = _stitch(vertices, indices, tolerance)
    refused, collapsed = None, 0
    if stitched:
        joined = target[indices]
        collapsed_mask = _degenerate(joined)
        joined = joined[~collapsed_mask]
        new_duplicates = int(np.count_nonzero(_duplicates(joined)))
        _, new_counts = _edge_counts(joined)
        new_nonmanifold = int(np.count_nonzero(new_counts > 2))
        if new_nonmanifold > clean_nonmanifold:
            refused = (f"it would create {new_nonmanifold - clean_nonmanifold} non-manifold "
                       "edge(s) (shared by more than two triangles)")
        elif new_duplicates:
            refused = f"it would create {new_duplicates} duplicate triangle(s)"
        elif np.count_nonzero(new_counts == 1) > clean_boundary:
            refused = "it would leave more boundary edges than before"
        if refused is None:
            indices, collapsed = joined, int(np.count_nonzero(collapsed_mask))
        else:
            stitched, largest = 0, 0.0
    n_before_compact = len(vertices)
    vertices, indices = _compact(vertices, indices)
    unreferenced = n_before_compact - len(vertices)
    if len(indices) == 0:
        raise ValueError("repair would remove every triangle")

    _, counts = _edge_counts(indices)
    report = RepairReport(
        tolerance=tolerance, rel_tol=None if tol is not None else rel_tol,
        median_edge_length=median, n_vertices_before=n_vertices, n_faces_before=n_faces,
        merged_coincident_vertices=merged,
        removed_degenerate_faces=int(np.count_nonzero(degenerate)),
        removed_duplicate_faces=int(np.count_nonzero(duplicate)),
        stitched_vertices=stitched, max_displacement=largest,
        removed_collapsed_faces=collapsed, removed_unreferenced_vertices=unreferenced,
        n_vertices_after=len(vertices), n_faces_after=len(indices),
        boundary_edges_before=boundary_before,
        boundary_edges_after=int(np.count_nonzero(counts == 1)),
        nonmanifold_edges_before=nonmanifold_before,
        nonmanifold_edges_after=int(np.count_nonzero(counts > 2)),
        surface_area_before=area_before, surface_area_after=surface_area(vertices, indices),
        refused=refused,
    )
    return vertices, indices, report


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _output_path(source: Path, suffix: str, avoid_overwrite: bool) -> Path:
    candidate = source.with_name(f"{source.stem}{suffix}.stl")
    number = 1
    while avoid_overwrite and candidate.exists():
        candidate = source.with_name(f"{source.stem}{suffix}_{number}.stl")
        number += 1
    return candidate


def _write_mesh(path: Path, nodes: np.ndarray, faces: np.ndarray, name: str) -> None:
    from .io import _get_trimesh, write_stl

    if path.suffix.lower() == ".stl":
        write_stl(path, nodes, faces, name=name)
        return
    trimesh = _get_trimesh()
    if trimesh is None:
        raise ImportError(f"Writing '{path.suffix}' files requires pyshape[mesh]; use .stl.")
    trimesh.Trimesh(vertices=nodes, faces=faces, process=False).export(str(path))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``pyshape-repair FILE [FILE ...] [options]``; exit code 1 if a file failed."""
    from . import __version__
    from .io import load_geometry

    parser = argparse.ArgumentParser(
        prog="pyshape-repair",
        description="Close small gaps (left by export or rounding) in particle meshes and "
                    "remove degenerate and duplicate triangles. The repaired mesh is written "
                    "to a new file, with a JSON report next to it; the input is not changed. "
                    "Holes are not filled.")
    parser.add_argument("files", nargs="+", type=Path, help="mesh files to repair")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file, for a single input (default: <name><suffix>.stl "
                             "next to the input); .stl is written by pySHAPE, other "
                             "extensions by trimesh")
    parser.add_argument("--suffix", default="_repaired",
                        help="appended to the name of each output file (default: _repaired)")
    tolerance = parser.add_mutually_exclusive_group()
    tolerance.add_argument("--tol", type=float,
                           help="absolute stitching tolerance, in the units of the file")
    tolerance.add_argument("--rel-tol", type=float, default=DEFAULT_REL_TOL,
                           help="stitching tolerance as a fraction of the median edge length "
                                f"(default: {DEFAULT_REL_TOL:g})")
    parser.add_argument("--check", action="store_true",
                        help="only report the defects and what repair would do; write nothing")
    parser.add_argument("--no-overwrite", action="store_true",
                        help="number new output files instead of replacing existing ones")
    parser.add_argument("--strict-stl", action="store_true",
                        help="read STL files only with the built-in parser")
    args = parser.parse_args(argv)
    if args.output is not None and len(args.files) > 1:
        parser.error("--output can be used with one input file only")

    failed = 0
    for source in args.files:
        try:
            geo = load_geometry(source, strict_stl=args.strict_stl)
            print(f"{source}: {len(geo.nodes)} vertices, {len(geo.faces)} faces", flush=True)
            for text in geo.warnings:
                print(f"    warning: {text}")
            if args.check:
                diagnosis = diagnose_mesh(geo.nodes, geo.faces, tol=args.tol, rel_tol=args.rel_tol)
                print(f"    {diagnosis.message() or 'closed surface: nothing to stitch.'}")
                continue
            nodes, faces, report = repair_mesh(geo.nodes, geo.faces, tol=args.tol,
                                               rel_tol=args.rel_tol)
            for line in report.summary():
                print(f"    {line}")
            if not report.closed:
                remaining = diagnose_mesh(nodes, faces, tol=args.tol, rel_tol=args.rel_tol)
                print(f"    {remaining.message()}")
            if not report.changed:
                print("    nothing changed; no file written")
                continue
            out = args.output or _output_path(source, args.suffix, args.no_overwrite)
            if out.resolve() == source.resolve():
                raise ValueError("the output would replace the input file")
            _write_mesh(out, nodes, faces, f"{source.stem} repaired by pySHAPE {__version__}")
            record = {
                "pyshape_version": __version__,
                "input": source.name,
                "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "input_reader": geo.reader,
                "output": out.name,
                "settings": {"tol": args.tol, "rel_tol": None if args.tol is not None
                             else args.rel_tol},
                "report": report.to_dict(),
            }
            report_path = out.with_suffix(".json")
            report_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            print(f"    -> {out} (report: {report_path.name})", flush=True)
        except Exception as exc:  # report and go on with the next file
            failed += 1
            print(f"{source}: FAILED: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
