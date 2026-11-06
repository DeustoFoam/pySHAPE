"""Oriented bounding boxes used to estimate the principal particle dimensions.

Two methods are provided, mirroring the ``options.Auxiliary_Geometries.OBB``
choices of the original MATLAB SHAPE:

``pca``
    Box aligned with the principal directions of the vertex covariance
    (SHAPE's ``PCA_points`` with ``Surface_points``). NumPy only. The
    directions are not unique when the covariance is isotropic or has repeated
    eigenvalues (e.g. cubes, regular tetrahedra/octahedra/icosahedra), in which
    case the box, and every index derived from it, depends on mesh orientation.

``min_volume``
    Minimum-volume box over a rotation-invariant set of candidate orientations
    derived from the convex hull (SHAPE's ``minVolume``, which uses the
    hull-based search of J. Korsawe's ``minboundbox``): one box axis is
    either normal to a hull facet or parallel to a hull edge, and the two
    remaining axes are found exactly with a two-dimensional minimum-area
    rectangle (rotating calipers); for small hulls, axes normal to pairs of
    hull edges are tested too. The best candidates are then refined by a local
    Nelder-Mead search over the box axis, each step again using the exact
    rectangle search. This recovers the exact box of polyhedra such as the
    Platonic solids but, like ``minboundbox``, it does not prove global
    optimality for arbitrary point sets. Requires SciPy.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

AXES_METHODS = ("pca", "min_volume")
# Hulls with at most this many distinct edge directions also test the axes
# perpendicular to every pair of edges (at most ~45,000 extra candidates).
MAX_EDGE_PAIR_DIRECTIONS = 300
# Number of best discrete candidates refined by a local (Nelder-Mead) search.
REFINEMENT_STARTS = 10


class OrientedBox(NamedTuple):
    """Box edge lengths sorted as short, intermediate, long, with directions.

    ``directions`` rows are unit vectors parallel to the S, I and L edges.
    """

    short: float
    intermediate: float
    long: float
    directions: np.ndarray
    center: np.ndarray

    @property
    def volume(self) -> float:
        return float(self.short * self.intermediate * self.long)


def _sorted_box(spans: np.ndarray, directions: np.ndarray, center: np.ndarray) -> OrientedBox:
    order = np.argsort(spans, kind="stable")
    s, i, l = (float(spans[k]) for k in order)
    return OrientedBox(s, i, l, np.asarray(directions)[order], center)


def pca_box(nodes: ArrayLike) -> OrientedBox:
    """Bounding box aligned with the principal directions of the vertices."""
    points = np.asarray(nodes, dtype=float)
    mean = points.mean(axis=0)
    centered = points - mean
    directions = np.linalg.svd(centered, full_matrices=False)[2]
    projected = centered @ directions.T
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    center = mean + ((lo + hi) / 2) @ directions
    return _sorted_box(hi - lo, directions, center)


def _perpendicular_bases(axes: np.ndarray) -> np.ndarray:
    """Return (N, 3, 2) arrays whose columns u, v complete each unit axis."""
    helpers = np.eye(3)[np.argmin(np.abs(axes), axis=1)]
    u = np.cross(axes, helpers)
    u /= np.linalg.norm(u, axis=1)[:, None]
    return np.stack((u, np.cross(axes, u)), axis=2)


def _canonical_directions(vectors: np.ndarray) -> np.ndarray:
    """Unit vectors with a sign convention, duplicates (within 1e-10) removed."""
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors[norms > 0] / norms[norms > 0, None]
    pivot = np.argmax(np.abs(vectors) > 1e-12, axis=1)
    signs = np.sign(vectors[np.arange(len(vectors)), pivot])
    vectors = vectors * signs[:, None]
    _, unique = np.unique(np.round(vectors, 10), axis=0, return_index=True)
    return vectors[np.sort(unique)]


def min_volume_box(nodes: ArrayLike) -> OrientedBox:
    """Minimum-volume oriented bounding box from convex-hull candidates.

    See the module docstring for the search strategy and its guarantees.
    Raises ``ValueError`` when the points do not span three dimensions.
    """
    from scipy.spatial import ConvexHull, QhullError

    points = np.asarray(nodes, dtype=float)
    origin = points.mean(axis=0)
    scale = float(np.ptp(points, axis=0).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot construct a bounding box for coincident points")
    try:
        hull = ConvexHull((points - origin) / scale)
    except QhullError as exc:
        raise ValueError("Cannot construct a three-dimensional convex hull") from exc

    # Work only with hull vertices, renumbered 0..m-1.
    vertex_ids = hull.vertices
    hull_points = hull.points[vertex_ids]
    remap = np.full(len(hull.points), -1, dtype=np.intp)
    remap[vertex_ids] = np.arange(len(vertex_ids))
    simplices = remap[hull.simplices]
    normals = hull.equations[:, :3]

    # Facet adjacency across each hull edge: neighbors[i, j] is opposite vertex j.
    facet_a = np.repeat(np.arange(len(simplices)), 3)
    facet_b = hull.neighbors.ravel()
    keep = facet_a < facet_b
    corner = np.tile(np.arange(3), len(simplices))
    edge_v0 = simplices[facet_a, (corner + 1) % 3][keep]
    edge_v1 = simplices[facet_a, (corner + 2) % 3][keep]
    facet_a, facet_b = facet_a[keep], facet_b[keep]

    edge_directions = _canonical_directions(hull_points[edge_v1] - hull_points[edge_v0])
    candidate_sets = [normals, edge_directions]
    if len(edge_directions) <= MAX_EDGE_PAIR_DIRECTIONS:
        # Axes perpendicular to two hull edges (O'Rourke's condition for two
        # adjacent box faces flush with hull edges), e.g. the cube enclosing a
        # regular tetrahedron. Affordable for polyhedral hulls only.
        first, second = np.triu_indices(len(edge_directions), k=1)
        candidate_sets.append(np.cross(edge_directions[first], edge_directions[second]))
    candidates = _canonical_directions(np.vstack(candidate_sets))

    # Exact lower bound for each candidate axis: height x shadow area, because
    # any rectangle enclosing the shadow is at least as large as the shadow.
    # Candidates are evaluated in increasing bound order and the search stops
    # when the bound exceeds the best volume found, without changing the result.
    corners = hull_points[simplices]
    facet_areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    heights = np.empty(len(candidates))
    bounds = np.empty(len(candidates))
    for start in range(0, len(candidates), 256):
        chunk = candidates[start:start + 256].T
        projected = hull_points @ chunk
        heights[start:start + 256] = projected.max(axis=0) - projected.min(axis=0)
        shadows = 0.5 * (facet_areas @ np.abs(normals @ chunk))
        bounds[start:start + 256] = heights[start:start + 256] * shadows
    bases = _perpendicular_bases(candidates)
    edge_vectors = hull_points[edge_v1] - hull_points[edge_v0]
    on_outline = np.zeros(len(hull_points), dtype=bool)
    eps = 1e-12

    def evaluate(axis: np.ndarray, basis: np.ndarray, height: float):
        """Exact minimum box volume and axes for a given box axis."""
        # The shadow boundary is the projection of the silhouette: hull edges
        # whose two facets are not strictly on the same side of the plane
        # perpendicular to the axis. Its edge directions are the only
        # orientations a minimum-area enclosing rectangle needs to test
        # (zero-length projected edges only add a harmless extra angle).
        side = normals @ axis
        code = (side > eps).astype(np.int8) - (side < -eps).astype(np.int8)
        silhouette = np.flatnonzero(code[facet_a] * code[facet_b] <= 0)
        if not len(silhouette) or height <= 0:
            return np.inf, None
        on_outline[:] = False
        on_outline[edge_v0[silhouette]] = True
        on_outline[edge_v1[silhouette]] = True
        planar = hull_points[on_outline] @ basis
        steps = edge_vectors[silhouette] @ basis
        angles = np.unique(np.mod(np.arctan2(steps[:, 1], steps[:, 0]), np.pi / 2))
        cos, sin = np.cos(angles)[:, None], np.sin(angles)[:, None]
        x = cos * planar[:, 0] + sin * planar[:, 1]
        y = cos * planar[:, 1] - sin * planar[:, 0]
        areas = (x.max(axis=1) - x.min(axis=1)) * (y.max(axis=1) - y.min(axis=1))
        k = int(np.argmin(areas))
        c, s = float(cos[k, 0]), float(sin[k, 0])
        u, v = basis[:, 0], basis[:, 1]
        return float(areas[k] * height), np.array([c * u + s * v, c * v - s * u, axis])

    best_volume = np.inf
    best = None
    evaluated: list[tuple[float, int]] = []
    for index in np.argsort(bounds, kind="stable"):
        if bounds[index] >= best_volume * (1 + 1e-12):
            break
        volume, directions = evaluate(candidates[index], bases[index], heights[index])
        evaluated.append((volume, int(index)))
        if volume < best_volume:
            best_volume, best = volume, directions
    if best is None:
        raise ValueError("Cannot construct a three-dimensional bounding box")

    # Local refinement: Nelder-Mead over the box axis (two tangent-plane
    # coordinates) from the best discrete candidates. Every evaluation is the
    # exact rectangle search above, so the result can only improve.
    from scipy.optimize import minimize

    def objective(step, axis, basis):
        trial = axis + basis @ step
        trial = trial / np.linalg.norm(trial)
        projected = hull_points @ trial
        return evaluate(trial, _perpendicular_bases(trial[None])[0],
                        float(projected.max() - projected.min()))

    for _, index in sorted(evaluated)[:REFINEMENT_STARTS]:
        axis, basis = candidates[index], bases[index]
        result = minimize(lambda step: objective(step, axis, basis)[0], np.zeros(2),
                          method="Nelder-Mead",
                          options={"initial_simplex": [[0, 0], [0.02, 0], [0, 0.02]],
                                   "xatol": 1e-10, "fatol": 1e-15 * best_volume,
                                   "maxiter": 400})
        volume, directions = objective(result.x, axis, basis)
        if volume < best_volume * (1 - 1e-12):
            best_volume, best = volume, directions

    directions = best
    projected = (points - origin) @ directions.T
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    center = origin + ((lo + hi) / 2) @ directions
    return _sorted_box(hi - lo, directions, center)


def oriented_box(nodes: ArrayLike, method: str = "pca") -> OrientedBox:
    """Dispatch to :func:`pca_box` or :func:`min_volume_box`."""
    if method == "pca":
        return pca_box(nodes)
    if method == "min_volume":
        return min_volume_box(nodes)
    raise ValueError(f"Unknown axes method '{method}'. Choose one of: {', '.join(AXES_METHODS)}")
