"""Principal dimensions from an equivalent ellipsoid (issue #6).

Two ellipsoids give the short, intermediate and long dimensions S, I, L as
full axis lengths (twice the semi-axes c <= b <= a):

``inertia_ellipsoid``
    The uniform ellipsoid with the same principal moments of inertia as the
    particle, for the particle's volume V. With I1 <= I2 <= I3 (unit density),
    a^2 = 5 (I2 + I3 - I1) / (2 V), b^2 = 5 (I1 + I3 - I2) / (2 V) and
    c^2 = 5 (I1 + I2 - I3) / (2 V), along the principal directions of inertia.
    It needs a closed surface, no fitting and no extra dependency. It is
    unique, rotation-invariant and insensitive to vertex density. Its volume
    4/3 pi a b c is in general not V, and its axes are not enclosing
    dimensions: they differ from box dimensions by a shape-dependent factor
    (for a cuboid, sqrt(5/3) ~ 1.29 times the sides).

``fitted_ellipsoid``
    Least-squares fit of a quadric
    A x^2 + B y^2 + C z^2 + 2D xy + 2E xz + 2F yz + 2G x + 2H y + 2I z + J = 0
    to the surface vertices, with the constraint A + B + C = 3, as in the
    ``ellipsoid_fit`` routine (Y. Petrov, 2015) used by SHAPE
    (Angelidakis et al., Comput. Phys. Commun. 265 (2021) 107983). Unlike that
    routine, the vertices are weighted by their share of the surface area by
    default (one third of the area of each adjacent triangle), so that the fit
    does not depend on how vertices are distributed, and the fitted quadric
    is checked: if it is not an ellipsoid, a ``ValueError`` is raised instead
    of returning signed radii. ``weighting="none"`` uses unweighted vertices,
    as SHAPE does. The fit is invariant to rotation and translation.

Both ellipsoids are placed with the particle, never at the coordinate origin:
the inertia-equivalent ellipsoid is centred on the barycentre (volume
centroid) and the fitted ellipsoid on the centre of the fitted quadric, which
is not constrained to the barycentre. ``Ellipsoid.center`` gives the centre;
S, I, L do not depend on it or on the position of the particle.

In :func:`pyshape.compute_metrics` (``axes="inertia_ellipsoid"`` or
``"fitted_ellipsoid"``), both use the outer surface when cavities are kept
(``cavities="keep"``): the envelope's volume and inertia, or its vertices.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

ELLIPSOID_METHODS = ("inertia_ellipsoid", "fitted_ellipsoid")


class Ellipsoid(NamedTuple):
    """Semi-axes ``a >= b >= c`` with unit ``directions`` (rows: S, I, L) and ``center``.

    ``center`` is in the coordinates of the mesh: the barycentre for
    :func:`inertia_ellipsoid`, the centre of the fitted quadric for
    :func:`fitted_ellipsoid`.
    """

    a: float
    b: float
    c: float
    directions: np.ndarray
    center: np.ndarray

    @property
    def short(self) -> float:
        """S = 2c."""
        return 2.0 * self.c

    @property
    def intermediate(self) -> float:
        """I = 2b."""
        return 2.0 * self.b

    @property
    def long(self) -> float:
        """L = 2a."""
        return 2.0 * self.a

    @property
    def volume(self) -> float:
        """Volume 4/3 pi a b c of the ellipsoid (not, in general, the particle's)."""
        return float(4.0 / 3.0 * np.pi * self.a * self.b * self.c)


def _sorted(semi_axes: np.ndarray, directions: np.ndarray, center: np.ndarray) -> Ellipsoid:
    """Order semi-axes so that a >= b >= c; ``directions`` columns match ``semi_axes``."""
    order = np.argsort(semi_axes, kind="stable")  # c, b, a
    c, b, a = (float(semi_axes[k]) for k in order)
    return Ellipsoid(a, b, c, np.asarray(directions).T[order], np.asarray(center, dtype=float))


def inertia_ellipsoid_from_moments(
    volume: float, moments: ArrayLike, axes: ArrayLike, centroid: ArrayLike,
) -> Ellipsoid:
    """Inertia-equivalent ellipsoid from volume, principal moments and their axes.

    ``moments`` are the unit-density principal moments I1 <= I2 <= I3 and
    ``axes`` the matching unit eigenvectors as columns, as returned by
    :func:`pyshape.closed_surface_inertia_tensor`.
    """
    if not volume > 0:
        raise ValueError("the inertia-equivalent ellipsoid needs a positive volume")
    i1, i2, i3 = (float(m) for m in moments)
    squares = 5.0 / (2.0 * volume) * np.array([i2 + i3 - i1, i1 + i3 - i2, i1 + i2 - i3])
    scale = max(float(np.max(np.abs(squares))), np.finfo(float).tiny)
    if np.any(squares <= -1e-12 * scale):
        raise ValueError("principal moments violate the triangle inequality")
    semi_axes = np.sqrt(np.clip(squares, 0.0, None))
    return _sorted(semi_axes, np.asarray(axes, dtype=float), centroid)


def inertia_ellipsoid(nodes: ArrayLike, faces: ArrayLike) -> Ellipsoid:
    """Inertia-equivalent ellipsoid of the solid bounded by a closed, outward-oriented surface.

    Use :func:`pyshape.prepare_mesh` first for meshes whose shells may be
    reversed or overlapping.
    """
    from .geometry import closed_surface_inertia_tensor

    points = np.asarray(nodes, dtype=float)
    indices = np.asarray(faces, dtype=int)
    origin = points.mean(axis=0)
    tri = points[indices] - origin
    signed = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
    volume = float(signed.sum())
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("mesh has zero, negative or non-finite enclosed volume")
    centroid = origin + (signed[:, None] * tri.sum(axis=1) / 4.0).sum(axis=0) / volume
    _, moments, axes = closed_surface_inertia_tensor(points, indices, centroid)
    return inertia_ellipsoid_from_moments(volume, moments, axes, centroid)


def vertex_areas(nodes: ArrayLike, faces: ArrayLike) -> np.ndarray:
    """Surface area attributed to each vertex: one third of each adjacent triangle."""
    points = np.asarray(nodes, dtype=float)
    indices = np.asarray(faces, dtype=int)
    tri = points[indices]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = np.zeros(len(points))
    np.add.at(weights, indices.ravel(), np.repeat(areas / 3.0, 3))
    return weights


def fit_ellipsoid(points: ArrayLike, weights: ArrayLike | None = None) -> Ellipsoid:
    """Weighted algebraic least-squares ellipsoid through ``points`` (constraint A + B + C = 3).

    Raises ``ValueError`` if the points do not determine a quadric (e.g. fewer
    than nine points in general position, or coplanar points) or if the
    fitted quadric is not an ellipsoid.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(p),) or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be finite, non-negative and one per point")
    keep = w > 0
    p, w = p[keep], w[keep]
    if len(p) < 9:
        raise ValueError("an ellipsoid fit needs at least nine points with positive weight")
    # The fit is invariant to translation, rotation and uniform scaling; normalise
    # for conditioning and transform back.
    shift = (w[:, None] * p).sum(axis=0) / w.sum()
    q = p - shift
    scale = float(np.sqrt((w * (q ** 2).sum(axis=1)).sum() / w.sum()))
    if not scale > 0:
        raise ValueError("points are all at the same location")
    x, y, z = (q / scale).T
    design = np.column_stack([
        x * x + y * y - 2 * z * z, x * x + z * z - 2 * y * y,
        2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, np.ones_like(x),
    ])
    target = x * x + y * y + z * z
    root = np.sqrt(w / w.sum())
    u, _, rank, singular = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)
    if rank < design.shape[1] or singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("the points do not determine a quadric (too few or degenerate points)")
    v = np.empty(10)
    v[0] = u[0] + u[1] - 1
    v[1] = u[0] - 2 * u[1] - 1
    v[2] = u[1] - 2 * u[0] - 1
    v[3:] = u[2:]
    quadric = np.array([
        [v[0], v[3], v[4], v[6]],
        [v[3], v[1], v[5], v[7]],
        [v[4], v[5], v[2], v[8]],
        [v[6], v[7], v[8], v[9]],
    ])
    try:
        center = np.linalg.solve(-quadric[:3, :3], v[6:9])
    except np.linalg.LinAlgError as exc:
        raise ValueError("the fitted quadric has no centre (not an ellipsoid)") from exc
    translate = np.eye(4)
    translate[3, :3] = center
    centred = translate @ quadric @ translate.T
    if centred[3, 3] == 0:
        raise ValueError("the fitted quadric is degenerate (not an ellipsoid)")
    eigenvalues, eigenvectors = np.linalg.eigh(centred[:3, :3] / -centred[3, 3])
    if not np.all(eigenvalues > 0):
        kinds = {1: "hyperboloid of one sheet", 2: "hyperboloid of two sheets"}
        negative = int(np.sum(eigenvalues <= 0))
        raise ValueError(
            "the fitted quadric is not an ellipsoid"
            + (f" ({kinds[negative]})" if negative in kinds else "")
        )
    semi_axes = scale / np.sqrt(eigenvalues)
    return _sorted(semi_axes, eigenvectors, center * scale + shift)


def fitted_ellipsoid(nodes: ArrayLike, faces: ArrayLike, *, weighting: str = "area") -> Ellipsoid:
    """Least-squares ellipsoid fitted to the surface vertices.

    ``weighting="area"`` (default) weights each vertex by its share of the
    surface area; ``weighting="none"`` uses unweighted vertices as SHAPE does.
    """
    if weighting == "area":
        weights = vertex_areas(nodes, faces)
    elif weighting == "none":
        weights = None
    else:
        raise ValueError("weighting must be 'area' or 'none'")
    points = np.asarray(nodes, dtype=float)
    used = np.unique(np.asarray(faces, dtype=int))
    return fit_ellipsoid(points[used], None if weights is None else weights[used])
