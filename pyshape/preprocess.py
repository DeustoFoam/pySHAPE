"""Mesh preparation: validation, orientation of closed shells and solid union.

This module sits between :mod:`pyshape.io`, which reads files, and
:mod:`pyshape.analysis`, which computes the descriptors. Its functions are also
available from :mod:`pyshape.analysis` for compatibility.
"""
from __future__ import annotations

from importlib import import_module
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from .geometry import surface_area

__all__ = ["prepare_mesh", "validate_mesh"]


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
    """Validate a triangle mesh, orient its closed shells outwards and merge them.

    This is the first step of :func:`pyshape.analysis.compute_metrics`, and
    therefore of :func:`pyshape.analyze_mesh` and :func:`pyshape.analyze_file`;
    call it directly to obtain the surface that is actually analysed (e.g. to
    save or inspect it). It is also available as ``pyshape.prepare_mesh`` and
    ``pyshape.analysis.prepare_mesh``.

    Steps:

    1. The arrays are validated (see :func:`validate_mesh`); one-based faces
       are recognised when unambiguous.
    2. Edges are compared as unordered vertex pairs. If any edge belongs to one
       triangle (a boundary) or to more than two (a non-manifold edge), the mesh
       is treated as open and returned unchanged; volume-based descriptors are
       then reported as unavailable by
       :func:`pyshape.analysis.compute_metrics`.
    3. Shells are the sets of triangles connected through shared edges. In each
       shell, the winding of the first triangle (in input order) is kept and
       propagated across shared edges: a neighbour that traverses a shared edge
       in the same direction as the triangle it is reached from is reversed.
    4. Each shell is reversed as a whole if its signed volume is negative, so
       that its normals point outwards.
    5. Two or more closed shells are treated as solid components (not cavities)
       and merged by a Boolean union with Manifold, so that overlapping material
       is counted once and internal surfaces are removed.

    Parameters
    ----------
    nodes : (N, 3) array_like of float
        Vertex coordinates.
    faces : (M, 3) array_like of int
        Triangle vertex indices, zero- or one-based.

    Returns
    -------
    nodes : (N', 3) ndarray of float
        Vertex coordinates; new arrays when several shells are merged.
    faces : (M', 3) ndarray of int
        Zero-based triangles, oriented outwards for closed meshes.

    Raises
    ------
    ValueError
        If the arrays are invalid, a closed surface is not orientable (a
        triangle would have to be both kept and reversed), or the union fails.
    ImportError
        If several closed shells are present and the ``mesh`` extra
        (trimesh and manifold3d) is not installed.

    Notes
    -----
    Input arrays are never modified. Meshes are not repaired or filled. Two
    cases are accepted without a warning: closed shells that touch at a single
    vertex (only edges are checked) are oriented separately and merged, and the
    outward direction is taken from the sign of the signed volume without a
    tolerance, so very thin shells lose accuracy as their volume approaches
    round-off; :func:`pyshape.analysis.compute_metrics` reports the volume as
    unavailable only when it is exactly zero.

    Examples
    --------
    >>> import numpy as np
    >>> from pyshape import prepare_mesh
    >>> nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    >>> inward = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]])
    >>> oriented_nodes, oriented_faces = prepare_mesh(nodes, inward)
    >>> oriented_faces.tolist()
    [[2, 1, 0], [1, 3, 0], [2, 3, 1], [0, 3, 2]]
    """
    nodes, faces = validate_mesh(nodes, faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, counts = np.unique(
        np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts != 2):
        return nodes, faces

    pairs = np.argsort(inverse, kind="stable").reshape(-1, 2)
    components = _consistent_components(len(faces), edges, pairs)
    if components is not None:
        return _orient_and_union(nodes, faces, np.zeros(len(faces), dtype=np.int8), components)

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in faces]
    for first, second in pairs:
        i, j = int(first % len(faces)), int(second % len(faces))
        same_direction = bool(np.array_equal(edges[first], edges[second]))
        adjacency[i].append((j, same_direction))
        adjacency[j].append((i, same_direction))

    flips = np.full(len(faces), -1, dtype=np.int8)
    components = []
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
    return _orient_and_union(nodes, faces, flips, components)


def _consistent_components(
    n_faces: int, edges: np.ndarray, pairs: np.ndarray,
) -> Optional[list[np.ndarray]]:
    """Edge-connected face components of an already consistently wound mesh.

    Fast path for the common case, using SciPy when available. Returns None
    when any shared edge is traversed twice in the same direction (faces need
    flipping) or SciPy is missing; the caller then uses the general search.
    """
    if np.any(np.all(edges[pairs[:, 0]] == edges[pairs[:, 1]], axis=1)):
        return None
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
    except ModuleNotFoundError:
        return None
    first, second = pairs[:, 0] % n_faces, pairs[:, 1] % n_faces
    graph = coo_matrix((np.ones(len(first)), (first, second)), shape=(n_faces, n_faces))
    count, labels = connected_components(graph, directed=False)
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.cumsum(np.bincount(labels, minlength=count))[:-1])


def _orient_and_union(
    nodes: np.ndarray, faces: np.ndarray, flips: np.ndarray, components: list,
) -> tuple[np.ndarray, np.ndarray]:
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
