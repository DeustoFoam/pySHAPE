import numpy as np
from typing import Iterable, Tuple

def surface_area(nodes: np.ndarray, faces: np.ndarray) -> float:
    """
    Compute the surface area of a 3-D triangular surface mesh.

    Parameters
    ----------
    nodes : (N, 3) array_like of float
        Vertex coordinates.
    faces : (M, 3) array_like of int
        Triangle vertex indices (0-based). If 1-based indices are detected,
        they will be converted automatically.

    Returns
    -------
    float
        Total surface area.
    """
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    # Auto-detect 1-based indexing (common in MATLAB) and convert to 0-based.
    if faces.size and faces.min() == 1 and faces.max() == nodes.shape[0]:
        faces = faces - 1

    if faces.size and (faces.min() < 0 or faces.max() >= nodes.shape[0]):
        raise ValueError("face indices out of range for nodes array")

    v1 = nodes[faces[:, 1]] - nodes[faces[:, 0]]
    v2 = nodes[faces[:, 2]] - nodes[faces[:, 0]]

    cross_prod = np.cross(v1, v2)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    return float(areas.sum())


def volume_centroid_inertia_tensor(
    nodes: np.ndarray,
    elements: np.ndarray,
    calculate_inertia: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute volume, centroid, current inertia tensor (about centroid),
    principal inertia tensor, and principal orientations for a tetrahedral mesh.

    Parameters
    ----------
    nodes : (N, 3) array_like of float
        Vertex coordinates.
    elements : (M, 4) array_like of int
        Tetra connectivity (0-based). If 1-based indices are detected,
        they will be converted automatically.
    calculate_inertia : bool, default True
        Whether to compute inertia tensors and principal orientations.

    Returns
    -------
    volume : float
        Total mesh volume (sum of tetra volumes).
    centroid : (3,) ndarray
        Volume-weighted centroid of the mesh.
    current_inertia_tensor : (3,3) ndarray
        Inertia tensor about the centroid (symmetric). Off-diagonals are negative
        products of inertia consistent with the standard inertia tensor definition.
    inertia_tensor : (3,3) ndarray
        Diagonal matrix of principal moments (eigenvalues, ascending).
    principal_orientations : (3,3) ndarray
        Columns are the unit eigenvectors (principal axes), matching the eigenvalues order.
    """
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (M, 4)")

    # Auto-detect 1-based indexing and convert to 0-based.
    if elements.size and elements.min() == 1 and elements.max() == nodes.shape[0]:
        elements = elements - 1

    if elements.size and (elements.min() < 0 or elements.max() >= nodes.shape[0]):
        raise ValueError("element indices out of range for nodes array")

    # Gather vertices per tetra
    a = nodes[elements[:, 0]]
    b = nodes[elements[:, 1]]
    c = nodes[elements[:, 2]]
    d = nodes[elements[:, 3]]

    # Volumes via scalar triple product: |(a-d) . ((b-d) x (c-d))| / 6
    ad = a - d
    bd = b - d
    cd = c - d
    v = np.abs(np.einsum('ij,ij->i', np.cross(bd, cd), ad)) / 6.0
    volume = float(v.sum())

    if volume <= 0.0:
        raise ValueError("Total volume is zero or negative; check elements or degeneracy.")

    # Centroids of tets and volume-weighted sum
    tet_centroids = (a + b + c + d) / 4.0
    centroid = (v[:, None] * tet_centroids).sum(axis=0) / volume

    if not calculate_inertia:
        Z = np.zeros((3, 3), dtype=float)
        return volume, centroid, Z, Z, Z

    # Shift coordinates so centroid is at the origin
    nodes_c = nodes - centroid[None, :]
    a = nodes_c[elements[:, 0]]
    b = nodes_c[elements[:, 1]]
    c = nodes_c[elements[:, 2]]
    d = nodes_c[elements[:, 3]]

    # Per-tetra coordinate arrays
    x = np.stack([a[:, 0], b[:, 0], c[:, 0], d[:, 0]], axis=1)
    y = np.stack([a[:, 1], b[:, 1], c[:, 1], d[:, 1]], axis=1)
    z = np.stack([a[:, 2], b[:, 2], c[:, 2], d[:, 2]], axis=1)

    Sx = x.sum(axis=1)
    Sy = y.sum(axis=1)
    Sz = z.sum(axis=1)
    Sxx = (x * x).sum(axis=1)
    Syy = (y * y).sum(axis=1)
    Szz = (z * z).sum(axis=1)
    Sxy = (x * y).sum(axis=1)
    Sxz = (x * z).sum(axis=1)
    Syz = (y * z).sum(axis=1)

    # Tonon (2005)-style discrete tetra integrals, vectorized
    coef = v / 20.0  # since 6*v/120 = v/20; for Ixx/Iyy/Izz also matches v/20 using identities

    Ixx_t = coef * (Sy * Sy + Syy + Sz * Sz + Szz)
    Iyy_t = coef * (Sx * Sx + Sxx + Sz * Sz + Szz)
    Izz_t = coef * (Sx * Sx + Sxx + Sy * Sy + Syy)

    Ixy_t = coef * (Sx * Sy + Sxy)
    Ixz_t = coef * (Sx * Sz + Sxz)
    Iyz_t = coef * (Sy * Sz + Syz)

    Ixx = float(Ixx_t.sum())
    Iyy = float(Iyy_t.sum())
    Izz = float(Izz_t.sum())
    Ixy = float(Ixy_t.sum())
    Ixz = float(Ixz_t.sum())
    Iyz = float(Iyz_t.sum())

    current_inertia = np.array(
        [
            [Ixx, -Ixy, -Ixz],
            [-Ixy, Iyy, -Iyz],
            [-Ixz, -Iyz, Izz],
        ],
        dtype=float,
    )

    # Principal moments and axes
    evals, evecs = np.linalg.eigh(current_inertia)
    inertia_diag = np.diag(evals)
    return volume, centroid, current_inertia, inertia_diag, evecs
