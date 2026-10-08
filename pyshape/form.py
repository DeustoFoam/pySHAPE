import numpy as np
from typing import Tuple

def convexity(volume: float, volume_convex_hull: float) -> float:
    if volume_convex_hull <= 0:
        raise ValueError("volume_convex_hull must be > 0")
    if volume < 0:
        raise ValueError("volume must be >= 0")
    return float(volume / volume_convex_hull)


def sphericity_wadell(volume: float, surface_area: float) -> float:
    if volume < 0:
        raise ValueError("volume must be >= 0")
    if surface_area <= 0:
        raise ValueError("surface_area must be > 0")
    # Equivalent to phi = pi^(1/3) * (6V)^(2/3) / A
    return float(6.0 * volume / (((6.0 * volume / np.pi) ** (1.0 / 3.0)) * surface_area))


def sphericity_krumbein(S: float, I: float, L: float) -> float:
    if L <= 0 or I < 0 or S < 0:
        raise ValueError("dimensions must be non-negative and L > 0")
    return float(((I * S) / (L * L)) ** (1.0 / 3.0))


def surface_orientation_tensor(nodes: np.ndarray, faces: np.ndarray) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """
    Surface orientation tensor (Bagi & Orosz, 2020) and shape indices.

    Parameters
    ----------
    nodes : (N,3) float array
    faces : (M,3) int array (0-based). 1-based faces are auto-detected and converted.

    Returns
    -------
    C, F, R : floats
        Compactness (equancy), Flakiness (platyness), Rodness (elongation)
    eigen_values : (3,) array, descending
    eigen_vectors : (3,3) array, columns correspond to eigen_values order
    """
    nodes = np.asarray(nodes, dtype=float)
    faces = np.asarray(faces, dtype=int)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (M, 3)")

    # Auto-detect 1-based indexing
    if faces.size and faces.min() == 1 and faces.max() == nodes.shape[0]:
        faces = faces - 1
    if faces.size and (faces.min() < 0 or faces.max() >= nodes.shape[0]):
        raise ValueError("face indices out of range for nodes array")

    # Triangle normals and areas
    A = nodes[faces[:, 0]]
    B = nodes[faces[:, 1]]
    Cn = nodes[faces[:, 2]]
    e_AB = B - A
    e_BC = Cn - B
    v = np.cross(e_AB, e_BC)
    areas = 0.5 * np.linalg.norm(v, axis=1)

    # normals normalized; handle zero-area faces safely
    norms = np.linalg.norm(v, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        n = np.divide(v, norms[:, None], out=np.zeros_like(v), where=norms[:, None] > 0)

    total_area = areas.sum()
    if total_area <= 0:
        raise ValueError("total surface area is zero; mesh may be degenerate")

    # Orientation tensor f = (1/A) sum_k A_k n_k n_k^T
    f = np.zeros((3, 3), dtype=float)
    for k in range(n.shape[0]):
        nk = n[k][:, None]
        f += areas[k] * (nk @ nk.T)
    f /= total_area

    # Eigen decomposition and sort descending
    vals, vecs = np.linalg.eig(f)
    order = np.argsort(vals)[::-1]
    eigen_values = np.real(vals[order])
    eigen_vectors = np.real(vecs[:, order])

    f1, f2, f3 = eigen_values
    # Avoid division by zero: f1 should be > 0 for a valid tensor
    if f1 <= 0:
        raise ValueError("largest eigenvalue non-positive; invalid orientation tensor")

    C = float(f3 / f1)
    F = float((f1 - f2) / f1)
    R = float((f2 - f3) / f1)
    return C, F, R, eigen_values, eigen_vectors


def form_functions_1(surface_area: float, volume: float, volume_convex_hull: float) -> Tuple[float, float]:
    """
    Python port of Form_functions_1.m

    Returns
    -------
    convexity, sphericity_wadell
    """
    con = convexity(volume, volume_convex_hull)
    spW = sphericity_wadell(volume, surface_area)
    return con, spW


def form_functions_2(S: float, I: float, L: float) -> Tuple[float, float, float, float, float, float, float]:
    """
    Python port of Form_functions_2.m

    Returns
    -------
    spK, flP, elP, flK, elK, SIZ, ILZ
    """
    spK = sphericity_krumbein(S, I, L)
    flP, elP = form_parameters_potticary_et_al(S, I, L)
    flK, elK = form_parameters_kong_and_fonseca(S, I, L)
    SIZ, ILZ = form_parameters_zingg(S, I, L)
    return spK, flP, elP, flK, elK, SIZ, ILZ


def form_parameters_kong_and_fonseca(S: float, I: float, L: float) -> Tuple[float, float]:
    flatness = (I - S) / I if I != 0 else 0.0
    elongation = (L - I) / L if L != 0 else 0.0
    return float(flatness), float(elongation)


def form_parameters_potticary_et_al(S: float, I: float, L: float) -> Tuple[float, float]:
    denom = (L + I + S)
    if denom == 0:
        return 0.0, 0.0
    flatness = 2.0 * (I - S) / denom
    elongation = (L - I) / denom
    return float(flatness), float(elongation)


def form_parameters_zingg(S: float, I: float, L: float) -> Tuple[float, float]:
    c_over_b = S / I if I != 0 else 0.0
    b_over_a = I / L if L != 0 else 0.0
    return float(c_over_b), float(b_over_a)
