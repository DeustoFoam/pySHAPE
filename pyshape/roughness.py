import numpy as np
from typing import Tuple, Optional


def _mean_center(z: np.ndarray) -> Tuple[np.ndarray, float]:
    z = np.asarray(z, dtype=float)
    if z.ndim < 1:
        raise ValueError("z must be at least 1-D")
    zm = float(np.mean(z))
    return z, zm


def sq(z: np.ndarray) -> float:
    """Root mean square (RMS) height of a rough surface.

    Parameters
    ----------
    z : array_like
        Elevations on an MxN grid (any shape is accepted; flattened).

    Returns
    -------
    float
        RMS of (z - mean(z)). Matches MATLAB Sq.m behavior (population RMS).
    """
    z, zm = _mean_center(z)
    return float(np.sqrt(np.mean((z - zm) ** 2)))


def sa(z: np.ndarray) -> float:
    """Arithmetical mean height (average absolute deviation from mean).

    Parameters
    ----------
    z : array_like
        Elevations on an MxN grid (any shape is accepted; flattened).

    Returns
    -------
    float
        Mean absolute value of (z - mean(z)).
    """
    z, zm = _mean_center(z)
    return float(np.mean(np.abs(z - zm)))


def sdq(z: np.ndarray, dx: float, dy: float) -> float:
    """Root mean square gradient of a rough surface.

    This follows the intent of MATLAB Sdq.m: use forward differences along
    x (columns) and y (rows), square them, average, and take the square root.

    Note: The original MATLAB file has a likely parenthesis typo in the y-term.
    Here we compute element-wise squares for both terms.

    Parameters
    ----------
    z : (M,N) array_like
        Elevation grid.
    dx, dy : float
        Grid spacings along x (columns) and y (rows), respectively.

    Returns
    -------
    float
        RMS gradient magnitude.
    """
    Z = np.asarray(z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("z must be a 2-D array of shape (M, N)")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")

    M, N = Z.shape
    if M < 2 or N < 2:
        raise ValueError("z must have at least 2x2 samples to compute gradients")

    # Forward differences along columns (x) and rows (y)
    dZdx = np.diff(Z, n=1, axis=1) / dx  # shape (M, N-1)
    dZdy = np.diff(Z, n=1, axis=0) / dy  # shape (M-1, N)

    num = np.sum(dZdx ** 2) + np.sum(dZdy ** 2)
    denom = (M - 1) * (N - 1)
    return float(np.sqrt(num / denom))


def sku(z: np.ndarray, sq_value: Optional[float] = None) -> float:
    """Kurtosis of rough surface height.

    Parameters
    ----------
    z : array_like
        Elevations on an MxN grid.
    sq_value : float, optional
        Precomputed RMS height (Sq). If None, computed internally.

    Returns
    -------
    float
        Kurtosis based on population moments, matching MATLAB Sku.m.
    """
    z, zm = _mean_center(z)
    if sq_value is None:
        sq_value = sq(z)
    if sq_value == 0:
        return float(np.inf)
    return float(np.mean((z - zm) ** 4) / (sq_value ** 4))


def ssk(z: np.ndarray, sq_value: Optional[float] = None) -> float:
    """Skewness of rough surface height.

    Parameters
    ----------
    z : array_like
        Elevations on an MxN grid.
    sq_value : float, optional
        Precomputed RMS height (Sq). If None, computed internally.

    Returns
    -------
    float
        Skewness based on population moments, matching MATLAB Ssk.m.
    """
    z, zm = _mean_center(z)
    if sq_value is None:
        sq_value = sq(z)
    if sq_value == 0:
        return float(np.nan)
    return float(np.mean((z - zm) ** 3) / (sq_value ** 3))


def roughness_functions(z: np.ndarray, dx: float, dy: float) -> Tuple[float, float, float, float, float]:
    """Convenience wrapper mirroring MATLAB Roughness_functions.m.

    Returns (sq, sa, sdq, sku, ssk).
    """
    sq_val = sq(z)
    sa_val = sa(z)
    sdq_val = sdq(z, dx, dy)
    sku_val = sku(z, sq_val)
    ssk_val = ssk(z, sq_val)
    return sq_val, sa_val, sdq_val, sku_val, ssk_val
