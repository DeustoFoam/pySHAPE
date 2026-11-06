"""Benchmark geometries and their exact (analytical) reference values.

Every mesh used in the validation of pySHAPE is generated here, so that the
benchmarks are reproducible without external data. Requires ``trimesh`` and
``scipy`` (``pip install "pyshape[mesh,analysis]"``).
"""
from __future__ import annotations

from math import pi, sqrt

import numpy as np
import trimesh
from numpy.polynomial import legendre
from scipy import special
from scipy.spatial import ConvexHull

PHI = (1 + sqrt(5)) / 2


# --------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------
def icosphere(subdivisions: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    """Geodesic sphere; subdivision 5 has 10,242 vertices and 20,480 faces."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(mesh.vertices) + np.asarray(center, float), np.asarray(mesh.faces)


def ellipsoid(radii, subdivisions: int):
    """Ellipsoid obtained by affine scaling of a geodesic sphere."""
    nodes, faces = icosphere(subdivisions)
    return nodes * np.asarray(radii, float), faces


def torus(major: float, minor: float, n_ring: int = 300, n_section: int = 100):
    """Ring torus sampled on a regular (ring x section) grid.

    The default 300 x 100 grid (30,000 vertices, 60,000 faces) is the torus of
    Table 3 in Angelidakis et al. (2021), the original SHAPE paper.
    """
    u = 2 * pi * np.arange(n_ring) / n_ring
    v = 2 * pi * np.arange(n_section) / n_section
    uu, vv = np.meshgrid(u, v, indexing="ij")
    nodes = np.column_stack((
        ((major + minor * np.cos(vv)) * np.cos(uu)).ravel(),
        ((major + minor * np.cos(vv)) * np.sin(uu)).ravel(),
        (minor * np.sin(vv)).ravel(),
    ))
    i, j = np.meshgrid(np.arange(n_ring), np.arange(n_section), indexing="ij")
    a = (i * n_section + j).ravel()
    b = (((i + 1) % n_ring) * n_section + j).ravel()
    c = (((i + 1) % n_ring) * n_section + (j + 1) % n_section).ravel()
    d = (i * n_section + (j + 1) % n_section).ravel()
    faces = np.vstack((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    return nodes, faces


def platonic(name: str, subdivisions: int = 0):
    """Platonic solid whose axis-aligned bounding box is the unit cube.

    This matches the convention of the Platonic-solid meshes distributed with
    SHAPE. ``subdivisions`` splits each triangle into four coplanar triangles,
    producing many coplanar and collinear vertices (a robustness test).
    Returns nodes, faces and the edge length.
    """
    if name == "tetrahedron":
        points, edge = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / 2, sqrt(2)
    elif name == "hexahedron":
        points = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        edge = 1.0
    elif name == "octahedron":
        points, edge = np.vstack((np.eye(3), -np.eye(3))) / 2, sqrt(2) / 2
    elif name == "icosahedron":
        base = [[0, s1, s2 * PHI] for s1 in (-1, 1) for s2 in (-1, 1)]
        points = np.array([np.roll(p, k) for p in base for k in range(3)]) / (2 * PHI)
        edge = 1 / PHI
    elif name == "dodecahedron":
        cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        ring = [np.roll([0, s1 / PHI, s2 * PHI], k)
                for s1 in (-1, 1) for s2 in (-1, 1) for k in range(3)]
        points = np.array(cube + ring) / (2 * PHI)
        edge = 1 / PHI ** 2
    else:
        raise ValueError(name)
    hull = ConvexHull(points)
    mesh = trimesh.Trimesh(points, hull.simplices, process=True)
    trimesh.repair.fix_normals(mesh)
    for _ in range(subdivisions):
        mesh = mesh.subdivide()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces), edge


PLATONIC = ("tetrahedron", "hexahedron", "octahedron", "dodecahedron", "icosahedron")


def cuboid(a: float, b: float, c: float):
    mesh = trimesh.creation.box(extents=(a, b, c))
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def sphere_cluster(centers, radii, subdivisions: int = 5):
    """Concatenated, overlapping sphere shells (a multi-sphere particle)."""
    nodes, faces, offset = [], [], 0
    for center, radius in zip(centers, radii):
        v, f = icosphere(subdivisions, radius, center)
        nodes.append(v)
        faces.append(f + offset)
        offset += len(v)
    return np.vstack(nodes), np.vstack(faces)


def satellite_particle(n_satellites: int, size_ratio: float, overlap: float = 0.5,
                       radius: float = 1.0, seed: int = 0, subdivisions: int = 5,
                       min_separation: float = 1e-3):
    """Spherical particle with satellites, as in gas-atomised metal powders.

    Satellites of radius ``size_ratio * radius`` are placed in random,
    mutually non-overlapping directions; ``overlap`` is the depth of
    penetration into the main sphere as a fraction of the satellite diameter.
    Returns nodes, faces, centers and radii.
    """
    rng = np.random.default_rng(seed)
    rho = size_ratio * radius
    distance = radius + rho - 2 * rho * overlap
    centers, radii = [np.zeros(3)], [radius]
    while len(centers) < n_satellites + 1:
        direction = rng.normal(size=3)
        candidate = distance * direction / np.linalg.norm(direction)
        if all(np.linalg.norm(candidate - c) > 2 * rho * (1 + min_separation) for c in centers[1:]):
            centers.append(candidate)
            radii.append(rho)
    nodes, faces = sphere_cluster(centers, radii, subdivisions)
    return nodes, faces, np.array(centers), np.array(radii)


def random_particle(seed: int, subdivisions: int = 5, degree: int = 10, roughness: float = 0.25,
                    decay: float = 1.6, aspect=None):
    """Random star-shaped particle from a band-limited radial function.

    r(u) = 1 + sum_{l=2..degree} sum_{k=1..2l+1} c_lk P_l(u . d_lk), with
    random unit vectors d_lk and c_lk ~ N(0, (roughness l^-decay)^2): a sum of
    2l+1 randomly oriented zonal harmonics spans spherical harmonics of degree
    l. The radial surface is then stretched by ``aspect`` (long, intermediate,
    short ratios; random if None) and randomly rotated. Larger ``roughness``
    produces concave particles.
    """
    rng = np.random.default_rng(seed)
    nodes, faces = icosphere(subdivisions)
    directions = nodes / np.linalg.norm(nodes, axis=1)[:, None]
    r = np.ones(len(nodes))
    for l in range(2, degree + 1):
        coefficients = np.zeros(l + 1)
        coefficients[l] = 1.0
        for _ in range(2 * l + 1):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            r += rng.normal() * roughness * l ** -decay * legendre.legval(directions @ axis, coefficients)
    r = np.maximum(r, 0.15)
    if aspect is None:
        b_over_a, c_over_b = rng.uniform(0.35, 1.0, size=2)
        aspect = (1.0, b_over_a, b_over_a * c_over_b)
    nodes = directions * r[:, None] * np.asarray(aspect)
    return nodes @ random_rotation(rng).T, faces


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


# --------------------------------------------------------------------------
# Exact reference values
# --------------------------------------------------------------------------
def wadell(volume: float, area: float) -> float:
    return (36 * pi * volume ** 2) ** (1 / 3) / area


def platonic_exact(name: str, edge: float) -> dict[str, float]:
    a = edge
    area, volume = {
        "tetrahedron": (sqrt(3) * a ** 2, a ** 3 / (6 * sqrt(2))),
        "hexahedron": (6 * a ** 2, a ** 3),
        "octahedron": (2 * sqrt(3) * a ** 2, sqrt(2) / 3 * a ** 3),
        "dodecahedron": (3 * sqrt(25 + 10 * sqrt(5)) * a ** 2, (15 + 7 * sqrt(5)) / 4 * a ** 3),
        "icosahedron": (5 * sqrt(3) * a ** 2, 5 * (3 + sqrt(5)) / 12 * a ** 3),
    }[name]
    return {"surface_area": area, "volume": volume, "sphericity_wadell": wadell(volume, area),
            "convexity": 1.0, "orientation_C": 1.0, "orientation_F": 0.0, "orientation_R": 0.0}


def ellipsoid_area(a: float, b: float, c: float) -> float:
    """Exact surface area via Legendre's incomplete elliptic integrals."""
    a, b, c = sorted((a, b, c), reverse=True)
    if np.isclose(a, c):
        return 4 * pi * a ** 2
    phi = np.arccos(c / a)
    m = (a ** 2 * (b ** 2 - c ** 2)) / (b ** 2 * (a ** 2 - c ** 2))
    return 2 * pi * c ** 2 + 2 * pi * a * b / np.sin(phi) * (
        special.ellipeinc(phi, m) * np.sin(phi) ** 2 + special.ellipkinc(phi, m) * np.cos(phi) ** 2
    )


def ellipsoid_orientation_tensor(a: float, b: float, c: float, n: int = 400) -> np.ndarray:
    """Eigenvalues (descending) of the surface orientation tensor of an ellipsoid.

    f = (1/A) int n n^T dA, by Gauss-Legendre quadrature in cos(theta) and the
    (spectrally accurate) periodic trapezoidal rule in the azimuth.
    """
    t, w = np.polynomial.legendre.leggauss(n)
    ph = 2 * pi * np.arange(2 * n) / (2 * n)
    T, P = np.meshgrid(t, ph, indexing="ij")
    W = np.repeat(w[:, None], len(ph), axis=1) * (2 * pi / len(ph))
    s = np.sqrt(1 - T ** 2)
    g = np.stack((s * np.cos(P) / a, s * np.sin(P) / b, T / c), axis=-1)
    norm = np.linalg.norm(g, axis=-1)
    dA = a * b * c * norm * W
    f = np.einsum("ij,ijk,ijl->kl", dA / norm ** 2, g, g) / dA.sum()
    return np.sort(np.linalg.eigvalsh(f))[::-1]


def ellipsoid_exact(a: float, b: float, c: float) -> dict[str, float]:
    volume = 4 / 3 * pi * a * b * c
    area = ellipsoid_area(a, b, c)
    f1, f2, f3 = ellipsoid_orientation_tensor(a, b, c)
    S, I, L = sorted((2 * a, 2 * b, 2 * c))
    moments = sorted(volume / 5 * np.array([b * b + c * c, a * a + c * c, a * a + b * b]))
    return {"surface_area": area, "volume": volume, "sphericity_wadell": wadell(volume, area),
            "convexity": 1.0, "orientation_C": f3 / f1, "orientation_F": (f1 - f2) / f1,
            "orientation_R": (f2 - f3) / f1, "axis_S": S, "axis_I": I, "axis_L": L,
            "inertia_I1": moments[0], "inertia_I2": moments[1], "inertia_I3": moments[2]}


def torus_exact(major: float, minor: float) -> dict[str, float]:
    R, r = major, minor
    volume = 2 * pi ** 2 * R * r ** 2
    area = 4 * pi ** 2 * R * r
    # Convex hull: revolved stadium = cylinder of radius R, height 2r, plus the
    # outer half-disc revolved about the axis (Pappus).
    hull = 2 * pi * R ** 2 * r + pi ** 2 * R * r ** 2 + 4 / 3 * pi * r ** 3
    return {"surface_area": area, "volume": volume, "sphericity_wadell": wadell(volume, area),
            "convexity": volume / hull,
            "inertia_I1": volume * (R ** 2 / 2 + 5 * r ** 2 / 8),
            "inertia_I2": volume * (R ** 2 / 2 + 5 * r ** 2 / 8),
            "inertia_I3": volume * (R ** 2 + 3 * r ** 2 / 4)}


def sphere_union_exact(centers, radii) -> dict[str, float]:
    """Volume and area of a main sphere (index 0) with satellites that overlap
    the main sphere but not each other (lens formulas)."""
    centers, radii = np.asarray(centers, float), np.asarray(radii, float)
    R = radii[0]
    volume = 4 / 3 * pi * R ** 3
    area = 4 * pi * R ** 2
    for center, rho in zip(centers[1:], radii[1:]):
        d = np.linalg.norm(center - centers[0])
        # Lens (intersection) volume of two spheres.
        lens = pi * (R + rho - d) ** 2 * (d ** 2 + 2 * d * rho - 3 * rho ** 2 + 2 * d * R
                                         + 6 * rho * R - 3 * R ** 2) / (12 * d)
        x = (d ** 2 - rho ** 2 + R ** 2) / (2 * d)  # plane of intersection from main center
        cap_main = 2 * pi * R * (R - x)              # main-sphere area inside satellite
        cap_sat = 2 * pi * rho * (rho - (d - x))     # satellite area inside main sphere
        volume += 4 / 3 * pi * rho ** 3 - lens
        area += 4 * pi * rho ** 2 - cap_main - cap_sat
    return {"volume": volume, "surface_area": area, "sphericity_wadell": wadell(volume, area)}


def angelidakis_indices(S: float, I: float, L: float) -> tuple[float, float, float]:
    """Flatness, elongation, compactness of Angelidakis et al. (2022), Eq. (5)."""
    a, b, c = L, I, S
    return (b * b / (a * c + b * b) - c / (a + c), a * c / (a * c + b * b) - c / (a + c), 2 * c / (a + c))
