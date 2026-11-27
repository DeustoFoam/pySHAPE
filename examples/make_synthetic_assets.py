"""Write the synthetic test particles in examples/assets/synthetic/ (binary STL).

    python examples/make_synthetic_assets.py

The files are committed, so tests and users have a fixed collection of
examples; this script documents how they were made and regenerates them
(NumPy only, fixed seeds, so the output is reproducible). Coordinates are
dimensionless (use units="as-is", or declare any unit). See README.md in the
output folder for what each file is for.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "assets" / "synthetic"


# --------------------------------------------------------------------------- meshes
def icosphere(subdivisions: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)):
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(v, float) / np.linalg.norm(v) for v in (
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
        (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1))]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdivisions):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts) * radius + np.asarray(center, float), np.array(faces)


def cuboid(a: float, b: float, c: float, center=(0.0, 0.0, 0.0)):
    nodes = np.array([[x, y, z] for z in (-c / 2, c / 2) for y in (-b / 2, b / 2)
                      for x in (-a / 2, a / 2)]) + np.asarray(center, float)
    faces = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                      [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return nodes, faces


def subdivide_flat(nodes, faces, levels: int):
    """Midpoint subdivision without moving vertices (more vertices, same shape)."""
    verts = [np.asarray(v, float) for v in nodes]
    faces = [tuple(f) for f in faces]
    for _ in range(levels):
        cache, new = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                verts.append((verts[a] + verts[b]) / 2)
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces)


def torus(major: float, minor: float, n_ring: int = 64, n_section: int = 24):
    u = np.arange(n_ring) * 2 * np.pi / n_ring
    v = np.arange(n_section) * 2 * np.pi / n_section
    uu, vv = np.meshgrid(u, v, indexing="ij")
    nodes = np.stack([(major + minor * np.cos(vv)) * np.cos(uu),
                      (major + minor * np.cos(vv)) * np.sin(uu),
                      minor * np.sin(vv)], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(n_ring):
        for j in range(n_section):
            a = i * n_section + j
            b = ((i + 1) % n_ring) * n_section + j
            c = ((i + 1) % n_ring) * n_section + (j + 1) % n_section
            d = i * n_section + (j + 1) % n_section
            faces += [(a, b, c), (a, c, d)]
    return nodes, np.array(faces)


def rotation(a: float, b: float, c: float):
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def combine(*meshes, reverse=()):
    """Concatenate shells; shells whose index is in ``reverse`` are wound the other way."""
    nodes, faces, offset = [], [], 0
    for k, (n, f) in enumerate(meshes):
        nodes.append(n)
        faces.append((f[:, ::-1] if k in reverse else f) + offset)
        offset += len(n)
    return np.vstack(nodes), np.vstack(faces)


def rough_particle(seed: int = 2026, subdivisions: int = 4):
    sphere, faces = icosphere(subdivisions)
    coeffs = np.random.default_rng(seed).uniform(-1, 1, 6)
    x, y, z = sphere.T
    radius = 1 + 0.12 * (coeffs[0] * np.sin(3 * x + coeffs[1]) * np.cos(2 * y)
                         + coeffs[2] * np.cos(4 * z + coeffs[3]) + coeffs[4] * x * y * z
                         + coeffs[5] * np.sin(5 * y))
    return sphere * radius[:, None] * [1.6, 1.0, 0.7], faces


# --------------------------------------------------------------------------- catalogue
def catalogue():
    """Name -> (nodes, faces, description)."""
    s4 = icosphere(4)
    s3 = icosphere(3)
    ellipsoid = (s4[0] * [3.0, 2.0, 1.0]) @ rotation(0.4, 0.9, -0.2).T + [1.0, -2.0, 0.5]
    open_nodes = s3[0] * [2.0, 1.4, 1.0]
    open_faces = s3[1][open_nodes[s3[1]].mean(axis=1)[:, 0] < 1.4]
    tetra = (np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float),
             np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]))
    big_cube = subdivide_flat(*cuboid(2.0, 2.0, 2.0), 3)
    return {
        "sphere": (*s4, "Sphere, radius 1 (5,120 triangles): S = I = L, sphericities close to 1."),
        "ellipsoid_3_2_1": (ellipsoid, s4[1], "Scalene ellipsoid, semi-axes 3, 2, 1, rotated and "
                            "translated: principal dimensions, both ellipsoid methods."),
        "cuboid_4_2_1": (*cuboid(4.0, 2.0, 1.0), "Box 4 x 2 x 1 (12 triangles): exact box "
                         "dimensions; inertia ellipsoid axes = sqrt(5/3) x sides."),
        "cube_subdivided": (*big_cube, "Cube of side 2 with subdivided faces (768 triangles): "
                            "enough vertices for the fitted ellipsoid; non-unique PCA directions."),
        "regular_tetrahedron": (*tetra, "Regular tetrahedron, edge 2*sqrt(2): non-unique PCA "
                                "directions, exact minimum-volume box."),
        "torus_3_1": (*torus(3.0, 1.0), "Ring torus, radii 3 and 1: non-convex, genus 1."),
        "rough_particle": (*rough_particle(), "Irregular, non-convex particle (seeded)."),
        "two_spheres_overlapping": (*combine(icosphere(3), icosphere(3, 0.6, (1.2, 0, 0))),
                                    "Two overlapping spheres (radii 1 and 0.6): a clump "
                                    "analysed as their solid union."),
        "open_ellipsoid": (open_nodes, open_faces, "Ellipsoid with one end cut off (open "
                           "surface): volume-based results unavailable; the preview shows the "
                           "inner side through the opening."),
        "hollow_sphere": (*combine(s3, icosphere(3, 0.5)),
                          "Sphere of radius 1 with a concentric spherical cavity of radius 0.5; "
                          "both shells wound outwards in the file (cavities are found by "
                          "nesting). cavities='keep': porosity 0.125; 'fill': solid sphere."),
        "cube_with_spherical_void": (*combine(big_cube, icosphere(3, 0.5, (0.3, 0.2, -0.1)),
                                              reverse={1}),
                                     "Cube of side 2 with an off-centre spherical cavity "
                                     "(radius 0.5, wound inwards in the file): the material "
                                     "centroid moves away from the void."),
        "sphere_with_two_cavities": (*combine(icosphere(3, 2.0), icosphere(3, 0.5, (-0.9, 0, 0)),
                                              icosphere(3, 0.4, (0.9, 0.3, 0))),
                                     "Sphere of radius 2 with two separate spherical cavities "
                                     "(radii 0.5 and 0.4): n_cavities = 2."),
        "nested_island": (*combine(icosphere(3, 3.0), icosphere(3, 2.0), icosphere(3, 1.0)),
                          "Three concentric spheres (radii 3, 2, 1): with cavities='keep', a "
                          "shell with a cavity containing a solid island (even-odd rule)."),
    }


def write_binary_stl(path: Path, nodes: np.ndarray, faces: np.ndarray, name: str) -> None:
    tri = nodes[faces].astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    header = f"pySHAPE synthetic particle: {name}".encode("ascii")[:80].ljust(80, b" ")
    records = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    records["n"], records["v"] = normals, tri
    with path.open("wb") as stream:
        stream.write(header + struct.pack("<I", len(tri)))
        stream.write(records.tobytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (nodes, faces, description) in catalogue().items():
        write_binary_stl(OUT / f"{name}.stl", np.asarray(nodes, float), np.asarray(faces), name)
        rows.append(f"| `{name}.stl` | {len(faces):,} | {description} |")
    readme = (
        "# Synthetic test particles\n\n"
        "A fixed collection of synthetic particles for tests, examples and trying the "
        "interfaces. Binary STL, dimensionless coordinates. Generated by "
        "`examples/make_synthetic_assets.py` (NumPy only, fixed seeds); rerun it to "
        "regenerate them.\n\n"
        "| File | Triangles | Purpose |\n| --- | ---: | --- |\n" + "\n".join(rows) + "\n\n"
        "Files with nested shells (`hollow_sphere`, `cube_with_spherical_void`, "
        "`sphere_with_two_cavities`, `nested_island`) are solids with cavities only with "
        "`cavities=\"keep\"`; with the default `cavities=\"fill\"` every shell is solid, as "
        "for the spheres of a clump.\n")
    (OUT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"Wrote {len(rows)} files to {OUT}")


if __name__ == "__main__":
    main()
