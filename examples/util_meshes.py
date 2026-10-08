import numpy as np


def make_cube(side: float = 1.0):
    s = side
    # Unit cube at origin [0, s]^3
    nodes = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom z=0
        [4, 6, 5], [4, 7, 6],  # top z=s
        [0, 4, 5], [0, 5, 1],  # y=0 side
        [1, 5, 6], [1, 6, 2],  # x=s side
        [2, 6, 7], [2, 7, 3],  # y=s side
        [3, 7, 4], [3, 4, 0],  # x=0 side
    ], dtype=int)
    return nodes, faces


def make_icosahedron(radius: float = 1.0):
    # Golden ratio
    phi = (1 + 5 ** 0.5) / 2
    a, b = 1.0, phi
    verts = np.array([
        [-a,  b,  0], [ a,  b,  0], [-a, -b,  0], [ a, -b,  0],
        [ 0, -a,  b], [ 0,  a,  b], [ 0, -a, -b], [ 0,  a, -b],
        [ b,  0, -a], [ b,  0,  a], [-b,  0, -a], [-b,  0,  a],
    ], dtype=float)
    # Normalize to unit radius then scale
    verts /= np.linalg.norm(verts, axis=1)[:, None]
    verts *= radius
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7,10], [0,10,11],
        [1, 5, 9], [5,11, 4], [11,10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4,11], [6, 2,10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    return verts, faces


def make_uv_sphere(radius: float = 1.0, stacks: int = 12, slices: int = 24):
    nodes = []
    for i in range(stacks + 1):
        v = i / stacks
        theta = np.pi * v    # 0..pi
        y = np.cos(theta)
        r = np.sin(theta)
        for j in range(slices + 1):
            u = j / slices
            phi = 2 * np.pi * u
            x = r * np.cos(phi)
            z = r * np.sin(phi)
            nodes.append([x, y, z])
    nodes = np.asarray(nodes, dtype=float) * radius

    def idx(i, j):
        return i * (slices + 1) + j

    faces = []
    for i in range(stacks):
        for j in range(slices):
            a = idx(i, j)
            b = idx(i, j + 1)
            c = idx(i + 1, j)
            d = idx(i + 1, j + 1)
            if i != 0:
                faces.append([a, c, b])
            if i != stacks - 1:
                faces.append([b, c, d])
    faces = np.asarray(faces, dtype=int)
    return nodes, faces


def make_ellipsoid(rx: float, ry: float, rz: float, stacks: int = 12, slices: int = 24):
    s_nodes, s_faces = make_uv_sphere(radius=1.0, stacks=stacks, slices=slices)
    nodes = s_nodes * np.array([rx, ry, rz])[None, :]
    return nodes, s_faces
