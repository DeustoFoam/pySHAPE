from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyshape import surface_area, surface_orientation_tensor
from util_meshes import make_cube, make_icosahedron, make_uv_sphere, make_ellipsoid

for name, (nodes, faces) in {
    "cube": make_cube(1.0),
    "icosahedron": make_icosahedron(1.0),
    "uv_sphere": make_uv_sphere(1.0, stacks=12, slices=24),
    "ellipsoid": make_ellipsoid(1, 2, 3, stacks=12, slices=24),
}.items():
    A = surface_area(nodes, faces)
    C, F, R, vals, vecs = surface_orientation_tensor(nodes, faces)
    print(f"{name}: area={A:.6f} C={C:.6f} F={F:.6f} R={R:.6f}")
