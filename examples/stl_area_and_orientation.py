from pathlib import Path
import sys
import numpy as np

# Ensure repo package is importable when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pyshape import load_stl, surface_area, surface_orientation_tensor

here = Path(__file__).resolve().parent
assets = here / "assets"
local_stl = assets / "Tetrahedron_ascii.stl"

# Prefer local asset for standalone usage; fallback to repo STL if present
stl_path = local_stl
if not stl_path.exists():
	repo_root = here.parents[1]
	candidate = repo_root / "examples" / "Platonic_solids" / "Tetrahedron.stl"
	if candidate.exists():
		stl_path = candidate

nodes, faces = load_stl(stl_path)
print("STL:", stl_path)
print("nodes:", nodes.shape, "faces:", faces.shape)

A = surface_area(nodes, faces)
C, F, R, vals, vecs = surface_orientation_tensor(nodes, faces)
print("area:", A)
print("C,F,R:", C, F, R)
