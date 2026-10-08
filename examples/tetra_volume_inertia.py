from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyshape import volume_centroid_inertia_tensor

# Regular tetrahedron with unit edge
nodes = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3)/2, 0.0],
    [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
], dtype=float)

elements = np.array([[0,1,2,3]], dtype=int)

vol, cen, Icur, Idiag, axes = volume_centroid_inertia_tensor(nodes, elements)
print("volume:", vol)
print("centroid:", cen)
print("inertia (about centroid):\n", Icur)
print("principal moments:\n", np.diag(Idiag))
print("principal axes (columns):\n", axes)
