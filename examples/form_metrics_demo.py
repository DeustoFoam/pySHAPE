from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyshape import (
    convexity,
    sphericity_wadell,
    sphericity_krumbein,
    surface_orientation_tensor,
    form_functions_1,
    form_functions_2,
)

# Wadell sphericity of unit sphere (V=4/3 pi, A=4 pi) -> 1
V = 4/3 * np.pi
A = 4 * np.pi
print("wadell:", sphericity_wadell(V, A))

# Krumbein sphericity
print("krumbein:", sphericity_krumbein(S=1.0, I=2.0, L=4.0))

# Convexity example
print("convexity:", convexity(2.0, 2.5))

# Orientation tensor for a square in XY plane
nodes = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float)
faces = np.array([[0,1,2],[0,2,3]], int)
C, F, R, vals, vecs = surface_orientation_tensor(nodes, faces)
print("C,F,R:", C, F, R)

# High-level wrappers
print("Form_functions_1:", form_functions_1(surface_area=1.0, volume=0.5, volume_convex_hull=0.6))
print("Form_functions_2:", form_functions_2(S=1.0, I=2.0, L=4.0))
