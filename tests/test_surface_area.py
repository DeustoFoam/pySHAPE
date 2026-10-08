import numpy as np
from pyshape import surface_area

def test_unit_square_two_tris():
    nodes = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3]], dtype=int)
    assert np.isclose(surface_area(nodes, faces), 1.0)

def test_regular_tetrahedron_edge1():
    # Regular tetrahedron with edge length 1 has surface area sqrt(3)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3)/2, 0.0],
        [0.5, np.sqrt(3)/6, np.sqrt(2/3)],
    ], dtype=float)
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
    ], dtype=int)
    assert np.isclose(surface_area(nodes, faces), np.sqrt(3), rtol=1e-12, atol=1e-12)

def test_auto_convert_1_based_faces():
    nodes = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
    faces_1_based = np.array([[1,2,3],[1,3,4]], dtype=int)
    assert np.isclose(surface_area(nodes, faces_1_based), 1.0)


def test_volume_centroid_inertia_simple_tet():
    from pyshape import volume_centroid_inertia_tensor
    # Unit right tetra with vertices at origin axes endpoints (volume = 1/6)
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    elements = np.array([[0, 1, 2, 3]], dtype=int)
    vol, cen, Icur, Idiag, axes = volume_centroid_inertia_tensor(nodes, elements, True)
    assert np.isclose(vol, 1/6)
    assert np.allclose(cen, np.array([0.25, 0.25, 0.25]))
    # Basic invariants
    assert Icur.shape == (3,3)
    assert Idiag.shape == (3,3)
    assert axes.shape == (3,3)

def test_volume_centroid_inertia_1_based_elements():
    from pyshape import volume_centroid_inertia_tensor
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    elements_1b = np.array([[1, 2, 3, 4]], dtype=int)
    vol, cen, Icur, Idiag, axes = volume_centroid_inertia_tensor(nodes, elements_1b, False)
    assert np.isclose(vol, 1/6)
    assert np.allclose(cen, np.array([0.25, 0.25, 0.25]))


def test_form_metrics_and_orientation_tensor():
    from pyshape import (
        convexity,
        sphericity_wadell,
        sphericity_krumbein,
        surface_orientation_tensor,
        surface_area,
    )

    # Convexity
    assert np.isclose(convexity(2.0, 2.5), 0.8)

    # Sphericity (Wadell): for unit sphere, phi = 1
    R = 1.0
    V = 4.0/3.0 * np.pi * R**3
    A = 4.0 * np.pi * R**2
    assert np.isclose(sphericity_wadell(V, A), 1.0)

    # Sphericity (Krumbein)
    assert np.isclose(sphericity_krumbein(1.0, 2.0, 4.0), ((2*1)/(4*4))**(1/3))

    # Orientation tensor on a simple square (two triangles) on XY plane
    nodes = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=float)
    faces = np.array([[0,1,2],[0,2,3]], dtype=int)
    C, F, R, eigvals, eigvecs = surface_orientation_tensor(nodes, faces)
    # For a flat square in XY, normals point along +Z, tensor should reflect planarity
    assert eigvals.shape == (3,)
    assert eigvecs.shape == (3,3)


def test_form_wrappers():
    from pyshape import (
        form_functions_1,
        form_functions_2,
        form_parameters_kong_and_fonseca,
        form_parameters_potticary_et_al,
        form_parameters_zingg,
    )
    con, spW = form_functions_1(surface_area=10.0, volume=5.0, volume_convex_hull=6.25)
    assert np.isclose(con, 0.8)
    assert spW > 0

    spK, flP, elP, flK, elK, SIZ, ILZ = form_functions_2(1.0, 2.0, 4.0)
    assert spK > 0
    assert np.isclose(SIZ, 1.0/2.0)
    assert np.isclose(ILZ, 2.0/4.0)

    # Direct parameters for cross-check
    assert np.allclose(form_parameters_kong_and_fonseca(1.0, 2.0, 4.0), ((2.0-1.0)/2.0, (4.0-2.0)/4.0))
    denom = 1.0+2.0+4.0
    assert np.allclose(form_parameters_potticary_et_al(1.0, 2.0, 4.0), (2*(2-1)/denom, (4-2)/denom))
    assert np.allclose(form_parameters_zingg(1.0, 2.0, 4.0), (0.5, 0.5))


def test_load_stl_and_area():
    import os
    from pathlib import Path
    from pyshape import load_stl, surface_area

    # Resolve repository root from this test file location
    repo_root = Path(__file__).resolve().parents[2]
    stl_path = repo_root / "examples" / "Platonic_solids" / "Tetrahedron.stl"
    assert stl_path.exists(), f"Missing STL for test: {stl_path}"

    nodes, faces = load_stl(stl_path)
    assert nodes.shape[1] == 3 and faces.shape[1] == 3
    assert nodes.shape[0] > 0 and faces.shape[0] > 0
    area = surface_area(nodes, faces)
    assert area > 0
