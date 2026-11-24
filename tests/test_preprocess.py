"""Tests of pyshape.preprocess: validation, orientation and solid union."""
import sys

import numpy as np
import pytest

from pyshape import preprocess
from pyshape.analysis import mesh_volume_centroid
from pyshape.preprocess import prepare_mesh, validate_mesh


def test_old_import_paths_still_work():
    import pyshape
    from pyshape import analysis

    assert pyshape.prepare_mesh is prepare_mesh
    assert analysis.prepare_mesh is prepare_mesh
    assert analysis.validate_mesh is validate_mesh
    assert analysis._volume_error is preprocess._volume_error
    assert "prepare_mesh" in pyshape.__all__


def test_validate_mesh_converts_one_based_faces(tetra_mesh):
    nodes, faces = tetra_mesh
    _, zero_based = validate_mesh(nodes, faces + 1)
    np.testing.assert_array_equal(zero_based, faces)


def test_union_of_overlapping_shells(cube_mesh):
    pytest.importorskip("manifold3d")
    nodes, faces = cube_mesh
    two = (np.vstack((nodes, nodes + [0.5, 0, 0])), np.vstack((faces, faces[:, ::-1] + len(nodes))))
    out_nodes, out_faces = prepare_mesh(*two)
    volume, centroid = mesh_volume_centroid(out_nodes, out_faces)
    assert volume == pytest.approx(1.5, rel=1e-6)
    assert centroid[0] == pytest.approx(0.75, rel=1e-6)
    assert preprocess._volume_error(out_faces) is None


def test_fast_and_general_orientation_paths_agree(cube_mesh, monkeypatch):
    nodes, faces = cube_mesh
    two_shells = (np.vstack((nodes, nodes + 5)), np.vstack((faces, faces + len(nodes))))
    for mesh in ((nodes, faces), (nodes, faces[:, ::-1]), two_shells):
        expected = preprocess.prepare_mesh(*mesh) if mesh is not two_shells else None
        fast = preprocess._consistent_components(
            len(mesh[1]), *_edge_pairs(mesh[1]))
        assert fast is not None
        if expected is None:
            assert len(fast) == 2
            continue
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "scipy.sparse", None)
            general = preprocess.prepare_mesh(*mesh)
        np.testing.assert_array_equal(expected[1], general[1])


def _edge_pairs(faces):
    faces = np.asarray(faces)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
    return edges, np.argsort(inverse.ravel(), kind="stable").reshape(-1, 2)


def test_prepare_mesh_is_public_and_orients_outwards(tetra_mesh):
    nodes, faces = tetra_mesh
    inward = faces[:, ::-1].copy()
    before = inward.copy()
    out_nodes, out_faces = prepare_mesh(nodes, inward)
    assert np.array_equal(inward, before)
    volume, _ = mesh_volume_centroid(out_nodes, out_faces)
    assert volume == pytest.approx(1 / 6)


def test_prepare_mesh_returns_open_mesh_unchanged(tetra_mesh):
    nodes, faces = tetra_mesh
    _, out_faces = prepare_mesh(nodes, faces[:3])
    assert np.array_equal(out_faces, faces[:3])


def test_non_orientable_surface_is_rejected():
    # Six-vertex triangulation of the real projective plane: every edge is
    # shared by exactly two triangles, but no consistent winding exists.
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5], [0, 5, 1],
        [1, 2, 4], [2, 3, 5], [3, 4, 1], [4, 5, 2], [5, 1, 3],
    ])
    nodes = np.random.default_rng(0).normal(size=(6, 3))
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    assert set(np.unique(edges, axis=0, return_counts=True)[1]) == {2}
    with pytest.raises(ValueError, match="not orientable"):
        prepare_mesh(nodes, faces)
