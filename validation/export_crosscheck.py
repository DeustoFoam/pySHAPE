"""Export benchmark meshes for the original MATLAB SHAPE and analyse them with pySHAPE.

Step 1 of the cross-validation against SHAPE (Angelidakis et al., 2021):

    python validation/export_crosscheck.py          # writes _work/*.mat and results/pyshape_crosscheck.json
    matlab -batch "addpath('validation'); shape_crosscheck('<SHAPE>/Matlab', 'validation/_work', 'validation/results/shape_matlab.json')"
    python validation/compare_crosscheck.py         # writes results/crosscheck.csv

Each mesh is exported after pySHAPE's own preparation (outward orientation
and, for multi-sphere particles, the solid union), so both programs analyse
exactly the same closed triangulated surface.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meshes as M  # noqa: E402
from pyshape import analyze_mesh  # noqa: E402
from pyshape.analysis import prepare_mesh  # noqa: E402

WORK = HERE / "_work"
RESULTS = HERE / "results"


def benchmark_set():
    for name in M.PLATONIC:
        nodes, faces, _ = M.platonic(name, subdivisions=3)
        yield name, nodes, faces
    for name, radii in {"sphere": (1, 1, 1), "prolate": (2, 1, 1), "oblate": (2, 2, 1),
                        "scalene": (3, 2, 1)}.items():
        yield name, *M.ellipsoid(radii, 5)
    yield "torus", *M.torus(3.0, 1.0)
    for name, count, ratio, overlap in (("satellites_1", 1, 0.3, 0.5),
                                        ("satellites_3", 3, 0.3, 0.5),
                                        ("satellites_6", 6, 0.2, 0.3)):
        nodes, faces, _, _ = M.satellite_particle(count, ratio, overlap, seed=11, subdivisions=4)
        yield name, nodes, faces
    for seed in range(20):
        roughness = 0.15 + 0.3 * seed / 19  # from convex-like to markedly concave
        yield f"random_{seed:02d}", *M.random_particle(seed, subdivisions=4, roughness=roughness)


def main() -> None:
    WORK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    results = {}
    for name, nodes, faces in benchmark_set():
        nodes, faces = prepare_mesh(nodes, faces)
        savemat(WORK / f"{name}.mat", {"nodes": nodes, "faces": (faces + 1).astype(float)})
        entry = {"n_vertices": len(nodes), "n_faces": len(faces)}
        for method in ("pca", "min_volume"):
            started = time.perf_counter()
            entry[method] = analyze_mesh(nodes, faces, axes=method).values
            entry[f"seconds_{method}"] = time.perf_counter() - started
        results[name] = entry
        print(f"{name:14s} {len(faces):6d} faces  pca {entry['seconds_pca']:.3f} s  "
              f"min_volume {entry['seconds_min_volume']:.3f} s")
    (RESULTS / "pyshape_crosscheck.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
