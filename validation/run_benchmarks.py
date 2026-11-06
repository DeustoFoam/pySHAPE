"""Run the pySHAPE validation benchmarks and write results/*.csv.

Usage (from the repository root, with ``pip install -e ".[mesh,analysis]"``):

    python validation/run_benchmarks.py            # all benchmarks
    python validation/run_benchmarks.py torus platonic

Benchmarks
----------
platonic     Exact area, volume, sphericity, convexity and orientation tensor of
             the five Platonic solids; rotation sensitivity of PCA vs
             minimum-volume boxes.
convergence  Tessellation convergence for a sphere and three ellipsoids.
torus        Reproduction of Table 3 of the original SHAPE paper.
union        Multi-sphere (satellite) particles against exact lens formulas.
indices      Worked cuboid examples of Angelidakis et al. (2022).
invariance   Rigid motion, scaling, re-indexing and winding invariance.
performance  Run time against mesh size.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import meshes as M  # noqa: E402
from pyshape import analyze_mesh, min_volume_box  # noqa: E402

RESULTS = HERE / "results"
DIMENSIONLESS = [
    "sphericity_wadell", "sphericity_krumbein", "convexity",
    "orientation_C", "orientation_F", "orientation_R",
    "potticary_flatness", "potticary_elongation", "kongfonseca_flatness",
    "kongfonseca_elongation", "angelidakis_flatness", "angelidakis_elongation",
    "zingg_SI", "zingg_IL",
]


def write(name: str, rows: list[dict]) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(HERE.parent)} ({len(rows)} rows)")


def rel(value: float, exact: float) -> float:
    return abs(value - exact) / abs(exact) if exact != 0 else abs(value - exact)


def platonic() -> None:
    rows, boxes = [], []
    rng = np.random.default_rng(2026)
    rotations = [M.random_rotation(rng) for _ in range(50)]
    for name in M.PLATONIC:
        nodes, faces, edge = M.platonic(name, subdivisions=3)
        exact = M.platonic_exact(name, edge)
        values = analyze_mesh(nodes, faces).values
        for key, ref in exact.items():
            rows.append({"solid": name, "n_vertices": len(nodes), "n_faces": len(faces),
                         "metric": key, "pyshape": values[key], "exact": ref,
                         "error": rel(values[key], ref)})
        for method in ("pca", "min_volume"):
            stats = np.array([[analyze_mesh(nodes @ R.T, faces, axes=method).values[k]
                               for k in ("axis_S", "axis_I", "axis_L", "sphericity_krumbein",
                                         "angelidakis_flatness", "angelidakis_elongation")]
                              for R in rotations])
            for j, key in enumerate(("axis_S", "axis_I", "axis_L", "sphericity_krumbein",
                                     "angelidakis_flatness", "angelidakis_elongation")):
                boxes.append({"solid": name, "axes": method, "metric": key,
                              "min": stats[:, j].min(), "max": stats[:, j].max(),
                              "spread": np.ptp(stats[:, j])})
    write("platonic_exact", rows)
    write("platonic_axes_rotation", boxes)


def convergence() -> None:
    rows = []
    shapes = {"sphere": (1, 1, 1), "prolate": (2, 1, 1), "oblate": (2, 2, 1), "scalene": (3, 2, 1)}
    for name, radii in shapes.items():
        exact = M.ellipsoid_exact(*radii)
        for subdivisions in range(1, 8):
            nodes, faces = M.ellipsoid(radii, subdivisions)
            methods = ("pca", "min_volume") if subdivisions <= 5 else ("pca",)
            for method in methods:
                values = analyze_mesh(nodes, faces, axes=method).values
                keys = exact if method == "pca" else ("axis_S", "axis_I", "axis_L")
                for key in keys:
                    rows.append({"shape": name, "subdivisions": subdivisions,
                                 "n_faces": len(faces), "axes": method, "metric": key,
                                 "pyshape": values[key], "exact": exact[key],
                                 "error": rel(values[key], exact[key])})
    write("convergence", rows)


SHAPE_TABLE3 = {  # Angelidakis et al. (2021), Table 3: (SHAPE, Meshlab)
    "volume": (59.17430, 59.17434), "surface_area": (118.41040, 118.40588),
    "inertia_I1": (303.22310, 303.22311), "inertia_I2": (303.22310, 303.22311),
    "inertia_I3": (576.87850, 576.87848), "convexity": (0.655200, 0.655200),
    "sphericity_wadell": (0.620200, 0.620202),
}


def torus() -> None:
    nodes, faces = M.torus(3.0, 1.0)
    exact = M.torus_exact(3.0, 1.0)
    values = analyze_mesh(nodes, faces).values
    rows = [{"metric": key, "theory": exact[key], "pyshape": values[key],
             "error_pyshape": 1 - values[key] / exact[key],
             "shape_2021": SHAPE_TABLE3[key][0], "meshlab": SHAPE_TABLE3[key][1]}
            for key in SHAPE_TABLE3]
    write("torus", rows)


def union() -> None:
    rows = []
    cases = [("doublet", 1, 1.0, 0.25), ("satellite_1", 1, 0.3, 0.5),
             ("satellites_3", 3, 0.3, 0.5), ("satellites_6", 6, 0.2, 0.3)]
    for name, count, ratio, overlap in cases:
        for subdivisions in (4, 5, 6):
            nodes, faces, centers, radii = M.satellite_particle(
                count, ratio, overlap, seed=11, subdivisions=subdivisions)
            exact = M.sphere_union_exact(centers, radii)
            started = time.perf_counter()
            values = analyze_mesh(nodes, faces).values
            elapsed = time.perf_counter() - started
            naive_volume = sum(4 / 3 * np.pi * r ** 3 for r in radii)
            for key, ref in exact.items():
                rows.append({"case": name, "subdivisions": subdivisions,
                             "n_input_faces": len(faces), "metric": key,
                             "pyshape": values[key], "exact": ref, "error": rel(values[key], ref),
                             "naive_sum_error": rel(naive_volume, ref) if key == "volume" else "",
                             "seconds": elapsed})
    write("union", rows)


def indices() -> None:
    rows = []
    for label, (a, b, c) in {"c/b=0.01,b/a=0.2": (1.0, 0.2, 0.002),
                             "c/b=b/a=2/3": (1.0, 2 / 3, 4 / 9),
                             "c/b=0.6,b/a=0.2": (1.0, 0.2, 0.12)}.items():
        nodes, faces = M.cuboid(a, b, c)
        nodes = nodes @ M.random_rotation(np.random.default_rng(5)).T
        values = analyze_mesh(nodes, faces, axes="min_volume").values
        fl, el, co = M.angelidakis_indices(c, b, a)
        rows.append({"cuboid": label, "flatness": values["angelidakis_flatness"],
                     "flatness_exact": fl, "elongation": values["angelidakis_elongation"],
                     "elongation_exact": el, "compactness_exact": co})
    write("angelidakis_indices", rows)


def invariance() -> None:
    rows = []
    rng = np.random.default_rng(42)
    nodes, faces = M.random_particle(7, roughness=0.35)
    for method in ("pca", "min_volume"):
        base = analyze_mesh(nodes, faces, axes=method).values
        deviation = {key: 0.0 for key in DIMENSIONLESS}
        for _ in range(20):
            R = M.random_rotation(rng)
            perm = rng.permutation(len(nodes))
            inverse = np.argsort(perm)
            moved = (nodes @ R.T * rng.uniform(1e-4, 1e3) + rng.normal(scale=10, size=3))[perm]
            shuffled = inverse[faces][rng.permutation(len(faces))]
            if rng.random() < 0.5:
                shuffled = shuffled[:, ::-1]
            values = analyze_mesh(moved, shuffled, axes=method).values
            for key in DIMENSIONLESS:
                deviation[key] = max(deviation[key], abs(values[key] - base[key]))
        rows += [{"axes": method, "metric": key, "value": base[key], "max_abs_deviation": dev}
                 for key, dev in deviation.items()]
    write("invariance", rows)


def performance() -> None:
    """Run time against mesh size; also exports the meshes for SHAPE.

    For the SHAPE timings run shape_crosscheck.m on validation/_work_performance
    (see README.md); it writes results/shape_performance.json.
    """
    from scipy.io import savemat

    work = HERE / "_work_performance"
    work.mkdir(exist_ok=True)
    warm_nodes, warm_faces = M.icosphere(1)  # exclude one-off import costs from the timings
    analyze_mesh(warm_nodes, warm_faces, axes="min_volume")
    rows = []
    for subdivisions in range(2, 9):
        nodes, faces = M.icosphere(subdivisions)
        if subdivisions <= 5:
            savemat(work / f"icosphere_{subdivisions}.mat",
                    {"nodes": nodes, "faces": (faces + 1).astype(float)})
        timings = []
        for _ in range(3 if subdivisions < 8 else 1):
            started = time.perf_counter()
            analyze_mesh(nodes, faces)
            timings.append(time.perf_counter() - started)
        row = {"n_vertices": len(nodes), "n_faces": len(faces), "analyze_pca_s": min(timings),
               "min_volume_box_s": ""}
        if subdivisions <= 6:
            started = time.perf_counter()
            min_volume_box(nodes)
            row["min_volume_box_s"] = time.perf_counter() - started
        rows.append(row)
        print(row)
    write("performance", rows)


BENCHMARKS = {f.__name__: f for f in (platonic, convergence, torus, union, indices,
                                      invariance, performance)}

if __name__ == "__main__":
    for name in sys.argv[1:] or BENCHMARKS:
        started = time.perf_counter()
        BENCHMARKS[name]()
        print(f"[{name}] {time.perf_counter() - started:.1f} s")
