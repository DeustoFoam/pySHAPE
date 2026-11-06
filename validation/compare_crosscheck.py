"""Compare pySHAPE with the original MATLAB SHAPE (step 3 of the cross-validation).

Reads results/pyshape_crosscheck.json and results/shape_matlab.json and writes
results/crosscheck.csv (one row per particle and descriptor) and
results/crosscheck_summary.csv (worst case per descriptor).

The difference is relative, |p - s| / |s|, except for quantities whose
reference value is below 1e-3 (e.g. flatness of a sphere), where the absolute
difference is reported. PCA-based dimensions are compared only for particles
whose vertex covariance has well separated eigenvalues (ratio of consecutive
eigenvalues below 1 - 1e-3); otherwise the principal directions are not unique
and both programs return an arbitrary, implementation-dependent box.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WORK = HERE / "_work"

SURFACE = ["volume", "surface_area", "convexity", "sphericity_wadell", "inertia_I1",
           "inertia_I2", "inertia_I3", "orientation_C", "orientation_F", "orientation_R"]
AXIS = ["axis_S", "axis_I", "axis_L", "sphericity_krumbein", "potticary_flatness",
        "potticary_elongation", "kongfonseca_flatness", "kongfonseca_elongation",
        "angelidakis_flatness", "angelidakis_elongation", "zingg_SI", "zingg_IL"]


def difference(p: float, s: float) -> tuple[float, str]:
    if abs(s) < 1e-3:
        return abs(p - s), "absolute"
    return abs(p - s) / abs(s), "relative"


def pca_is_unique(name: str) -> bool:
    nodes = loadmat(WORK / f"{name}.mat")["nodes"]
    eigen = np.sort(np.linalg.eigvalsh(np.cov((nodes - nodes.mean(0)).T)))[::-1]
    return bool(eigen[1] / eigen[0] < 1 - 1e-3 and eigen[2] / eigen[1] < 1 - 1e-3)


def main() -> None:
    ours = json.loads((RESULTS / "pyshape_crosscheck.json").read_text())
    theirs = json.loads((RESULTS / "shape_matlab.json").read_text())
    rows = []
    for name, p in ours.items():
        s = theirs[name]
        unique = pca_is_unique(name)
        for key in SURFACE:
            diff, kind = difference(p["pca"][key], s[key])
            rows.append({"particle": name, "group": "surface", "metric": key,
                         "pyshape": p["pca"][key], "shape": s[key], "difference": diff,
                         "kind": kind, "pca_unique": unique})
        for method, group in (("pca", "pca_box"), ("min_volume", "min_volume_box")):
            for key in AXIS:
                diff, kind = difference(p[method][key], s[method][key])
                rows.append({"particle": name, "group": group, "metric": key,
                             "pyshape": p[method][key], "shape": s[method][key],
                             "difference": diff, "kind": kind, "pca_unique": unique})
        rows.append({"particle": name, "group": "timing", "metric": "seconds",
                     "pyshape": p["seconds_pca"], "shape": s["seconds_particle"],
                     "difference": "", "kind": "", "pca_unique": unique})
        rows.append({"particle": name, "group": "timing", "metric": "seconds_min_volume_box",
                     "pyshape": p["seconds_min_volume"] - p["seconds_pca"],
                     "shape": s["seconds_minboundbox"], "difference": "", "kind": "",
                     "pca_unique": unique})
    with (RESULTS / "crosscheck.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for group, keys in (("surface", SURFACE), ("pca_box", AXIS), ("min_volume_box", AXIS)):
        for key in keys:
            selected = [r for r in rows if r["group"] == group and r["metric"] == key
                        and (group != "pca_box" or r["pca_unique"])]
            worst = max(selected, key=lambda r: r["difference"])
            summary.append({"group": group, "metric": key, "n_particles": len(selected),
                            "max_difference": worst["difference"], "kind": worst["kind"],
                            "worst_particle": worst["particle"],
                            "median_difference": float(np.median([r["difference"] for r in selected]))})
    with (RESULTS / "crosscheck_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['group']:15s} {row['metric']:24s} n={row['n_particles']:2d} "
              f"max={row['max_difference']:.2e} ({row['kind']}, {row['worst_particle']}) "
              f"median={row['median_difference']:.2e}")


if __name__ == "__main__":
    main()
