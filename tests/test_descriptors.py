"""In-app help texts (pyshape.descriptors); no Tk or Matplotlib is imported."""
import ast
import re
from pathlib import Path

import numpy as np
import pytest

from pyshape import analyze_mesh, descriptors
from pyshape.bounding import AXES_METHODS

PACKAGE = Path(descriptors.__file__).parent
TETRA_NODES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
TETRA_FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
FIELDS = {"short", "description", "formula", "depends_on", "references"}


def _check_entry(name, entry):
    assert set(entry) == FIELDS, name
    assert entry["short"].strip() and entry["description"].strip(), name
    assert "\n" not in entry["short"], name
    for key in entry["references"]:
        assert key in descriptors.REFERENCES, (name, key)


def test_every_metric_key_has_help():
    keys = {metric.key for metric in analyze_mesh(TETRA_NODES, TETRA_FACES).metrics}
    open_keys = {metric.key for metric in analyze_mesh(TETRA_NODES, TETRA_FACES[:3]).metrics}
    assert keys == open_keys
    for key in keys:
        assert descriptors.metric_help(key) is not None, key
        _check_entry(key, descriptors.METRICS[key])
    assert set(descriptors.METRICS) == keys  # no help for keys that do not exist


def test_every_axes_method_has_help():
    for method in AXES_METHODS:
        _check_entry(method, descriptors.axes_help(method))
    for method, entry in descriptors.AXES_METHODS.items():  # entries for future methods too
        _check_entry(method, entry)


def test_references_are_well_formed():
    for key, ref in descriptors.REFERENCES.items():
        assert ref["citation"].strip(), key
        assert ref["url"] == "" or re.fullmatch(r"https://doi\.org/10\.\d{4,9}/\S+", ref["url"]), key
    used = {key for table in (descriptors.METRICS, descriptors.CONTROLS, descriptors.AXES_METHODS)
            for entry in table.values() for key in entry["references"]}
    assert used == set(descriptors.REFERENCES)


def test_orientation_indices_formulas():
    assert descriptors.METRICS["orientation_C"]["formula"] == "C = f3 / f1"
    assert descriptors.METRICS["orientation_F"]["formula"] == "F = (f1 − f2) / f1"
    assert descriptors.METRICS["orientation_E"]["formula"] == "E = (f2 − f3) / f1"
    for key in ("orientation_C", "orientation_F", "orientation_E"):
        assert descriptors.METRICS[key]["references"] == ["bagiorosz2020", "orosz2021"]


def _gui_control_names():
    """Control names used by the desktop GUI, read from its source (no Tk import)."""
    source = (PACKAGE / "gui.py").read_text(encoding="utf-8")
    names = set(re.findall(r'_help_label\(\s*[\w.]+,\s*"[^"]*",\s*"([a-z_]+)"\)', source))
    names |= set(re.findall(r'_tooltip\(\s*[\w.]+,\s*"([a-z_]+)"\)', source))
    return names


def _web_control_names(page):
    return set(re.findall(r'data-help="([a-z_]+)"', (PACKAGE / "static" / page).read_text(
        encoding="utf-8")))


@pytest.mark.parametrize("names, expected", [
    (_gui_control_names(), {"file", "input_units", "output_units", "axes", "load", "calculate",
                            "results", "select_all", "select_none", "suffix", "avoid_overwrite",
                            "export", "auto_rotate", "opacity", "surface_sides"}),
    (_web_control_names("index.html"), {"file", "input_units", "output_units", "axes",
                                        "calculate", "results", "select_all", "select_none",
                                        "suffix", "export", "auto_rotate", "opacity",
                                        "surface_sides"}),
    (_web_control_names("batch.html"), {"folder", "files", "recursive", "input_units",
                                        "output_units", "axes", "suffix", "summary", "run",
                                        "stop", "download_all"}),
], ids=["desktop", "web-single", "web-batch"])
def test_every_control_has_help(names, expected):
    assert names == expected
    for name in names:
        _check_entry(name, descriptors.control_help(name))


def test_group_help_is_optional():
    assert descriptors.group_help("Mesh")
    assert descriptors.group_help("Principal dimensions (PCA)") == \
        descriptors.GROUPS["Principal dimensions"]
    assert descriptors.group_help("No such group") == ""


def test_explanations():
    text = descriptors.explanation(descriptors.METRICS["orientation_E"], "Elongation")
    assert text.startswith("Elongation\n\n")
    assert "Formula: E = (f2 − f3) / f1" in text
    assert "https://doi.org/10.1016/j.powtec.2021.08.054" in text
    text = descriptors.axes_control_explanation([*AXES_METHODS, "unknown"])
    for method in AXES_METHODS:
        assert f"{method}: {descriptors.AXES_METHODS[method]['short']}" in text
    assert text.index("unknown") < text.index("References:")


def test_module_has_no_numpy_or_ui_imports():
    tree = ast.parse((PACKAGE / "descriptors.py").read_text(encoding="utf-8"))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= {"__future__", "typing"}
