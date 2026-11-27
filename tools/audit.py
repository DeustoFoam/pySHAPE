"""Audit pySHAPE's functions and results, and detect changes to existing behaviour.

    python tools/audit.py list [--all] [--module NAME] [--format table|md|json]
    python tools/audit.py snapshot [--yes]
    python tools/audit.py check [--strict] [--rtol R] [--atol A]
    python tools/audit.py tests

``list`` prints every function, class and method of the package with its
location, whether it is public, exported, documented, deprecated and used by
the tests. ``snapshot`` records the public interface and the results of a fixed
set of reference meshes in ``tools/baseline/``. ``check`` compares the current
code with that baseline and exits with status 1 if existing behaviour changed.
See ``docs/audit.md``.

Only the standard library and NumPy are needed; SciPy (and manifold3d for the
multi-component cases) are needed to check the reference results.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pyshape"
TESTS = ROOT / "tests"
BASELINE = ROOT / "tools" / "baseline"
API_FILE = BASELINE / "api.json"
RESULTS_FILE = BASELINE / "results.json"
ASSETS = ROOT / "examples" / "assets"
# Synthetic files with nested shells, checked with cavities="keep".
CAVITY_FILES = ("hollow_sphere", "cube_with_spherical_void", "sphere_with_two_cavities",
                "nested_island")
# Synthetic open surfaces: volume-based results unavailable, diagnosis warning
# (a hole, and gaps that pyshape-repair closes).
OPEN_FILES = ("open_ellipsoid", "sphere_with_gaps")

# Modules whose internals belong to an interface (GUI/server) rather than to the
# library: changes there are reported as WARN, not BREAKING (their CLI options
# and HTTP routes are still checked strictly).
INTERFACE_MODULES = {"gui", "web"}
DEPRECATION_WARNINGS = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}
# Results that depend on the PCA box; they are not compared for meshes whose
# principal directions are not unique (the box then depends on round-off).
BOX_KEYS = {
    "axis_S", "axis_I", "axis_L", "sphericity_krumbein",
    "potticary_flatness", "potticary_elongation",
    "kongfonseca_flatness", "kongfonseca_elongation",
    "angelidakis_flatness", "angelidakis_elongation",
    "zingg_SI", "zingg_IL",
}


# --------------------------------------------------------------------------- #
# Static inventory (ast): works without importing the package or its extras.
# --------------------------------------------------------------------------- #

@dataclass
class Item:
    module: str
    qualname: str
    kind: str            # function, class, method, property, staticmethod, classmethod
    line: int
    public: bool
    exported: bool = False
    params: list = field(default_factory=list)
    returns: str | None = None
    bases: list = field(default_factory=list)
    docstring: bool = False
    deprecated: str | None = None
    tested: bool = False
    interface: bool = False

    @property
    def location(self) -> str:
        return f"pyshape/{self.module}.py:{self.line}"

    @property
    def signature(self) -> str:
        if self.kind in {"class", "property"} and not self.params:
            return ""
        parts, star = [], False
        for p in self.params:
            kind = p["kind"]
            if kind == "keyword_only" and not star:
                parts.append("*")
                star = True
            prefix = {"var_positional": "*", "var_keyword": "**"}.get(kind, "")
            if kind == "var_positional":
                star = True
            text = prefix + p["name"]
            if p["default"] is not None:
                text += "=" + p["default"]
            parts.append(text)
            if kind == "positional_only" and (
                    p is self.params[-1] or self.params[self.params.index(p) + 1]["kind"] != "positional_only"):
                parts.append("/")
        return "(" + ", ".join(parts) + ")"


def _unparse(node) -> str | None:
    return None if node is None else ast.unparse(node)


def _params(args: ast.arguments, drop_first: bool) -> list[dict]:
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    out = []
    for i, (arg, default) in enumerate(zip(positional, defaults)):
        kind = "positional_only" if i < len(args.posonlyargs) else "positional_or_keyword"
        out.append({"name": arg.arg, "kind": kind, "default": _unparse(default),
                    "annotation": _unparse(arg.annotation)})
    if drop_first and out:
        out = out[1:]
    if args.vararg:
        out.append({"name": args.vararg.arg, "kind": "var_positional", "default": None,
                    "annotation": _unparse(args.vararg.annotation)})
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append({"name": arg.arg, "kind": "keyword_only", "default": _unparse(default),
                    "annotation": _unparse(arg.annotation)})
    if args.kwarg:
        out.append({"name": args.kwarg.arg, "kind": "var_keyword", "default": None,
                    "annotation": _unparse(args.kwarg.annotation)})
    return out


def _decorator_names(node) -> set[str]:
    names = set()
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        names.add(ast.unparse(target).split(".")[-1])
    return names


def _deprecation(node) -> str | None:
    """Reason why a definition counts as deprecated, or None."""
    if "deprecated" in _decorator_names(node):
        return "@deprecated decorator"
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call) and ast.unparse(sub.func).split(".")[-1] == "warn":
            for arg in [*sub.args, *(k.value for k in sub.keywords)]:
                name = ast.unparse(arg).split(".")[-1]
                if name in DEPRECATION_WARNINGS:
                    return f"warns {name}"
    doc = ast.get_docstring(node) or ""
    if "deprecated" in doc.lower():
        return "docstring says deprecated"
    return None


def _dataclass_fields(node: ast.ClassDef) -> list[dict]:
    fields = []
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if ast.unparse(stmt.annotation).startswith("ClassVar"):
                continue
            fields.append({"name": stmt.target.id, "kind": "positional_or_keyword",
                           "default": _unparse(stmt.value),
                           "annotation": _unparse(stmt.annotation)})
    return fields


def _exports() -> tuple[list[str], dict[str, str]]:
    """``pyshape.__all__`` and, for each exported name, the module it comes from."""
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    names, origin = [], {}
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.level == 1 and stmt.module:
            for alias in stmt.names:
                origin[alias.asname or alias.name] = stmt.module
        if isinstance(stmt, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets):
            names = list(ast.literal_eval(stmt.value))
    return names, origin


def _test_names() -> set[str]:
    """Every identifier, attribute and imported name used in tests/."""
    names = set()
    for path in TESTS.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add((node.asname or node.name).split(".")[-1])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.isidentifier():
                names.add(node.value)
    return names


def modules() -> list[Path]:
    return sorted(p for p in PACKAGE.glob("*.py"))


def inventory() -> list[Item]:
    exported, origin = _exports()
    exported_from = {(origin.get(name), name) for name in exported}
    tested = _test_names()
    items: list[Item] = []
    for path in modules():
        module = path.stem
        if module == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        interface = module in INTERFACE_MODULES
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                items.append(Item(
                    module, node.name, "function", node.lineno, not node.name.startswith("_"),
                    exported=(module, node.name) in exported_from,
                    params=_params(node.args, drop_first=False), returns=_unparse(node.returns),
                    docstring=ast.get_docstring(node) is not None, deprecated=_deprecation(node),
                    tested=node.name in tested, interface=interface))
            elif isinstance(node, ast.ClassDef):
                is_dataclass = "dataclass" in _decorator_names(node)
                cls = Item(
                    module, node.name, "class", node.lineno, not node.name.startswith("_"),
                    exported=(module, node.name) in exported_from,
                    params=_dataclass_fields(node) if is_dataclass else [],
                    bases=[ast.unparse(b) for b in node.bases],
                    docstring=ast.get_docstring(node) is not None, deprecated=_deprecation(node),
                    tested=node.name in tested, interface=interface)
                items.append(cls)
                for sub in node.body:
                    if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    decorators = _decorator_names(sub)
                    kind = ("property" if "property" in decorators or "cached_property" in decorators
                            else "staticmethod" if "staticmethod" in decorators
                            else "classmethod" if "classmethod" in decorators else "method")
                    public = cls.public and (not sub.name.startswith("_") or sub.name == "__init__")
                    if sub.name.startswith("__") and sub.name.endswith("__") and sub.name != "__init__":
                        public = False
                    items.append(Item(
                        module, f"{node.name}.{sub.name}", kind, sub.lineno, public,
                        exported=cls.exported,
                        params=[] if kind == "property" else _params(sub.args, drop_first=kind != "staticmethod"),
                        returns=_unparse(sub.returns),
                        docstring=ast.get_docstring(sub) is not None, deprecated=_deprecation(sub),
                        tested=sub.name in tested, interface=interface))
    return items


def compat_modules() -> set[str]:
    """Modules kept for backward compatibility (docstring says so)."""
    out = set()
    for path in modules():
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        if "backward-compatible" in doc.lower() or "backwards compatib" in doc.lower():
            out.add(path.stem)
    return out


def aliases() -> dict[str, str]:
    """Public names that a module imports from another pyshape module.

    They can be imported from either module (e.g. ``pyshape.analysis.prepare_mesh``).
    """
    out = {}
    for path in modules():
        if path.stem == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.level == 1 and stmt.module:
                for alias in stmt.names:
                    name = alias.asname or alias.name
                    if not name.startswith("_"):
                        out[f"{path.stem}.{name}"] = f"{stmt.module}.{alias.name}"
    return out


def routes() -> list[str]:
    """HTTP routes declared in web.py (method and path)."""
    tree = ast.parse((PACKAGE / "web.py").read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr in {"get", "post", "put", "delete", "patch"}
                        and dec.args and isinstance(dec.args[0], ast.Constant)):
                    out.append(f"{dec.func.attr.upper()} {dec.args[0].value}")
    return sorted(set(out))


def cli() -> dict[str, dict]:
    """Console scripts from pyproject.toml and the options their parsers accept."""
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    out = {}
    for name, target in sorted(scripts.items()):
        module = target.split(":")[0].split(".")[-1]
        tree = ast.parse((PACKAGE / f"{module}.py").read_text(encoding="utf-8"))
        options = {}
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument" and node.args
                    and isinstance(node.args[0], ast.Constant)):
                flags = [a.value for a in node.args if isinstance(a, ast.Constant)]
                keywords = {k.arg: ast.unparse(k.value) for k in node.keywords
                            if k.arg in {"default", "choices", "action", "type", "nargs"}}
                options[" ".join(flags)] = keywords
        out[name] = {"target": target, "options": options}
    return out


# --------------------------------------------------------------------------- #
# Reference meshes (NumPy only, deterministic) and their results.
# --------------------------------------------------------------------------- #

def _icosphere(subdivisions: int):
    import numpy as np

    t = (1 + 5 ** 0.5) / 2
    verts = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t),
             (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
             (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
             (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(v, dtype=float) / np.linalg.norm(v) for v in verts]
    for _ in range(subdivisions):
        cache, new_faces = {}, []

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    return np.array(verts), np.array(faces)


def _rotation(a: float, b: float, c: float):
    import numpy as np

    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _cuboid(a, b, c):
    import numpy as np

    nodes = np.array([[x, y, z] for z in (0, c) for y in (0, b) for x in (0, a)], dtype=float)
    faces = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                      [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    return nodes, faces


def _torus(major, minor, n_ring, n_section):
    import numpy as np

    u = np.arange(n_ring) * 2 * np.pi / n_ring
    v = np.arange(n_section) * 2 * np.pi / n_section
    uu, vv = np.meshgrid(u, v, indexing="ij")
    nodes = np.stack([(major + minor * np.cos(vv)) * np.cos(uu),
                      (major + minor * np.cos(vv)) * np.sin(uu),
                      minor * np.sin(vv)], axis=-1).reshape(-1, 3)
    faces = []
    for i in range(n_ring):
        for j in range(n_section):
            a = i * n_section + j
            b = ((i + 1) % n_ring) * n_section + j
            c = ((i + 1) % n_ring) * n_section + (j + 1) % n_section
            d = i * n_section + (j + 1) % n_section
            faces += [(a, b, c), (a, c, d)]
    return nodes, np.array(faces)


def reference_meshes() -> dict[str, tuple]:
    """Name -> (nodes, faces) or a Path of an example file."""
    import numpy as np

    meshes = {}
    meshes["cube"] = _cuboid(1, 1, 1)
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    meshes["regular_tetrahedron"] = (tetra, np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]))
    nodes, faces = _cuboid(4, 2, 1)
    meshes["cuboid_4x2x1_rotated"] = (nodes @ _rotation(0.3, -0.7, 1.1).T + [5, -2, 1], faces)
    sphere, sfaces = _icosphere(3)
    meshes["ellipsoid_3_2_1_rotated"] = ((sphere * [3, 2, 1]) @ _rotation(0.4, 0.9, -0.2).T, sfaces)
    coeffs = np.random.default_rng(2026).uniform(-1, 1, 6)
    x, y, z = sphere.T
    radius = 1 + 0.12 * (coeffs[0] * np.sin(3 * x + coeffs[1]) * np.cos(2 * y)
                         + coeffs[2] * np.cos(4 * z + coeffs[3]) + coeffs[4] * x * y * z
                         + coeffs[5] * np.sin(5 * y))
    meshes["rough_particle"] = ((sphere * radius[:, None]) * [1.6, 1.0, 0.7], sfaces)
    meshes["torus_3_1"] = _torus(3.0, 1.0, 48, 16)
    small, small_faces = _icosphere(2)
    meshes["two_sphere_union"] = (np.vstack([small, small * 0.6 + [1.2, 0, 0]]),
                                  np.vstack([small_faces, small_faces + len(small)]))
    for path in sorted(ASSETS.glob("*.stl")):
        meshes[f"file:{path.name}"] = path
    for name in CAVITY_FILES + OPEN_FILES:
        meshes[f"file:synthetic/{name}.stl"] = ASSETS / "synthetic" / f"{name}.stl"
    return meshes


def _components(faces) -> int:
    import numpy as np

    parent = np.arange(int(faces.max()) + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in faces:
        ra = find(a)
        parent[find(b)] = ra
        parent[find(c)] = ra
    return len({find(i) for i in np.unique(faces)})


def _pca_unique(nodes) -> bool:
    import numpy as np

    sv = np.linalg.svd(nodes - nodes.mean(axis=0), compute_uv=False)
    gaps = np.abs(np.diff(sv)) / sv[0]
    return bool(gaps.min() > 1e-6)


def cases() -> list[dict]:
    """Every (mesh, options) combination checked, with the extras it needs."""
    out = []
    for name, mesh in reference_meshes().items():
        for axes in ("pca", "min_volume", "inertia_ellipsoid", "fitted_ellipsoid"):
            out.append({"id": f"{name}|axes={axes}", "mesh": name, "axes": axes,
                        "units": "as-is", "output_units": "m"})
    out.append({"id": "cuboid_4x2x1_rotated|units=mm->um", "mesh": "cuboid_4x2x1_rotated",
                "axes": "min_volume", "units": "mm", "output_units": "um"})
    # Nested shells as cavities (static files in examples/assets/synthetic).
    for name in CAVITY_FILES:
        for axes in ("pca", "min_volume"):
            out.append({"id": f"file:synthetic/{name}.stl|axes={axes}|cavities=keep",
                        "mesh": f"file:synthetic/{name}.stl", "axes": axes, "units": "as-is",
                        "output_units": "m", "cavities": "keep"})
    return out


def run_case(case: dict, meshes: dict) -> dict:
    """Analyse one case; returns its record (or raises ImportError for a missing extra)."""
    import numpy as np
    from pyshape import analyze_file, analyze_mesh, load_geometry

    mesh = meshes[case["mesh"]]
    options = dict(axes=case["axes"], units=case["units"], output_units=case["output_units"])
    if case.get("cavities", "fill") != "fill":
        options["cavities"] = case["cavities"]
    if isinstance(mesh, Path):
        geometry = load_geometry(mesh)
        nodes, faces = geometry.nodes, geometry.faces
        result = analyze_file(mesh, **options)
    else:
        nodes, faces = (np.asarray(a) for a in mesh)
        result = analyze_mesh(nodes, faces, **options)
    data = result.to_dict()
    excluded = []
    if case["axes"] == "pca":
        # The PCA box is measured on the envelope (outer surface) of the prepared mesh.
        from pyshape import prepare_solid
        envelope = prepare_solid(nodes, faces, cavities=case.get("cavities", "fill")).envelope_nodes
        if not _pca_unique(np.asarray(envelope, float)):
            excluded = sorted(BOX_KEYS)
    record = {
        "requires": ["scipy"] + (["manifold3d"] if _components(np.asarray(faces)) > 1 else []),
        "meta": {k: data[k] for k in ("source_name", "input_units", "scale_factor", "axes_method",
                                      "output_units", "output_scale", "reader", "warnings")},
        "excluded": excluded,
        "metrics": {m["key"]: {k: m[k] for k in ("value", "unit", "label", "group",
                                                  "default_selected", "error")}
                    for m in data["metrics"]},
    }
    if isinstance(mesh, Path):
        import csv
        import io
        rows = list(csv.reader(io.StringIO(result.to_csv())))
        record["csv"] = [[r[0], r[2]] for r in rows if r[0] not in {"reader_version"}]
    return record


def _missing(requires) -> list[str]:
    return [m for m in requires if importlib.util.find_spec(m) is None]


def results_snapshot() -> dict:
    if _missing(["scipy", "manifold3d"]):
        raise SystemExit("snapshot needs SciPy and manifold3d: pip install -e \".[mesh,analysis]\"")
    meshes = reference_meshes()
    out = {case["id"]: run_case(case, meshes) for case in cases()}
    with tempfile.TemporaryDirectory() as tmp:
        from pyshape import analyze_folder
        target = Path(tmp) / ASSETS.joinpath("Tetrahedron_ascii.stl").name
        target.write_bytes(ASSETS.joinpath("Tetrahedron_ascii.stl").read_bytes())
        batch = analyze_folder(tmp, summary=True)
        header = batch.summary_path.read_text(encoding="utf-8").splitlines()[0]
    return {"cases": out, "batch_summary_header": header}


def api_snapshot() -> dict:
    exported, _ = _exports()
    return {
        "exports": sorted(exported),
        "objects": {
            f"{i.module}.{i.qualname}": {
                "kind": i.kind, "params": i.params, "returns": i.returns, "bases": i.bases,
                "deprecated": i.deprecated, "interface": i.interface, "exported": i.exported}
            for i in inventory() if i.public},
        "aliases": aliases(),
        "routes": routes(),
        "cli": cli(),
    }


# --------------------------------------------------------------------------- #
# Comparison.
# --------------------------------------------------------------------------- #

class Report:
    LEVELS = ("BREAKING", "WARN", "NOTICE", "ADDED", "SKIPPED")

    def __init__(self):
        self.entries: list[tuple[str, str, str]] = []

    def add(self, level: str, where: str, text: str):
        self.entries.append((level, where, text))

    def count(self, level: str) -> int:
        return sum(1 for e in self.entries if e[0] == level)

    def print(self, title: str):
        print(f"\n== {title}")
        if not self.entries:
            print("   no changes")
        for level in self.LEVELS:
            for lvl, where, text in self.entries:
                if lvl == level:
                    print(f"   {level:<9}{where}: {text}")


def compare_params(old: list, new: list, where: str, report: Report, level: str):
    new_by = {p["name"]: p for p in new}
    old_by = {p["name"]: p for p in old}
    for p in old:
        q = new_by.get(p["name"])
        if q is None:
            report.add(level, where, f"parameter '{p['name']}' removed or renamed")
            continue
        if p["kind"] != q["kind"]:
            report.add(level, where, f"parameter '{p['name']}' changed from {p['kind']} to {q['kind']}")
        if p["default"] is not None and q["default"] is None and q["kind"] not in {"var_positional", "var_keyword"}:
            report.add(level, where, f"parameter '{p['name']}' is now required (default was {p['default']})")
        elif p["default"] is not None and p["default"] != q["default"]:
            report.add(level, where, f"default of '{p['name']}' changed: {p['default']} -> {q['default']}")
        elif p["default"] is None and q["default"] is not None:
            report.add("ADDED", where, f"parameter '{p['name']}' now has a default ({q['default']})")
        if p["annotation"] != q["annotation"]:
            report.add("NOTICE", where, f"annotation of '{p['name']}': {p['annotation']} -> {q['annotation']}")
    positional = lambda ps: [p["name"] for p in ps if p["kind"] in {"positional_only", "positional_or_keyword"}]
    old_pos = [n for n in positional(old) if n in new_by]
    new_pos = [n for n in positional(new) if n in old_by]
    if old_pos != new_pos:
        report.add(level, where, f"order of positional parameters changed: {old_pos} -> {new_pos}")
    for q in new:
        if q["name"] in old_by:
            continue
        if q["default"] is None and q["kind"] not in {"var_positional", "var_keyword"}:
            report.add(level, where, f"new required parameter '{q['name']}'")
        else:
            report.add("ADDED", where, f"new optional parameter '{q['name']}'")


def check_api(old: dict, new: dict) -> Report:
    report = Report()
    for name in sorted(set(old["exports"]) - set(new["exports"])):
        report.add("BREAKING", "pyshape.__all__", f"'{name}' no longer exported")
    for name in sorted(set(new["exports"]) - set(old["exports"])):
        report.add("ADDED", "pyshape.__all__", f"'{name}' exported")
    for name, o in sorted(old["objects"].items()):
        level = "WARN" if o["interface"] else "BREAKING"
        n = new["objects"].get(name)
        if n is None:
            report.add(level, name, f"{o['kind']} removed, renamed or made private")
            continue
        if o["kind"] != n["kind"]:
            report.add(level, name, f"changed from {o['kind']} to {n['kind']}")
        compare_params(o["params"], n["params"], name, report, level)
        if o["returns"] != n["returns"]:
            report.add("NOTICE", name, f"return annotation {o['returns']} -> {n['returns']}")
        if o["bases"] != n["bases"]:
            report.add("NOTICE", name, f"base classes {o['bases']} -> {n['bases']}")
        if n["deprecated"] and not o["deprecated"]:
            report.add("NOTICE", name, f"now deprecated ({n['deprecated']})")
    for name, n in sorted(new["objects"].items()):
        if name not in old["objects"]:
            report.add("ADDED", name, f"new {n['kind']}")
    compat = compat_modules()
    for name, target in sorted(old["aliases"].items()):
        if name not in new["aliases"]:
            level = "BREAKING" if name.split(".")[0] in compat else "WARN"
            report.add(level, name, f"no longer importable from here (was {target})")
        elif new["aliases"][name] != target:
            report.add("NOTICE", name, f"re-export now points to {new['aliases'][name]}")
    for route in sorted(set(old["routes"]) - set(new["routes"])):
        report.add("BREAKING", "HTTP API", f"route '{route}' removed")
    for route in sorted(set(new["routes"]) - set(old["routes"])):
        report.add("ADDED", "HTTP API", f"route '{route}'")
    for script, o in sorted(old["cli"].items()):
        n = new["cli"].get(script)
        if n is None:
            report.add("BREAKING", script, "console script removed")
            continue
        if n["target"] != o["target"]:
            report.add("NOTICE", script, f"entry point {o['target']} -> {n['target']}")
        for flag, spec in sorted(o["options"].items()):
            if flag not in n["options"]:
                report.add("BREAKING", script, f"option '{flag}' removed")
            elif n["options"][flag] != spec:
                for key in sorted(set(spec) | set(n["options"][flag])):
                    a, b = spec.get(key), n["options"][flag].get(key)
                    if a != b:
                        level = "ADDED" if key == "choices" and a and b and _superset(b, a) else "BREAKING"
                        report.add(level, script, f"option '{flag}' {key}: {a} -> {b}")
        for flag in sorted(set(n["options"]) - set(o["options"])):
            report.add("ADDED", script, f"new option '{flag}'")
    return report


def _superset(new_text: str, old_text: str) -> bool:
    try:
        return set(ast.literal_eval(old_text)) <= set(ast.literal_eval(new_text))
    except (ValueError, SyntaxError, TypeError):
        return False


def _close(a, b, rtol, atol) -> bool:
    if a is None or b is None:
        return a is b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def check_results(old: dict, rtol: float, atol: float) -> Report:
    report = Report()
    meshes = reference_meshes()
    current = {c["id"]: c for c in cases()}
    for case_id, base in old["cases"].items():
        missing = _missing(base["requires"])
        if missing:
            report.add("SKIPPED", case_id, f"needs {', '.join(missing)}")
            continue
        if case_id not in current:
            report.add("WARN", case_id, "reference case no longer defined in tools/audit.py")
            continue
        try:
            new = run_case(current[case_id], meshes)
        except Exception as exc:  # noqa: BLE001 - any failure is a finding
            report.add("BREAKING", case_id, f"analysis failed: {type(exc).__name__}: {exc}")
            continue
        for key, value in base["meta"].items():
            if key in {"scale_factor", "output_scale"}:
                same = _close(value, new["meta"][key], rtol, atol)
            else:
                same = value == new["meta"][key]
            if not same:
                report.add("BREAKING", case_id, f"{key}: {value!r} -> {new['meta'][key]!r}")
        for key, m in base["metrics"].items():
            n = new["metrics"].get(key)
            if n is None:
                report.add("BREAKING", case_id, f"result '{key}' missing")
                continue
            if key not in base["excluded"] and not _close(m["value"], n["value"], rtol, atol):
                rel = ("" if None in (m["value"], n["value"]) or m["value"] == 0
                       else f" (relative change {abs(n['value'] - m['value']) / abs(m['value']):.2e})")
                report.add("BREAKING", case_id, f"{key}: {m['value']!r} -> {n['value']!r}{rel}")
            if m["unit"] != n["unit"]:
                report.add("BREAKING", case_id, f"unit of '{key}': {m['unit']!r} -> {n['unit']!r}")
            for attr in ("label", "group", "default_selected", "error"):
                if m[attr] != n[attr]:
                    report.add("NOTICE", case_id, f"{attr} of '{key}': {m[attr]!r} -> {n[attr]!r}")
        for key in sorted(set(new["metrics"]) - set(base["metrics"])):
            report.add("ADDED", case_id, f"new result '{key}'")
        if "csv" in base and base["csv"] != new.get("csv"):
            report.add("BREAKING", case_id, "CSV rows (parameter/unit columns) changed")
    if not _missing(["scipy"]):
        try:
            header = results_header()
        except Exception as exc:  # noqa: BLE001
            report.add("BREAKING", "batch summary", f"failed: {type(exc).__name__}: {exc}")
        else:
            if header != old["batch_summary_header"]:
                report.add("BREAKING", "batch summary",
                           f"CSV header changed:\n             {old['batch_summary_header']}\n"
                           f"          -> {header}")
    for case_id in sorted(set(current) - set(old["cases"])):
        report.add("ADDED", case_id, "new reference case (not in the baseline yet)")
    return report


def results_header() -> str:
    from pyshape import analyze_folder

    with tempfile.TemporaryDirectory() as tmp:
        source = ASSETS / "Tetrahedron_ascii.stl"
        (Path(tmp) / source.name).write_bytes(source.read_bytes())
        batch = analyze_folder(tmp, summary=True)
        return batch.summary_path.read_text(encoding="utf-8").splitlines()[0]


# --------------------------------------------------------------------------- #
# Commands.
# --------------------------------------------------------------------------- #

def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"No baseline at {path.relative_to(ROOT)}; run: python tools/audit.py snapshot")
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8", newline="\n")


def cmd_list(args) -> int:
    items = [i for i in inventory()
             if (args.all or i.public) and (args.module is None or i.module == args.module)]
    if args.format == "json":
        print(json.dumps([{**i.__dict__, "location": i.location, "signature": i.signature}
                          for i in items], indent=1))
        return 0
    yes = lambda flag: "yes" if flag else "-"
    headers = ["name", "kind", "location", "public", "exported", "doc", "tested", "deprecated"]
    rows = [[i.qualname + i.signature if args.signatures else i.qualname, i.kind, i.location,
             yes(i.public), yes(i.exported), yes(i.docstring), yes(i.tested), i.deprecated or "-"]
            for i in items]
    if args.format == "md":
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for row in rows:
            print("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    else:
        widths = [min(60, max(len(h), *(len(r[k]) for r in rows))) if rows else len(h)
                  for k, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  ".join("-" * w for w in widths))
        for row in rows:
            print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    public = [i for i in items if i.public]
    print(f"\n{len(items)} definitions listed ({len(public)} public, "
          f"{sum(i.exported for i in items if i.kind in ('function', 'class'))} exported from pyshape)")
    untested = [i for i in public if not i.tested and not i.interface]
    undocumented = [i for i in public if not i.docstring and i.kind != "property"]
    deprecated = [i for i in items if i.deprecated]
    print(f"Public library definitions not referenced by any test: {len(untested)}")
    for i in untested:
        print(f"   {i.module}.{i.qualname}  ({i.location})")
    print(f"Public definitions without a docstring: {len(undocumented)}")
    for i in undocumented:
        print(f"   {i.module}.{i.qualname}  ({i.location})")
    print(f"Deprecated: {len(deprecated)}")
    for i in deprecated:
        print(f"   {i.module}.{i.qualname}  ({i.location}): {i.deprecated}")
    compat = sorted(compat_modules())
    shim_names = [n for n in aliases() if n.split(".")[0] in compat]
    print(f"Backward-compatibility modules: {', '.join(compat) or '-'} "
          f"({len(shim_names)} re-exported names, all checked)")
    return 0


def run_check(rtol: float, atol: float, strict: bool) -> int:
    api = check_api(_load(API_FILE), api_snapshot())
    api.print("Interface (functions, classes, signatures, exports, CLI options, HTTP routes)")
    if _missing(["numpy"]):
        raise SystemExit("NumPy is needed: pip install -e .")
    sys.path.insert(0, str(ROOT))
    results = check_results(_load(RESULTS_FILE), rtol, atol)
    results.print(f"Results of the reference meshes (rtol={rtol:g}, atol={atol:g})")
    breaking = api.count("BREAKING") + results.count("BREAKING")
    warn = api.count("WARN") + results.count("WARN")
    added = api.count("ADDED") + results.count("ADDED")
    skipped = results.count("SKIPPED")
    print(f"\nSummary: {breaking} breaking, {warn} warnings, {added} additions, {skipped} skipped cases.")
    failed = breaking or (strict and warn)
    if failed:
        print("FAILED: existing behaviour changed. If the change is intended, run "
              "'python tools/audit.py snapshot' and commit tools/baseline/.")
    else:
        print("OK: existing behaviour is preserved.")
    return 1 if failed else 0


def cmd_check(args) -> int:
    return run_check(args.rtol, args.atol, args.strict)


def cmd_snapshot(args) -> int:
    sys.path.insert(0, str(ROOT))
    if API_FILE.exists() and RESULTS_FILE.exists() and not args.yes:
        print("Current differences from the baseline that the new snapshot would accept:")
        status = run_check(args.rtol, args.atol, strict=False)
        if status:
            print("\nNot written: re-run with --yes to accept these breaking changes.")
            return 1
    _write(API_FILE, api_snapshot())
    _write(RESULTS_FILE, results_snapshot())
    print(f"Baseline written: {API_FILE.relative_to(ROOT)}, {RESULTS_FILE.relative_to(ROOT)}")
    return 0


def cmd_tests(args) -> int:
    print("== pytest")
    status = subprocess.call([sys.executable, "-m", "pytest", "-q", str(TESTS)], cwd=ROOT)
    audit = run_check(args.rtol, args.atol, args.strict)
    print(f"\nVerdict: tests {'passed' if status == 0 else 'FAILED'}, "
          f"audit {'passed' if audit == 0 else 'FAILED'}.")
    return 1 if status or audit else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/audit.py", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list functions, classes and methods")
    p.add_argument("--all", action="store_true", help="include private definitions")
    p.add_argument("--module", help="only this module, e.g. analysis")
    p.add_argument("--signatures", action="store_true", help="show parameters next to the names")
    p.add_argument("--format", choices=["table", "md", "json"], default="table")
    p.set_defaults(func=cmd_list)
    for name, func, text in (("check", cmd_check, "compare the code with the baseline"),
                             ("snapshot", cmd_snapshot, "record the baseline in tools/baseline/"),
                             ("tests", cmd_tests, "run pytest, then check")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--rtol", type=float, default=1e-9, help="relative tolerance (default 1e-9)")
        p.add_argument("--atol", type=float, default=1e-12, help="absolute tolerance (default 1e-12)")
        if name == "snapshot":
            p.add_argument("--yes", action="store_true", help="accept breaking changes")
        else:
            p.add_argument("--strict", action="store_true", help="also fail on warnings")
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
