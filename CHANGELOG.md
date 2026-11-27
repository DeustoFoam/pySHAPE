# Changelog

All notable changes to pySHAPE. Versions follow the tags of the GitHub
repository; each release is archived on Zenodo (all versions:
https://doi.org/10.5281/zenodo.24004678).

## Unreleased

### New
- Preview (desktop and browser): each triangle is coloured by the side facing
  the viewer, light blue for the outer side and darker blue for the inner
  side (orange is kept for nested shells), so that
  the inside of the surface showing through openings or depth-sorting gaps is
  not mistaken for the outside; an opacity control makes the surface
  transparent (in preparation for showing internal shells and cavities).
  Display only; results are unchanged.

## 0.1.4 (2026-11-27)

### New
- Principal dimensions from an equivalent ellipsoid (#6):
  `axes="inertia_ellipsoid"` (ellipsoid with the particle's volume and
  principal moments of inertia) and `axes="fitted_ellipsoid"` (area-weighted
  least-squares ellipsoid through the surface vertices, checked to be an
  ellipsoid), in the library, `pyshape-batch --axes`, the HTTP API and both
  interfaces. New module `pyshape.ellipsoid` (`inertia_ellipsoid`,
  `fitted_ellipsoid`, `fit_ellipsoid`, `Ellipsoid`). Defaults are unchanged.

- In-app help in the desktop viewer and on both browser pages: hovering over
  a result or a control shows a one-line description; the "?" buttons (and,
  in the browser, a click on a result's name) open the definition, formula,
  what it depends on and the references. The texts are in the new module
  `pyshape.descriptors`, keyed by metric key and control name, and are served
  by `GET /api/descriptors`. The table of metric keys in `/docs` gains a
  description column. Results are unchanged.
- The browser interface (single-particle and batch pages) and the HTTP API
  accept the formats of the library except the text glTF (`.gltf`): STL
  (native, no extra package) and OBJ, PLY, OFF, GLB, 3MF and DAE (read by
  trimesh), with any case of the extension. A `.glb` must embed its buffers
  and images; DAE also needs `pycollada`. Results for STL files are
  unchanged.

- Every result of a file records its reader: `reader` is `builtin-stl`,
  `trimesh` (formats other than STL) or `trimesh-fallback`, with
  `reader_version` (the trimesh version when trimesh was used) and `warnings`.
  They are in the JSON results and the HTTP API, in CSV files (rows `reader`,
  `reader_version`, `warning` after `source_file`), in the batch summary
  (columns `reader`, `warnings`) and in both interfaces.
- `strict_stl=True` (`load_geometry`, `analyze_file`, `analyze_folder`) and
  `pyshape-batch --strict-stl` read STL files only with the built-in parser.

### Changed
- trimesh 5.0 or later is required (extras `mesh`, `gui`, `web`): it refuses
  references of uploaded files to files outside their folder.
- STL files are always read by the built-in parser, also when trimesh is
  installed (before, trimesh read them whenever it was installed). If the
  parser rejects a file and trimesh is installed, trimesh is tried and the
  result carries a warning with the parser's reason. Results for well-formed
  STL files are unchanged: merged vertices are now kept in order of first
  appearance, as trimesh did. Without trimesh, the vertex order changes from
  sorted to first appearance, which can change the PCA box of shapes whose
  principal directions are not unique.
- The STL parser gives the exact reason for a rejected file (e.g. a triangle
  count that does not match the file size), reads ASCII files with 8-bit
  (e.g. Latin-1) names, and rejects NaN or infinite coordinates in binary files
  too.
- CSV exports of files have the new metadata rows, and the batch summary the
  new columns, described above.
- Results are grouped by what they depend on (#7): "Mesh", "Size and mass
  properties", "Form, surface-based (independent of the box)" (Wadell
  sphericity, convexity, surface orientation tensor), "Principal dimensions
  (<box>)" (S, I, L) and "Form from S, I, L (<box>)" (Krumbein sphericity,
  Zingg ratios, Potticary, Kong & Fonseca and Angelidakis indices). Keys,
  values and units are unchanged; the order of the metrics, and therefore of
  the CSV rows and batch-summary columns, follows the groups (convexity now
  comes after Wadell sphericity). The eigenvalues are labelled
  "Eigenvalue f1 (surface orientation tensor)" etc.
  `pyshape.analysis.result_groups(axes)` gives the group names in order.
- The desktop viewer labels the axes control "Principal dimensions (S, I, L)",
  as the browser interface does (it was "Principal axes").
- The flatness/elongation indices of Angelidakis et al. are cited as 2022
  (Powder Technology 396, 689-695; published online in 2021).

### Development
- `tools/audit.py` (see `docs/audit.md`): inventory of every function, class
  and method (location, exported, documented, tested, deprecated) and a check
  of the public interface, CLI options, HTTP routes and the results of
  reference meshes against a recorded baseline (`tools/baseline/`).
  `python tools/audit.py tests` runs the unit tests and the check;
  `tests/test_audit.py` runs the check in pytest and CI.
- `networkx` is part of the `dev` extra (validation meshes in the tests).

## 0.1.3 (2026-11-24)

### Changed
- Mesh preparation (`validate_mesh`, `prepare_mesh`: validation, orientation of
  closed shells, solid union) moved from `pyshape.analysis` to the new module
  `pyshape.preprocess`. The old imports still work (`from pyshape import
  prepare_mesh`, `pyshape.analysis.prepare_mesh`,
  `pyshape.analysis.validate_mesh`). No change in results.

## 0.1.2 (2026-11-24)

### New
- Batch analysis of a folder of particles: `analyze_folder()` in the Python
  module and the `pyshape-batch` command write one CSV file per particle, named
  after its input file, with an optional summary table (`pyshape_summary.csv`).
  Files that cannot be analysed are reported and skipped.
- Batch page in the browser interface (`/batch`), separate from the
  single-particle page: choose a folder (optionally with subfolders) or several
  files, follow the progress per file, and download the CSV files one by one or
  as a zip archive with an optional summary.
- `POST /api/analyze` accepts `include_csv=true` and then also returns the CSV
  text.
- Complete API documentation at `/docs`: overview with examples, the list of
  metric keys, descriptions and allowed values of every field, response
  schemas, CSV download and error responses.
- `prepare_mesh` is public (`from pyshape import prepare_mesh`) and documented:
  it returns the surface that is actually analysed, after orientation of the
  closed shells and solid union.
- Explicit units: every result carries its unit, and the new `output_units`
  option (library, HTTP API and both interfaces) gives dimensional results in
  m (default, as before), mm, µm, cm, inch or the input units.

### Breaking changes
- The surface-orientation-tensor elongation is renamed from `orientation_R`
  ("rodness") to `orientation_E`, as in Orosz et al. (2021); values are
  unchanged. Readers of CSV or JSON exports that use `orientation_R` must
  be updated.
- CSV exports have a third column, `unit`.
- `units="auto"` (size-based guess of the input units) has been removed and is
  rejected with a list of the valid units; the desktop interface now starts
  with `as-is`, as the browser interface does.

### Documentation
- Orosz, Angelidakis and Bagi (2021) cited for the indices of the surface
  orientation tensor.
- Sadegh Nadimi added as author; author affiliations updated.

## 0.1.1 (2026-11-06)

See the GitHub release: https://github.com/DeustoFoam/pySHAPE/releases/tag/v0.1.1

## 0.1.0

First release.
