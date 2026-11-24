# Changelog

All notable changes to pySHAPE. Versions follow the tags of the GitHub
repository; each release is archived on Zenodo (all versions:
https://doi.org/10.5281/zenodo.24004678).

## Unreleased

### New
- The browser interface (single-particle and batch pages) and the HTTP API
  accept the formats of the library except the text glTF (`.gltf`): STL
  (native, no extra package) and OBJ, PLY, OFF, GLB, 3MF and DAE (read by
  trimesh), with any case of the extension. A `.glb` must embed its buffers
  and images; DAE also needs `pycollada`. Results for STL files are
  unchanged.

### Changed
- trimesh 5.0 or later is required (extras `mesh`, `gui`, `web`): it refuses
  references of uploaded files to files outside their folder.

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
