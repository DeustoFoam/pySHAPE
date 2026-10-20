# SHAPE Python

This repository contains a lightweight Python port of selected functionality from SHAPE, the SHape Analyser for Particle Engineering. It is derivative work based on the original project by Vasileios Angelidakis, Sadegh Nadimi, and Stefano Utili. Visit the [original SHAPE GitHub repository](https://github.com/vsangelidakis/SHAPE) for the full MATLAB toolbox and project history.

The Python package provides tools for 3D particle geometry and form analysis. It does not currently include all features of the original MATLAB toolbox.

## Features

- Geometry: triangle-mesh surface area and tetrahedral-mesh volume, centroid, and inertia
- Form: convexity, Wadell and Krumbein sphericities, Zingg ratios, and Potticary and Kong-Fonseca flatness/elongation parameters
- Surface orientation tensor
- Surface roughness statistics
- STL loading, with `trimesh` support and a built-in fallback
- An installable, headless Python library with JSON-compatible analysis results
- Optional local HTTP API and HTML viewer, plus the existing desktop viewer

## Project layout

- `pyshape/` — Python package
- `pyshape/analysis.py` — shared analysis service, independent of the interfaces
- `pyshape/web.py` and `pyshape/static/` — optional API and browser viewer
- `examples/` — runnable demos and a bundled ASCII STL asset
- `tests/` — unit tests
- `pyproject.toml` — package metadata and optional dependency groups
- `requirements.txt` — convenience installation of all development/viewer extras

## Requirements and installation

Python 3.9 or later is required. The core library depends only on NumPy and
does not import Tk, Matplotlib, FastAPI, or SciPy at startup.

From a local checkout:

```bash
python -m pip install .
# Editable installation while developing:
python -m pip install -e ".[mesh,analysis,web,dev]"
```

From another project, **after the `module` branch has been pushed to GitHub**:

```bash
python -m pip install "pyshape @ git+https://github.com/Lollum89/pySHAPE.git@module"
# Include all file loaders and convexity:
python -m pip install "pyshape[mesh,analysis] @ git+https://github.com/Lollum89/pySHAPE.git@module"
```

Pin a release tag or commit instead of a moving branch for reproducible
downstream installations. The Python import name is `pyshape`.

Optional extras can be combined:

| Extra | Purpose |
| --- | --- |
| `mesh` | Trimesh loaders and Manifold solid unions for multi-sphere/multi-component particles; some file formats need additional trimesh dependencies |
| `analysis` | SciPy convex-hull volume and convexity |
| `web` | FastAPI, Uvicorn, multipart uploads, the bundled HTML viewer, and the multi-component union backend |
| `gui` | Matplotlib, Trimesh, and SciPy for the desktop viewer; Tk must be installed through your OS |
| `dev` | Pytest, HTTP test client, and build tooling |

To install all development and viewer dependencies:

```bash
python -m pip install -r requirements.txt
```

## Quick start

### Analyze a file

```python
from pyshape import analyze_file

result = analyze_file("particle.stl", units="mm")
print(result.values["surface_area"])  # m²
print(result.values["volume"])        # m³, or None when unavailable

for metric in result.metrics:
    print(metric.key, metric.text)

result.write_csv("particle_SHAPE.csv", keys=["surface_area", "volume"])

import json
json_text = json.dumps(result.to_dict(), allow_nan=False)
```

`analyze_file` loads geometry and delegates to `analyze_mesh(nodes, faces)`.
Both return an `AnalysisResult` containing 25 metric records. No window or
server is started. `load_geometry(path)` is also public when you need the
mesh arrays and source metadata.

### Analyze arrays or call individual functions

```python
import numpy as np
from pyshape import analyze_mesh, surface_area, surface_orientation_tensor

nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

print(surface_area(nodes, faces))
C, F, R, eigenvalues, eigenvectors = surface_orientation_tensor(nodes, faces)
result = analyze_mesh(nodes, faces)
# This is an open surface: area is available; volume-dependent results are not.
print(result.values["surface_area"])
```

All existing low-level public imports remain available, including tetrahedral
volume/inertia and roughness functions. The high-level triangle-mesh analysis
does not derive surface roughness or tetrahedral inertia automatically.

### Input and result contract

- `analyze_mesh` accepts nonempty `(N, 3)` vertex coordinates and `(M, 3)`
  integer triangle indices. Invalid shapes, non-finite coordinates,
  out-of-range indices, and zero-area meshes raise `ValueError`.
- Indices should be zero-based. One-based MATLAB-style indices are converted
  only when the minimum is 1 and the maximum equals the vertex count.
  Ambiguous arrays must be converted explicitly by the caller.
- Units default to `"as-is"`: coordinates are unchanged. `"mm"`, `"cm"`,
  `"m"`, and `"inch"` convert lengths to metres; area and volume then use
  m² and m³. `"auto"` explicitly opts into the old desktop heuristic:
  maximum span > 10000 scales by 1e-6, > 10 by 1e-3, otherwise by 1.
  This guesses a scale; it does not detect units. The desktop retains its
  existing `"auto"` default.
- Results contain mesh counts, surface area, volume, centroid, Wadell and
  Krumbein sphericity, orientation indices/eigenvalues, PCA-estimated axes,
  Potticary/Kong-Fonseca/Zingg parameters, and convexity. PCA axes are
  estimates, not exact minimum bounding-box dimensions.
- Closed, orientable mesh shells have their triangle winding corrected
  automatically, including wholly reversed shells; the original arrays/files
  are not changed. Volume requires nonzero enclosed volume. Open/non-manifold
  meshes are not filled and still report unavailable volume-dependent metrics.
- Multi-sphere/multi-component particles are interpreted as a **solid union**:
  overlapping material counts once, and internal surfaces are excluded from
  area and orientation calculations. Nested shells represent additional solid
  components, not cavities. Install `.[mesh]` for Python-library use; the web
  and desktop extras already include the union backend. Missing union support
  raises an explicit dependency error, never a sum-of-volumes fallback.
  Single-shell winding correction remains NumPy-only.
- Boolean unions operate on the supplied triangle surfaces, not fitted ideal
  spheres. Coordinates are normalized during the union to preserve precision
  for micron-scale inputs. Returned mesh counts, PCA estimates, and previews
  describe the union mesh. Self-intersections within a shell are unsupported;
  this is not a general mesh-repair or cavity-detection tool.
- Each metric has `key`, `label`, `value`, `group`, `default_selected`,
  and `error`. Unavailable values are `None` (`null` in JSON), with an
  explanatory error. Missing SciPy affects only convexity. Other unexpected
  computation or loader errors propagate rather than becoming valid results.
- `result.values` maps stable metric keys to numeric values or `None`.
  `result.to_dict()` includes the full metric records, `source_name`,
  `input_units`, and `scale_factor`. `source_name` is `None` for array input.
- `result.to_csv()` exports default-selected available results using stable
  metric keys. Pass `keys=[...]` for an explicit selection; unknown,
  unavailable, or empty selections raise `ValueError`.
  `write_csv(path, keys=...)` overwrites the chosen destination.
  The legacy desktop CSV continues using its original display labels.

### Optional HTTP API and HTML GUI

```bash
python -m pip install ".[web,analysis]"
pyshape-web
# Equivalent:
python -m pyshape.web --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> to upload an STL, choose input units,
view grouped results, select descriptors, and download CSV. The interactive
mesh preview supports drag rotation, zoom, an XYZ triad, and auto-rotation.
For large meshes only the preview is sampled (at most 5000 triangles);
analysis always uses the full mesh.

The optional HTTP adapter accepts **self-contained STL files only**, up to
32 MiB. Other formats remain available through Python and the desktop viewer.
Uploads use temporary files that are removed after each request; no server-side
analysis sessions are retained. CSV export resubmits and analyzes the file.
Browser downloads follow the browser's save/duplicate-filename behavior rather
than writing alongside a file on the server.

Call it from another application:

```bash
curl -F "file=@particle.stl" -F "units=mm" \
  http://127.0.0.1:8000/api/analyze

curl -F "file=@particle.stl" -F "units=mm" \
  -F "keys=surface_area,volume" \
  http://127.0.0.1:8000/api/export -o particle_SHAPE.csv
```

- `POST /api/analyze`: multipart `file`, optional `units` (default `as-is`),
  optional `include_preview` (default `false`). Returns the same JSON data
  as the Python result, plus `preview` when requested.
- `POST /api/export`: multipart `file`, optional `units`, optional
  comma-separated `keys`. Returns CSV.
- `GET /api/health`: readiness response.
- `/docs`: generated interactive API documentation.
- Invalid data/selection returns 422, unsupported formats 415, and uploads
  exceeding the file-size limit 413. A missing union dependency returns 503.

For ASGI integration use `from pyshape.web import create_app`. The server
binds to loopback by default and is intended for local use. It has no
authentication, deployment-level request-body limits, or resource isolation.
Do not expose it directly to untrusted users; remote deployment needs
appropriate authentication, proxy limits, and workload isolation.

## Examples

Run from the repository root:

```bash
python examples/stl_area_and_orientation.py
python examples/tetra_volume_inertia.py
python examples/form_metrics_demo.py
python examples/shapes_gallery.py
```

The optional GUI viewer is in `examples/gui_shape_viewer.py`; it requires Tk and Matplotlib.

## Desktop GUI

A desktop viewer is available in `pyshape/gui.py` (requires Tk, Matplotlib, and
for non-STL formats or the convexity metric, `trimesh` and `scipy`):

```bash
python -m pip install ".[gui]"
pyshape-gui
# or
python -m pyshape.gui
# or
python examples/gui_shape_viewer.py
```

It lets you:

- **Load compatible geometry** — STL natively, plus OBJ, PLY, OFF, glTF/GLB,
  3MF and DAE via `trimesh`, which are transparently unified into the same
  triangle-mesh representation used for STL.
- **Review result parameters and tick which to export** — clicking
  "Calculate results" computes surface area, volume, centroid, Wadell and
  Krumbein sphericity, the surface orientation tensor (C, F, R), Potticary,
  Kong & Fonseca and Zingg form parameters, convexity, and PCA-estimated
  axes, each shown with a checkbox.
- **Export to CSV** — by default the file is written next to the input as
  `<name><suffix>.csv` (suffix defaults to `_SHAPE`, and is editable). Toggle
  "Avoid overwrite" to automatically append a progressive counter
  (`<name>_SHAPE_1.csv`, `_2.csv`, ...) instead of overwriting an existing
  export.
- **Preview the geometry** — a 3D view with an XYZ axes triad scaled to the
  mesh's bounding box and a slow auto-rotation (pauses while you drag to
  rotate manually).

## Tests

Install test and optional analysis/API dependencies, then run:

```bash
python -m pip install -e ".[mesh,analysis,web,dev]"
python -m pytest -q tests
```

Tests cover the public Python API, existing helpers, fallback STL loading,
unit conversion, explicit unavailable metrics, HTTP responses, CSV output,
and bundled web assets. CI runs the suite and builds the distribution, then
checks that the installed wheel imports and analyzes without UI dependencies.

## Building and publishing

The initial package version is `0.1.0`. To build wheel and source archives:

```bash
python -m pip install ".[dev]"
python -m build
```

The resulting archives are in `dist/`; the HTML/JS/CSS assets and GPL license
are included. Install a wheel in a fresh environment to validate a release.
Before publishing to PyPI, confirm ownership/availability of the distribution
name `pyshape` (change `project.name` if needed), update the version in
`pyproject.toml` and the API metadata, run tests, and create a release tag.
An import name and a distribution name can differ.

Once the branch is ready, publish it with `git push -u origin module`.
For a registry release, install Twine, run `python -m twine check dist/*`,
then upload to TestPyPI or PyPI using credentials configured outside this
repository. No branch push or registry publication is performed automatically.
Users distributing this derivative library must preserve its GPL licensing
and attribution.

## Citation and acknowledgement

Please cite the original SHAPE paper when using this derivative project:

Angelidakis, V., Nadimi, S. and Utili, S., 2021. [SHape Analyser for Particle Engineering (SHAPE): Seamless characterisation and simplification of particle morphology from imaging data](https://doi.org/10.1016/j.cpc.2021.107983). *Computer Physics Communications*, 265, 107983.

The BibTeX citation is available in [CITATION.bib](CITATION.bib). The original MATLAB project and its development are credited to Vasileios Angelidakis, Sadegh Nadimi, and Stefano Utili at [vsangelidakis/SHAPE](https://github.com/vsangelidakis/SHAPE).

The package also implements descriptors and methods reported by Wadell (1932), Krumbein (1941), Zingg (1935), Potticary et al. (2015), Kong and Fonseca (2018), and Bagi and Orosz (2020), as applicable.

## License

This is derivative work based on the original SHAPE project and is distributed under the GNU General Public License, version 3. See [LICENSE](LICENSE) for the license terms and attribution.
