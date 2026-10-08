# SHAPE Python

This repository contains a lightweight Python port of selected functionality from SHAPE, the SHape Analyser for Particle Engineering. It is derivative work based on the original project by Vasileios Angelidakis, Sadegh Nadimi, and Stefano Utili. Visit the [original SHAPE GitHub repository](https://github.com/vsangelidakis/SHAPE) for the full MATLAB toolbox and project history.

The Python package provides tools for 3D particle geometry and form analysis. It does not currently include all features of the original MATLAB toolbox.

## Features

- Geometry: triangle-mesh surface area and tetrahedral-mesh volume, centroid, and inertia
- Form: convexity, Wadell and Krumbein sphericities, Zingg ratios, and Potticary and Kong-Fonseca flatness/elongation parameters
- Surface orientation tensor
- Surface roughness statistics
- STL loading, with `trimesh` support and a built-in fallback

## Project layout

- `pyshape/` — Python package
- `examples/` — runnable demos and a bundled ASCII STL asset
- `tests/` — unit tests
- `requirements.txt` — runtime and test dependencies

## Requirements and installation

Python 3.9 or later is required. Install dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

## Quick start

```python
import numpy as np
from pyshape import surface_area, surface_orientation_tensor

nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

print(surface_area(nodes, faces))
C, F, R, eigenvalues, eigenvectors = surface_orientation_tensor(nodes, faces)
```

Mesh face and tetrahedron indices are normally zero-based. One-based MATLAB-style indices are automatically converted where supported.

## Examples

Run from the repository root:

```bash
python examples/stl_area_and_orientation.py
python examples/tetra_volume_inertia.py
python examples/form_metrics_demo.py
python examples/shapes_gallery.py
```

The optional GUI viewer is in `examples/gui_shape_viewer.py`; it requires Tk and Matplotlib.

## Tests

Run the tests from the repository root:

```bash
python -m pytest -q tests
```

## Citation and acknowledgement

Please cite the original SHAPE paper when using this derivative project:

Angelidakis, V., Nadimi, S. and Utili, S., 2021. [SHape Analyser for Particle Engineering (SHAPE): Seamless characterisation and simplification of particle morphology from imaging data](https://doi.org/10.1016/j.cpc.2021.107983). *Computer Physics Communications*, 265, 107983.

The BibTeX citation is available in [CITATION.bib](CITATION.bib). The original MATLAB project and its development are credited to Vasileios Angelidakis, Sadegh Nadimi, and Stefano Utili at [vsangelidakis/SHAPE](https://github.com/vsangelidakis/SHAPE).

The package also implements descriptors and methods reported by Wadell (1932), Krumbein (1941), Zingg (1935), Potticary et al. (2015), Kong and Fonseca (2018), and Bagi and Orosz (2020), as applicable.

## License

This is derivative work based on the original SHAPE project and is distributed under the GNU General Public License, version 3. See [LICENSE](LICENSE) for the license terms and attribution.
