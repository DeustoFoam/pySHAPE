# SHAPE — MATLAB and Python

Unified repository for SHape Analyser for Particle Engineering (SHAPE) with:
- MATLAB (original, full-featured) implementation in `Matlab/`
- Python (lightweight port) implementation in `python/`

Both versions provide tools for 3D particle geometry and form characterisation: surface area, volume/centroid/inertia, sphericities, Zingg ratios, flatness/elongation metrics, and orientation tensors. See the subfolder READMEs for details.

## Folder structure

- `Matlab/` — Original SHAPE toolbox: classes, functions, examples, figures, and bundled third‑party libs.
- `python/` — Python port: package `pyshape/`, examples, and tests.

## Quick start

### MATLAB

1) Open MATLAB and clone/open this repository.
2) Explore the examples in `Matlab/examples/` (e.g. load meshes, point clouds, voxel images).

Minimal usage from MATLAB:

```matlab
addpath(genpath('Matlab/functions'));
addpath(genpath('Matlab/lib'));
addpath('Matlab/classes');
run('Matlab/examples/Example_1abc_Load_PointCloud_SurfaceMesh_Voxelated_Image.m');
```

More info: `Matlab/README.md`.

### Python

Requirements: Python 3.9+, NumPy, PyTest; `trimesh` recommended.

Install dependencies (run from repo root):

```bash
python -m pip install -r python/requirements.txt
```

Run examples (from repo root):

```bash
python python/examples/stl_area_and_orientation.py
python python/examples/tetra_volume_inertia.py
python python/examples/form_metrics_demo.py
```

Run tests (from repo root):

```bash
python -m pytest -q python/tests
```

Package code: see `python/pyshape/*` and `python/README.md`.

## Citation

The original SHAPE project was created by Vasileios Angelidakis. This repository builds on that work and includes a Python port. See the [original SHAPE GitHub repository](https://github.com/vsangelidakis/SHAPE).

If you use SHAPE, please cite the original paper. The BibTeX entry is available in `Matlab/CITATION.bib`.

Angelidakis, V., Nadimi, S. and Utili, S., 2021. [SHape Analyser for Particle Engineering (SHAPE): Seamless characterisation and simplification of particle morphology from imaging data](https://doi.org/10.1016/j.cpc.2021.107983). *Computer Physics Communications*, 265, 107983.

## License

See `Matlab/LICENSE` for licensing terms that apply to the repository contents and bundled third‑party code where noted.

## Contributing

Contributions and issues are welcome for both MATLAB and Python code. Keep parity notes in PRs when changing algorithms shared across implementations.
