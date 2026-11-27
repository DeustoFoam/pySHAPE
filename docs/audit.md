# Code audit: checking that existing behaviour is preserved

`tools/audit.py` is a development tool for checking that a change, for example one made with an AI coding assistant, has not silently changed what pySHAPE already does. It complements the unit tests:

| | Unit tests (`tests/`) | Audit (`tools/audit.py`) |
| --- | --- | --- |
| Checks | chosen cases against exact or expected answers | everything public against a recorded baseline |
| Notices | wrong results in the cases tested | a function removed or renamed, a parameter renamed or reordered, a changed default, a removed CLI option or HTTP route, any change in the results of the reference meshes |
| Updated | when a test is written | when a change is *intended* (`snapshot`) |

The tool is not installed with the package. Run it from a clone of the repository, in an environment set up with:

```bash
pip install -e ".[mesh,analysis,web,dev]"
```

## Commands

| Command | What it does |
| --- | --- |
| `python tools/audit.py list` | Inventory of the code. |
| `python tools/audit.py check` | Compares the code with the baseline. |
| `python tools/audit.py tests` | Runs the unit tests and then `check`, and prints one verdict. |
| `python tools/audit.py snapshot` | Records a new baseline after an intended change. |

### `list`: inventory of the code

```bash
python tools/audit.py list                    # public functions, classes and methods
python tools/audit.py list --all              # private helpers (leading underscore) too
python tools/audit.py list --module analysis  # one module
python tools/audit.py list --signatures       # parameters next to the names
python tools/audit.py list --format md        # Markdown table (or --format json)
```

For each definition, the table shows:

| Column | Meaning |
| --- | --- |
| location | `file:line` |
| public | Name has no leading underscore. |
| exported | Importable as `pyshape.<name>`, i.e. listed in `pyshape.__all__`. |
| doc | Has a docstring. |
| tested | The name appears in `tests/`. This is a static check: the name is used, but that does not prove every branch is covered. |
| deprecated | Why the definition counts as deprecated: a `@deprecated` decorator, a `warnings.warn(..., DeprecationWarning)` call (or `PendingDeprecationWarning`, `FutureWarning`), or a docstring that says it is deprecated. |

Below the table it lists:
- public library definitions that no test uses;
- public definitions without a docstring;
- deprecated definitions;
- backward-compatibility modules, such as `gui_core`.

The inventory parses the source files and does not import them, so it works without the optional extras.

### `check`: compare the code with the baseline

```bash
python tools/audit.py check            # exit status 1 if existing behaviour changed
python tools/audit.py check --strict   # also fail on warnings
python tools/audit.py check --rtol 1e-6 --atol 1e-12
```

Two things are compared.

1. **The interface**: every public function, class and method, with:
   - parameter names, kinds (positional, keyword-only, ...), order and defaults;
   - dataclass fields;
   - the names exported from `pyshape`;
   - the names re-exported by backward-compatibility modules;
   - the options of the console scripts (`pyshape-batch`, `pyshape-web`, `pyshape-gui`);
   - the HTTP routes of the server.

   This baseline is `tools/baseline/api.json`.
2. **The results**: the complete result record (all metrics, their units, labels and groups, and metadata such as the reader and units) for a fixed set of reference meshes:
   - cube and regular tetrahedron;
   - a rotated 4×2×1 cuboid and a rotated 3:2:1 ellipsoid;
   - a rough, non-convex particle;
   - a torus;
   - two overlapping spheres (solid union);
   - the STL files in `examples/assets/`;
   - the synthetic files with nested shells in `examples/assets/synthetic/` (`hollow_sphere`, `cube_with_spherical_void`, `sphere_with_two_cavities`, `nested_island`), analysed with `cavities="keep"`.

   Each mesh is analysed with the four principal-dimension methods (`pca`, `min_volume`, `inertia_ellipsoid`, `fitted_ellipsoid`); the files with nested shells with `pca` and `min_volume`, and the cuboid also has a unit-conversion case. The CSV layout of a file result and the header of the batch summary are compared too. This baseline is `tools/baseline/results.json`.

   The meshes are generated in code (NumPy only, fixed seeds), so the check is reproducible on any machine. Values are compared with a relative tolerance of 1e-9 and an absolute one of 1e-12, which absorbs round-off differences between platforms. For meshes whose principal directions are not unique (cube, regular tetrahedron, torus, ...), the PCA box depends on round-off, so the results derived from it are not compared for the `pca` cases. This is judged on the outer envelope of the prepared mesh, on which the PCA box is measured (with `cavities="keep"`, the cavity walls do not count).

Each finding has a level:

| Level | Meaning | Fails `check`? |
| --- | --- | --- |
| `BREAKING` | Existing code that used pySHAPE may now fail or give different numbers. Examples: a removed or renamed function or parameter, a parameter that became required, a changed default or parameter order, a removed export, CLI option or HTTP route, a result that is missing or changed beyond tolerance, a changed unit, a changed CSV layout, an analysis that now raises. | yes |
| `WARN` | Something changed in the internals of the GUI or server (`gui.py`, `web.py`), or a module no longer imports a name from another module. Their CLI options and HTTP routes are still `BREAKING`. | with `--strict` |
| `NOTICE` | A change in type annotations, labels, groups or error messages, or a newly deprecated definition. | no |
| `ADDED` | New functions, optional parameters, CLI options, routes or results. These do not affect existing users. | no |
| `SKIPPED` | A reference case needs an extra that is not installed (SciPy, or manifold3d for the solid union). | no |

### `tests`: unit tests and audit together

```bash
python tools/audit.py tests
```

This runs `python -m pytest -q tests` and then `check`, and prints one verdict. Use it before committing, and after any AI-assisted change.

The test suite also contains `tests/test_audit.py`, which runs `check`, so `pytest` and the GitHub CI fail on a breaking change too.

### `snapshot`: record a new baseline after an intended change

```bash
python tools/audit.py snapshot         # refuses if the change is breaking
python tools/audit.py snapshot --yes   # accept breaking changes
```

`snapshot` first runs `check` and shows what would be accepted. It refuses to write the baseline if a change is breaking, unless you pass `--yes`. Run it only when a change is intended, for example a bug fix that changes a value or a deliberately renamed parameter, then commit `tools/baseline/` together with the change. The diff of `tools/baseline/*.json` in the pull request then shows exactly which signatures or values changed, and reviewers can judge whether that was intended.

SciPy and manifold3d must be installed to record a snapshot, so that every reference case is included.

## Typical workflow

1. Before a change: `python tools/audit.py tests` should report `OK`.
2. Make the change, by hand or with an assistant.
3. Run `python tools/audit.py tests` again.
   - If it reports `OK`, existing behaviour is preserved, and any `ADDED` lines show the new things.
   - If it reports `BREAKING`, either fix the change, or, if the change is intended, run `snapshot --yes`. Record the reason in `CHANGELOG.md`, and commit the baseline with the change.
4. To review what exists and what is untested, use `python tools/audit.py list`.

## Instructions for AI assistants

- Run `python tools/audit.py tests` before and after a change.
- Never run `snapshot --yes` just to make the check pass. Do it only if the user has agreed that the change is intended, and say which findings it accepts.
- New public functions need a docstring and a test, so that `list` shows them as documented and tested.
- Deprecate before removing:
  - keep the old name or parameter working;
  - make it emit `warnings.warn(..., DeprecationWarning, stacklevel=2)`;
  - say so in the docstring and in `CHANGELOG.md`.
