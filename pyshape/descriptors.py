"""Help texts for the results and controls of the pySHAPE interfaces.

Every result of :func:`pyshape.compute_metrics` has an entry in ``METRICS``,
keyed by its stable metric key (never by its label or group, which may
change). Each entry has:

``short``
    One line, shown as a tooltip.
``description``
    A few sentences: what the result measures and how to read it.
``formula``
    The definition in plain text ("" when there is none).
``depends_on``
    What the result needs: a closed mesh, the principal dimensions S, I, L
    (and so the bounding-box method), SciPy, ...
``references``
    Keys of ``REFERENCES``.

``CONTROLS`` holds the same fields (``formula`` and ``depends_on`` may be
empty) for the controls of the desktop and browser interfaces, keyed by a
control name. ``AXES_METHODS`` explains each choice of the principal-axes
control, keyed by the method name of :data:`pyshape.bounding.AXES_METHODS`;
``GROUPS`` gives an optional line for result groups. The browser interface
reads all of this from ``GET /api/descriptors`` (:func:`as_dict`).

The module has no NumPy or UI imports, so it can be used anywhere.
"""
from __future__ import annotations

from typing import Iterable, Optional

REFERENCES: dict[str, dict[str, str]] = {
    "shape2021": {
        "citation": "Angelidakis, V., Nadimi, S. and Utili, S., 2021. SHape Analyser for "
                    "Particle Engineering (SHAPE): Seamless characterisation and simplification "
                    "of particle morphology from imaging data. Computer Physics Communications, "
                    "265, 107983.",
        "url": "https://doi.org/10.1016/j.cpc.2021.107983",
    },
    "wadell1932": {"citation": "Wadell (1932).", "url": ""},
    "krumbein1941": {"citation": "Krumbein (1941).", "url": ""},
    "zingg1935": {"citation": "Zingg (1935).", "url": ""},
    "potticary2015": {"citation": "Potticary et al. (2015).", "url": ""},
    "kongfonseca2018": {"citation": "Kong and Fonseca (2018).", "url": ""},
    "angelidakis2022": {
        "citation": "Angelidakis et al. (2022). Powder Technology, 396, 689-695.",
        "url": "",
    },
    "bagiorosz2020": {
        "citation": "Bagi and Orosz (2020). ECMS 2020, 256-260.",
        "url": "https://doi.org/10.7148/2020-0256",
    },
    "orosz2021": {
        "citation": "Orosz, Angelidakis and Bagi (2021). Powder Technology, 394, 312-325.",
        "url": "https://doi.org/10.1016/j.powtec.2021.08.054",
    },
    "tonon2005": {"citation": "Tonon (2005).", "url": ""},
}

_BOX = ("The principal dimensions S, I, L, and so the method chosen under Principal axes "
        "(bounding box).")
_NO_BOX = "Not on any bounding box."
_ORIENTATION_NOTE = (
    " These indices come from the surface normals only; they are not the box-based "
    "flatness and elongation of Potticary et al., Kong & Fonseca or Angelidakis et al."
)


def _entry(short: str, description: str, formula: str = "", depends_on: str = "",
           references: tuple[str, ...] = ()) -> dict[str, object]:
    return {"short": short, "description": description, "formula": formula,
            "depends_on": depends_on, "references": list(references)}


METRICS: dict[str, dict[str, object]] = {
    "n_vertices": _entry(
        "Number of vertices of the analysed mesh.",
        "Vertices of the triangle mesh after preparation (orientation of closed shells and, "
        "for several shells, their solid union). It describes the mesh, not the particle.",
        depends_on="The mesh resolution only."),
    "n_faces": _entry(
        "Number of triangles of the analysed mesh.",
        "Triangles of the mesh after preparation (orientation of closed shells and, for "
        "several shells, their solid union). It describes the mesh, not the particle.",
        depends_on="The mesh resolution only."),
    "surface_area": _entry(
        "Total area of the triangles of the surface.",
        "Sum of the areas of all triangles of the analysed surface. For several overlapping "
        "shells, internal surfaces of the solid union are excluded. Available for open "
        "meshes too.",
        "A = Σ ½ |(b − a) × (c − a)| over the triangles (a, b, c)",
        _NO_BOX, ("shape2021",)),
    "volume": _entry(
        "Volume enclosed by the closed surface.",
        "Volume of the solid bounded by the surface, as the sum of signed tetrahedra joining "
        "each triangle to a reference point. It needs a closed, consistently oriented mesh; "
        "for open meshes it is unavailable.",
        "V = Σ (1/6) (a − o) · ((b − o) × (c − o)), o = reference point",
        "A closed mesh. " + _NO_BOX, ("shape2021",)),
    **{
        f"centroid_{axis}": _entry(
            f"{axis.upper()} coordinate of the volume centroid.",
            f"{axis.upper()} coordinate of the centre of volume of the solid, in the "
            "coordinate system of the file (converted to the output units). It needs a "
            "closed mesh.",
            "centroid = Σ V_t (a + b + c + o)/4 / V over the signed tetrahedra t",
            "A closed mesh and the position of the particle in the file.",
            ("shape2021",))
        for axis in "xyz"
    },
    **{
        f"inertia_I{i}": _entry(
            f"Principal moment of inertia I{i} per unit density ({order}).",
            "Eigenvalues of the inertia tensor of the solid about its centroid, for a "
            "density of 1 (multiply by the density for the mass moments), sorted I1 ≤ I2 ≤ I3. "
            "Computed exactly from signed tetrahedra on the closed surface. Units: "
            "length to the fifth power.",
            "I = ∫ (|r|² 1 − r rᵀ) dV about the centroid; I1 ≤ I2 ≤ I3 are its eigenvalues",
            "A closed mesh. " + _NO_BOX, ("tonon2005", "shape2021"))
        for i, order in ((1, "smallest"), (2, "intermediate"), (3, "largest"))
    },
    "sphericity_wadell": _entry(
        "Surface area of the sphere of equal volume divided by the particle's surface area.",
        "True sphericity: 1 for a sphere and smaller for any other shape. It uses only the "
        "volume and surface area, so it needs a closed mesh but no bounding box. It is "
        "sensitive to surface roughness and to mesh noise, which increase the area.",
        "Ψ = π^(1/3) (6V)^(2/3) / A",
        "A closed mesh. " + _NO_BOX, ("wadell1932", "shape2021")),
    "convexity": _entry(
        "Volume of the particle divided by the volume of its convex hull.",
        "1 for a convex particle, smaller when the particle has concavities. The convex hull "
        "is the smallest convex solid that contains every vertex.",
        "Convexity = V / V_convex_hull",
        "A closed mesh and SciPy (pyshape[analysis]). " + _NO_BOX, ("shape2021",)),
    "orientation_C": _entry(
        "Compactness C from the surface orientation tensor (1 for a sphere).",
        "The surface orientation tensor F = (1/A) Σ A_t n_t n_tᵀ averages the directions of "
        "the unit face normals n_t, weighted by triangle area A_t. With its eigenvalues "
        "f1 ≥ f2 ≥ f3, compactness C is near 1 when normals point evenly in all directions "
        "(sphere, cube and any shape with cubic or icosahedral symmetry) and small for flat "
        "or elongated particles. C + F + E = 1." + _ORIENTATION_NOTE,
        "C = f3 / f1",
        _NO_BOX + " Only face normals and areas; refining the mesh does not bias it.",
        ("bagiorosz2020", "orosz2021")),
    "orientation_F": _entry(
        "Flakiness F from the surface orientation tensor (large for flat particles).",
        "With the eigenvalues f1 ≥ f2 ≥ f3 of the surface orientation tensor "
        "F = (1/A) Σ A_t n_t n_tᵀ, flakiness is large when most of the surface faces one "
        "direction, as for a flat particle. It is 0 for a sphere. C + F + E = 1."
        + _ORIENTATION_NOTE,
        "F = (f1 − f2) / f1",
        _NO_BOX + " Only face normals and areas.", ("bagiorosz2020", "orosz2021")),
    "orientation_E": _entry(
        "Elongation E from the surface orientation tensor (large for rod-like particles).",
        "With the eigenvalues f1 ≥ f2 ≥ f3 of the surface orientation tensor "
        "F = (1/A) Σ A_t n_t n_tᵀ, elongation is large when the normals lie mostly in one "
        "plane, as for a rod. It is 0 for a sphere. C + F + E = 1." + _ORIENTATION_NOTE,
        "E = (f2 − f3) / f1",
        _NO_BOX + " Only face normals and areas.", ("bagiorosz2020", "orosz2021")),
    **{
        f"orientation_eig{i}": _entry(
            f"Eigenvalue f{i} of the surface orientation tensor ({order}).",
            "Eigenvalues f1 ≥ f2 ≥ f3 of the surface orientation tensor "
            "F = (1/A) Σ A_t n_t n_tᵀ (n_t unit normal, A_t area of triangle t). They sum "
            "to 1; for a sphere each is 1/3. Compactness, flakiness and elongation are "
            "built from them.",
            "f1 ≥ f2 ≥ f3, f1 + f2 + f3 = 1",
            _NO_BOX + " Only face normals and areas.", ("bagiorosz2020", "orosz2021"))
        for i, order in ((1, "largest"), (2, "intermediate"), (3, "smallest"))
    },
    **{
        f"axis_{axis}": _entry(
            f"{name} principal dimension {axis} (S ≤ I ≤ L).",
            f"{name} edge of the oriented bounding box chosen under Principal axes; the box "
            "edges sorted by length are S ≤ I ≤ L. Every index computed from S, I, L "
            "(Krumbein sphericity, Zingg ratios, flatness and elongation) inherits the "
            "method.",
            "S ≤ I ≤ L: edge lengths of the oriented box",
            _BOX, ("shape2021",))
        for axis, name in (("S", "Short"), ("I", "Intermediate"), ("L", "Long"))
    },
    "sphericity_krumbein": _entry(
        "Sphericity from the principal dimensions: (I·S / L²)^(1/3).",
        "Krumbein's intercept sphericity: 1 when S = I = L and smaller for flat or elongated "
        "particles. It uses only the principal dimensions, not the surface.",
        "ψ_K = (I S / L²)^(1/3)",
        _BOX, ("krumbein1941", "shape2021")),
    "potticary_flatness": _entry(
        "Flatness of Potticary et al.: 2(I − S) / (L + I + S).",
        "Flatness index from the principal dimensions; 0 when S = I.",
        "Flatness = 2 (I − S) / (L + I + S)",
        _BOX, ("potticary2015", "shape2021")),
    "potticary_elongation": _entry(
        "Elongation of Potticary et al.: (L − I) / (L + I + S).",
        "Elongation index from the principal dimensions; 0 when I = L.",
        "Elongation = (L − I) / (L + I + S)",
        _BOX, ("potticary2015", "shape2021")),
    "kongfonseca_flatness": _entry(
        "Flatness of Kong & Fonseca: (I − S) / I.",
        "Flatness index from the principal dimensions, between 0 (S = I) and 1.",
        "Flatness = (I − S) / I",
        _BOX, ("kongfonseca2018", "shape2021")),
    "kongfonseca_elongation": _entry(
        "Elongation of Kong & Fonseca: (L − I) / L.",
        "Elongation index from the principal dimensions, between 0 (I = L) and 1.",
        "Elongation = (L − I) / L",
        _BOX, ("kongfonseca2018", "shape2021")),
    "angelidakis_flatness": _entry(
        "Flatness of Angelidakis et al.: I² / (L·S + I²) − S / (L + S).",
        "Flatness index from the principal dimensions, in [0, 1]. With elongation it maps a "
        "particle to a point of a triangular chart (flatness + elongation = (L − S)/(L + S) "
        "≤ 1); a sphere is at the origin.",
        "Flatness = I² / (L S + I²) − S / (L + S)",
        _BOX, ("angelidakis2022",)),
    "angelidakis_elongation": _entry(
        "Elongation of Angelidakis et al.: L·S / (L·S + I²) − S / (L + S).",
        "Elongation index from the principal dimensions, in [0, 1]. With flatness it maps a "
        "particle to a point of a triangular chart (flatness + elongation = (L − S)/(L + S) "
        "≤ 1); a sphere is at the origin.",
        "Elongation = L S / (L S + I²) − S / (L + S)",
        _BOX, ("angelidakis2022",)),
    "zingg_SI": _entry(
        "Zingg ratio S/I (small for flat particles).",
        "Ratio of the short to the intermediate dimension. With I/L it places the particle in "
        "one of Zingg's four classes (discs, spheres, blades, rods), with the boundaries "
        "at 2/3.",
        "S / I",
        _BOX, ("zingg1935", "shape2021")),
    "zingg_IL": _entry(
        "Zingg ratio I/L (small for elongated particles).",
        "Ratio of the intermediate to the long dimension. With S/I it places the particle in "
        "one of Zingg's four classes (discs, spheres, blades, rods), with the boundaries "
        "at 2/3.",
        "I / L",
        _BOX, ("zingg1935", "shape2021")),
}

AXES_METHODS: dict[str, dict[str, object]] = {
    "pca": _entry(
        "Box aligned with the principal directions of the vertices (fast).",
        "S, I, L are the extents of the vertices along the principal directions of their "
        "covariance (SHAPE's PCA_points option). These directions are not unique for shapes "
        "with isotropic vertex covariance (cubes, regular tetrahedra, octahedra, "
        "icosahedra, spheres), so the box then depends on how the mesh is oriented.",
        depends_on="NumPy only.", references=("shape2021",)),
    "min_volume": _entry(
        "Minimum-volume oriented bounding box (orientation-independent, slower).",
        "S, I, L are the edges of the smallest-volume box that contains the particle "
        "(SHAPE's default minVolume option). Candidate orientations come from the convex "
        "hull (facet normals, edge directions) and are refined locally. The cost grows "
        "roughly quadratically with the number of hull vertices.",
        depends_on="SciPy (pyshape[analysis]).", references=("shape2021",)),
    "inertia_ellipsoid": _entry(
        "Ellipsoid with the same inertia as the particle.",
        "S, I, L are the axes of the ellipsoid that has the same principal moments of inertia "
        "as the particle. It needs a closed mesh.",
        depends_on="A closed mesh."),
    "fitted_ellipsoid": _entry(
        "Ellipsoid fitted to the surface by least squares.",
        "S, I, L are the axes of the ellipsoid fitted to the surface points by least squares.",
        depends_on="The surface points."),
}

CONTROLS: dict[str, dict[str, object]] = {
    "file": _entry(
        "Mesh file of the particle.",
        "STL (ASCII or binary) is read by pySHAPE's built-in parser. OBJ, PLY, OFF, GLB, 3MF "
        "and DAE (and glTF on the desktop) are read by trimesh (pyshape[mesh]). Every mesh "
        "becomes a triangle mesh; closed shells are oriented and several shells are combined "
        "as a solid union."),
    "input_units": _entry(
        "Units of the coordinates in the file.",
        "Declares what one unit of the file's coordinates means (µm, mm, cm, m or inch) so "
        "that results can be converted. As-is keeps the coordinates unchanged: results are "
        "then in the file's own units and the output units are ignored."),
    "output_units": _entry(
        "Length unit of the results.",
        "Lengths are given in this unit; areas in its square, volumes in its cube and inertia "
        "per unit density in its fifth power (e.g. mm, mm², mm³, mm⁵). Same as input follows "
        "the input units. Disabled when the input units are as-is. Dimensionless results are "
        "unaffected."),
    "axes": _entry(
        "How the principal dimensions S, I, L are measured.",
        "Chooses the oriented bounding box whose edges give S ≤ I ≤ L. It affects only the "
        "principal dimensions and the indices computed from them (Krumbein sphericity, Zingg "
        "ratios, flatness and elongation); the other results do not depend on it.",
        references=("shape2021",)),
    "strict_stl": _entry(
        "Read STL files only with the built-in parser.",
        "Without it, an STL file that the built-in parser rejects is read by trimesh when it "
        "is installed, with a warning. With it, the parser's error is raised instead."),
    "load": _entry(
        "Read the file and show the mesh.",
        "Reads the file with the chosen input units, prepares the mesh (orientation and "
        "solid union) and shows the preview. Results are computed separately."),
    "calculate": _entry(
        "Compute all results for the mesh.",
        "Computes every result with the chosen principal-axes method and output units. "
        "Results that cannot be computed (e.g. volume of an open mesh) are shown as n/a "
        "with the reason."),
    "results": _entry(
        "Results of the particle; tick the ones to export.",
        "Results are grouped. Hover over a result for a one-line description and click its "
        "name for the definition, formula and references. Ticked results are exported."),
    "select_all": _entry("Tick every available result for export.",
                         "Ticks all results that have a value."),
    "select_none": _entry("Untick every result.", "Unticks all results."),
    "export": _entry(
        "Save the ticked results as CSV.",
        "Writes one row per ticked result with its value and unit. The first rows name the "
        "source file and its reader."),
    "suffix": _entry(
        "Text added to the file name of the CSV.",
        "The CSV is named after the mesh file plus this suffix, e.g. particle_SHAPE.csv."),
    "avoid_overwrite": _entry(
        "Append _1, _2, ... instead of overwriting an existing CSV.",
        "When a CSV with the same name exists, a progressive number is appended to the new "
        "file name."),
    "auto_rotate": _entry(
        "Rotate the preview slowly.",
        "Turns the slow rotation of the 3D preview on or off. Dragging rotates the view by "
        "hand. The preview is for display only."),
    "folder": _entry(
        "Folder of mesh files to analyse.",
        "Every mesh file in the folder is analysed with the same settings; other files are "
        "ignored."),
    "files": _entry("Individual mesh files to analyse.",
                    "Choose one or more mesh files instead of a folder."),
    "recursive": _entry("Also analyse mesh files in subfolders.",
                        "Includes mesh files in the subfolders of the chosen folder."),
    "summary": _entry(
        "Also make one CSV with one row per particle.",
        "Adds pyshape_summary.csv: one row per particle and one column per result exported "
        "by default."),
    "run": _entry("Analyse every chosen file.",
                  "Analyses the files one after the other and makes one CSV per particle."),
    "stop": _entry("Stop after the file being analysed.",
                   "The files analysed so far stay available for download."),
    "download_all": _entry(
        "Download every CSV in one zip file.",
        "The zip holds one CSV per particle, named after its mesh file and in the same "
        "subfolder."),
}

# Optional one-line help for result groups. Group names may carry the axes
# method in parentheses, e.g. "Principal dimensions (PCA)": the part before
# " (" is looked up too. A group without an entry simply has no help.
GROUPS: dict[str, str] = {
    "Mesh": "The triangle mesh that is analysed, after preparation.",
    "Size and mass properties": "Surface area, volume, centroid and inertia of the solid.",
    "Form, surface-based (independent of the box)":
        "Form indices from the surface itself; no bounding box is needed.",
    "Principal dimensions":
        "Short, intermediate and long dimensions S ≤ I ≤ L of the chosen bounding box.",
    "Form from S, I, L":
        "Form indices computed only from the principal dimensions S, I, L.",
}


def metric_help(key: str) -> Optional[dict[str, object]]:
    """Help entry of a metric key, or None."""
    return METRICS.get(key)


def control_help(name: str) -> Optional[dict[str, object]]:
    """Help entry of a control name, or None."""
    return CONTROLS.get(name)


def axes_help(method: str) -> Optional[dict[str, object]]:
    """Help entry of a principal-axes method, or None."""
    return AXES_METHODS.get(method)


def group_help(name: str) -> str:
    """One line of help for a result group, or "" if it has none."""
    if name in GROUPS:
        return GROUPS[name]
    return GROUPS.get(name.split(" (", 1)[0], "")


def explanation(entry: dict[str, object], title: str = "",
                extra: Iterable[str] = ()) -> str:
    """Plain-text long explanation of an entry, e.g. for a help window.

    ``extra`` paragraphs go after the definition and before the references.
    """
    parts = [title] if title else []
    parts.append(str(entry["description"]))
    if entry.get("formula"):
        parts.append(f"Formula: {entry['formula']}")
    if entry.get("depends_on"):
        parts.append(f"Depends on: {entry['depends_on']}")
    parts.extend(extra)
    references = [REFERENCES[key] for key in entry.get("references", ())]
    if references:
        parts.append("References:\n" + "\n".join(
            f"- {ref['citation']}" + (f" {ref['url']}" if ref["url"] else "")
            for ref in references))
    return "\n\n".join(parts)


def axes_control_explanation(methods: Iterable[str], title: str = "") -> str:
    """Explanation of the principal-axes control, with one section per method.

    Methods without a help entry are listed by name only.
    """
    sections = []
    for method in methods:
        entry = AXES_METHODS.get(method)
        if entry is None:
            sections.append(method)
            continue
        section = f"{method}: {entry['short']}\n{entry['description']}"
        if entry.get("depends_on"):
            section += f" Depends on: {entry['depends_on']}"
        sections.append(section)
    return explanation(CONTROLS["axes"], title, sections)


def as_dict() -> dict[str, object]:
    """Every help text as JSON-compatible data (served at ``/api/descriptors``)."""
    return {"metrics": METRICS, "controls": CONTROLS, "axes_methods": AXES_METHODS,
            "groups": GROUPS, "references": REFERENCES}
