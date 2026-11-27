"""Particle analysis as a Python library; UI dependencies are optional."""

__version__ = "0.2.1"

from .analysis import AnalysisResult, Metric, analyze_file, analyze_mesh, compute_metrics
from .batch import BatchItem, BatchResult, analyze_folder, find_particle_files
from .bounding import AXES_METHODS, OrientedBox, min_volume_box, oriented_box, pca_box
from .ellipsoid import ELLIPSOID_METHODS, Ellipsoid, fitted_ellipsoid, inertia_ellipsoid
from .geometry import closed_surface_inertia_tensor, surface_area, volume_centroid_inertia_tensor
from .form import (
	convexity,
	sphericity_wadell,
	sphericity_krumbein,
	surface_orientation_tensor,
	form_functions_1,
	form_functions_2,
	form_parameters_kong_and_fonseca,
	form_parameters_potticary_et_al,
	form_parameters_angelidakis_et_al,
	form_parameters_zingg,
)
from .io import LoadedGeometry, load_geometry, load_stl, write_stl
from .preprocess import CAVITY_MODES, PreparedSolid, prepare_mesh, prepare_solid
from .repair import MeshDiagnosis, RepairReport, diagnose_mesh, repair_mesh
from .roughness import sa, sq, sdq, sku, ssk, roughness_functions

__all__ = [
	"AnalysisResult",
	"Metric",
	"LoadedGeometry",
	"analyze_file",
	"analyze_folder",
	"find_particle_files",
	"BatchItem",
	"BatchResult",
	"analyze_mesh",
	"compute_metrics",
	"prepare_mesh",
	"prepare_solid",
	"PreparedSolid",
	"CAVITY_MODES",
	"load_geometry",
	"AXES_METHODS",
	"OrientedBox",
	"min_volume_box",
	"oriented_box",
	"pca_box",
	"ELLIPSOID_METHODS",
	"Ellipsoid",
	"inertia_ellipsoid",
	"fitted_ellipsoid",
	"surface_area",
	"volume_centroid_inertia_tensor",
	"closed_surface_inertia_tensor",
	"convexity",
	"sphericity_wadell",
	"sphericity_krumbein",
	"surface_orientation_tensor",
	"form_functions_1",
	"form_functions_2",
	"form_parameters_kong_and_fonseca",
	"form_parameters_potticary_et_al",
	"form_parameters_angelidakis_et_al",
	"form_parameters_zingg",
	"load_stl",
	"write_stl",
	"diagnose_mesh",
	"repair_mesh",
	"MeshDiagnosis",
	"RepairReport",
	"sa",
	"sq",
	"sdq",
	"sku",
	"ssk",
	"roughness_functions",
]
