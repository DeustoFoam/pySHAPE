"""Particle analysis as a Python library; UI dependencies are optional."""

from .analysis import AnalysisResult, Metric, analyze_file, analyze_mesh, compute_metrics
from .geometry import surface_area, volume_centroid_inertia_tensor
from .form import (
	convexity,
	sphericity_wadell,
	sphericity_krumbein,
	surface_orientation_tensor,
	form_functions_1,
	form_functions_2,
	form_parameters_kong_and_fonseca,
	form_parameters_potticary_et_al,
	form_parameters_zingg,
)
from .io import LoadedGeometry, load_geometry, load_stl
from .roughness import sa, sq, sdq, sku, ssk, roughness_functions

__all__ = [
	"AnalysisResult",
	"Metric",
	"LoadedGeometry",
	"analyze_file",
	"analyze_mesh",
	"compute_metrics",
	"load_geometry",
	"surface_area",
	"volume_centroid_inertia_tensor",
	"convexity",
	"sphericity_wadell",
	"sphericity_krumbein",
	"surface_orientation_tensor",
	"form_functions_1",
	"form_functions_2",
	"form_parameters_kong_and_fonseca",
	"form_parameters_potticary_et_al",
	"form_parameters_zingg",
	"load_stl",
	"sa",
	"sq",
	"sdq",
	"sku",
	"ssk",
	"roughness_functions",
]
