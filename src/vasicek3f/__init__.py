"""3-factor Gaussian Vasicek (A0(3)) term structure model with Kalman-filter MLE
and an affine term premium decomposition. Educational project."""

from .affine import eh_loadings, p_loadings, q_loadings
from .decomposition import loadings_table, term_premium_decomposition
from .estimation import estimate, starting_values
from .model import build_state_space, filter_and_smooth, loglik
from .params import ParamTransform, VasicekParams
from .simulate import default_true_params, simulate_panel

__version__ = "1.2.0"

__all__ = [
    "ParamTransform",
    "VasicekParams",
    "build_state_space",
    "default_true_params",
    "eh_loadings",
    "estimate",
    "filter_and_smooth",
    "loadings_table",
    "loglik",
    "p_loadings",
    "q_loadings",
    "simulate_panel",
    "starting_values",
    "term_premium_decomposition",
]
