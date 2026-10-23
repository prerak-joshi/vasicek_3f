"""3-factor Gaussian Vasicek (A0(3)) term structure model with Kalman-filter MLE
and an affine term premium decomposition. Educational project."""
from .params import VasicekParams, ParamTransform
from .affine import q_loadings, p_loadings, eh_loadings
from .model import loglik, filter_and_smooth, build_state_space
from .estimation import estimate, starting_values
from .decomposition import term_premium_decomposition, loadings_table
from .simulate import default_true_params, simulate_panel

__version__ = "1.0.0"
