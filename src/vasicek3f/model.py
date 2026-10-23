"""
Glue between the economic model and the state-space machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .affine import q_loadings
from .discretize import discretize_ou, unconditional_moments
from .kalman import kalman_filter, kalman_loglik_collapsed, rts_smoother, KalmanResult
from .params import VasicekParams


@dataclass
class StateSpace:
    a: np.ndarray
    H: np.ndarray
    R: np.ndarray
    c: np.ndarray
    F: np.ndarray
    Q: np.ndarray
    x0: np.ndarray
    P0: np.ndarray


def build_state_space(params: VasicekParams, maturities: np.ndarray, dt: float) -> StateSpace:
    """Measurement eq. from Q pricing, transition eq. from exact P discretisation."""
    a, H = q_loadings(params, maturities)
    R = np.diag(params.sigma_e ** 2)
    c, F, Q = discretize_ou(params.K_P, params.theta_P, params.Sigma, dt)
    x0, P0 = unconditional_moments(params.theta_P, F, Q)
    return StateSpace(a, H, R, c, F, Q, x0, P0)


def loglik(params: VasicekParams, y: np.ndarray, maturities: np.ndarray, dt: float) -> float:
    """Exact Gaussian log-likelihood of the yield panel (-inf if invalid)."""
    if not params.is_stationary_P():
        return -np.inf
    try:
        ss = build_state_space(params, maturities, dt)
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf
    if not np.all(np.isfinite(ss.P0)) or np.any(np.linalg.eigvalsh(ss.P0) <= 0):
        return -np.inf
    if not np.isnan(y).any():
        # exact and ~3-5x faster: filter on the 3-dim GLS projection of the yields
        return kalman_loglik_collapsed(y, ss.a, ss.H, np.diag(ss.R), ss.c, ss.F, ss.Q,
                                       ss.x0, ss.P0)
    res = kalman_filter(y, ss.a, ss.H, ss.R, ss.c, ss.F, ss.Q, ss.x0, ss.P0, store=False)
    return res.loglik


def filter_and_smooth(params: VasicekParams, y: np.ndarray, maturities: np.ndarray,
                      dt: float) -> Dict[str, object]:
    """Run filter + smoother at given parameters; return states and fitted yields."""
    ss = build_state_space(params, maturities, dt)
    kf: KalmanResult = kalman_filter(y, ss.a, ss.H, ss.R, ss.c, ss.F, ss.Q, ss.x0, ss.P0)
    xs, Ps = rts_smoother(kf, ss.F)
    return {
        "state_space": ss,
        "kalman": kf,
        "x_filtered": kf.x_filt,
        "x_smoothed": xs,
        "P_smoothed": Ps,
        "fitted_filtered": ss.a[None, :] + kf.x_filt @ ss.H.T,
        "fitted_smoothed": ss.a[None, :] + xs @ ss.H.T,
    }
