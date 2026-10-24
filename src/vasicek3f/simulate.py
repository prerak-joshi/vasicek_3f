"""
Simulate a yield panel from known parameters (for testing and parameter recovery).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .affine import p_loadings, q_loadings
from .discretize import discretize_ou, unconditional_moments
from .params import VasicekParams


def default_true_params(n_maturities: int = 10) -> VasicekParams:
    """
    A realistic parameterisation (decimals, annual units):
      * slow 'level' factor, faster 'slope' and 'curvature' factors under Q
      * Q long-run short rate 6.0%, P long-run short rate 4.0% -> positive average TP
      * strongly correlated factor shocks, as is typical in canonical form
      * 4bp measurement error
    """
    return VasicekParams(
        kappa_Q=[0.06, 0.45, 1.40],
        delta0=0.060,
        Sigma=[[0.0080, 0.0, 0.0], [-0.0080, 0.0150, 0.0], [0.0030, -0.0100, 0.0140]],
        K_P=[[0.20, 0.00, 0.00], [0.15, 0.70, 0.00], [0.00, 0.30, 1.50]],
        theta_P=[-0.020, 0.000, 0.000],
        sigma_e=np.full(n_maturities, 0.0004),
    )


def simulate_panel(
    params: VasicekParams,
    maturities: Sequence[float],
    n_periods: int,
    dt: float = 1 / 12,
    seed: int = 42,
    start: str = "1986-01-31",
) -> dict[str, pd.DataFrame]:
    """Simulate states from the exact P transition and yields from Q pricing + noise."""
    rng = np.random.default_rng(seed)
    tau = np.asarray(maturities, float)
    c, F, Q = discretize_ou(params.K_P, params.theta_P, params.Sigma, dt)
    m0, P0 = unconditional_moments(params.theta_P, F, Q)
    LQ = np.linalg.cholesky(Q)
    X = np.empty((n_periods, 3))
    x = rng.multivariate_normal(m0, P0)
    for t in range(n_periods):
        x = c + F @ x + LQ @ rng.standard_normal(3)
        X[t] = x
    aQ, bQ = q_loadings(params, tau)
    aP, bP = p_loadings(params, tau)
    yQ = aQ[None, :] + X @ bQ.T
    yP = aP[None, :] + X @ bP.T
    noise = rng.standard_normal(yQ.shape) * params.sigma_e[None, :]
    idx = pd.date_range(start=start, periods=n_periods, freq="ME")
    cols = pd.Index(tau, name="maturity")
    return {
        "yields": pd.DataFrame(yQ + noise, index=idx, columns=cols),
        "yields_true": pd.DataFrame(yQ, index=idx, columns=cols),
        "states": pd.DataFrame(X, index=idx, columns=["x1", "x2", "x3"]),
        "term_premium": pd.DataFrame(yQ - yP, index=idx, columns=cols),
    }
