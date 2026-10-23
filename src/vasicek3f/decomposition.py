"""
Term premium decomposition inside the affine framework.

    Term Premium(tau)_t = y^Q(tau)_t  -  y^P(tau)_t

    y^Q : pricing-measure fitted yield   = a_Q(tau) + b_Q(tau)' x_t
    y^P : physical-measure affine yield  = a_P(tau) + b_P(tau)' x_t
          (the yield implied by pricing bonds with the P dynamics, i.e. zero
           market price of risk -- a tractable no-risk-premium benchmark)

IMPORTANT: y^P is *not* a strict Expectations-Hypothesis estimate. The strict EH
component is the average expected short rate (1/tau) int E^P_t[r_{t+s}] ds, and

    y^P = EH-average-short-rate  -  P-measure convexity / tau

so the two differ by a (small, maturity-increasing) convexity term. Both are
reported; the headline decomposition uses y^P as specified.

Because both y^Q and y^P are affine in the same state, the term premium is itself
affine:  TP_t = (a_Q - a_P) + (b_Q - b_P)' x_t, so its loadings show which latent
factors drive variation in the premium.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .affine import q_loadings, p_loadings, eh_loadings, model_yields
from .params import VasicekParams


def loadings_table(params: VasicekParams, maturities: np.ndarray) -> pd.DataFrame:
    """a and b loadings under Q, P and for the term premium, by maturity."""
    aQ, bQ = q_loadings(params, maturities)
    aP, bP = p_loadings(params, maturities)
    aE, bE = eh_loadings(params, maturities)
    rows = {}
    for lab, a, b in (("Q", aQ, bQ), ("P", aP, bP), ("EH", aE, bE), ("TP", aQ - aP, bQ - bP)):
        rows[f"a_{lab}"] = a
        for i in range(3):
            rows[f"b{i+1}_{lab}"] = b[:, i]
    return pd.DataFrame(rows, index=pd.Index(maturities, name="maturity"))


def term_premium_decomposition(params: VasicekParams, states: np.ndarray,
                               maturities: np.ndarray, index: pd.Index
                               ) -> Dict[str, pd.DataFrame]:
    """
    Returns DataFrames (T x N, columns = maturities, decimals):
        fitted_Q        pricing-measure fitted yield
        affine_P        physical-measure affine yield (expectations component)
        term_premium    fitted_Q - affine_P
        eh_short_rate   strict EH average expected short rate (reference)
        convexity_P     eh_short_rate - affine_P  (P convexity, >= 0 typically)
    """
    tau = np.asarray(maturities, float)
    aQ, bQ = q_loadings(params, tau)
    aP, bP = p_loadings(params, tau)
    aE, bE = eh_loadings(params, tau)
    cols = pd.Index(tau, name="maturity")
    yQ = pd.DataFrame(model_yields(aQ, bQ, states), index=index, columns=cols)
    yP = pd.DataFrame(model_yields(aP, bP, states), index=index, columns=cols)
    eh = pd.DataFrame(model_yields(aE, bE, states), index=index, columns=cols)
    return {
        "fitted_Q": yQ,
        "affine_P": yP,
        "term_premium": yQ - yP,
        "eh_short_rate": eh,
        "convexity_P": eh - yP,
    }


def decomposition_at(dec: Dict[str, pd.DataFrame], maturity: float,
                     observed: pd.DataFrame | None = None) -> pd.DataFrame:
    """Single-maturity view: observed, fitted, P-affine, TP, EH reference (decimals)."""
    out = pd.DataFrame({
        "fitted_Q": dec["fitted_Q"][maturity],
        "affine_P": dec["affine_P"][maturity],
        "term_premium": dec["term_premium"][maturity],
        "eh_short_rate": dec["eh_short_rate"][maturity],
    })
    if observed is not None and maturity in observed.columns:
        out.insert(0, "observed", observed[maturity])
    return out
