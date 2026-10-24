"""
Parameter-uncertainty bands for the term premium.

Method (parametric bootstrap / Krinsky-Robb simulation)
------------------------------------------------------
1. Draw u^(s) ~ N(u_hat, Cov(u_hat)) in the unconstrained parameter space, where
   Cov(u_hat) is the inverse numerical Hessian from the MLE.
2. Map each draw to model parameters (draws with non-stationary K_P are discarded).
3. Re-run the Kalman smoother at the drawn parameters (no re-optimisation) and
   recompute the decomposition.
4. Report pointwise quantiles of the term premium across draws.

The bands capture *parameter* uncertainty, which dominates for term premia because
of the weakly identified P dynamics. Filtering uncertainty about the states given
the parameters is small by comparison (pricing errors are a few bp) and is not
added. Bands are pointwise, not simultaneous.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .decomposition import term_premium_decomposition
from .model import filter_and_smooth
from .params import ParamTransform


def term_premium_bands(
    u_hat: np.ndarray,
    cov_u: np.ndarray,
    transform: ParamTransform,
    y: pd.DataFrame,
    maturities: Sequence[float],
    dt: float,
    n_draws: int = 200,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    seed: int = 0,
) -> dict[str, object]:
    """
    Simulate term premium bands from the asymptotic distribution of the MLE.

    Returns a dict with one DataFrame per quantile (``"q05"``, ``"q50"``, ``"q95"``;
    rows = dates, columns = maturities, decimals) plus the number of valid draws.
    """
    rng = np.random.default_rng(seed)
    cov = 0.5 * (cov_u + cov_u.T)
    w, V = np.linalg.eigh(cov)
    L = V * np.sqrt(np.clip(w, 0, None))
    tau = np.asarray(maturities, float)
    draws = []
    n_bad = 0
    for _ in range(n_draws):
        u = u_hat + L @ rng.standard_normal(u_hat.size)
        p = transform.to_params(u)
        if not p.is_stationary_P():
            n_bad += 1
            continue
        try:
            fs = filter_and_smooth(p, y.values, tau, dt)
            dec = term_premium_decomposition(p, fs["x_smoothed"], tau, y.index)
            tp = dec["term_premium"].values
            if np.all(np.isfinite(tp)):
                draws.append(tp)
            else:
                n_bad += 1
        except (np.linalg.LinAlgError, ValueError, RuntimeError):
            n_bad += 1
    if not draws:
        raise RuntimeError("no valid parameter draws for uncertainty bands")
    arr = np.stack(draws)
    cols = pd.Index(tau, name="maturity")
    out: dict[str, object] = {}
    for q in quantiles:
        out[f"q{int(round(q * 100)):02d}"] = pd.DataFrame(
            np.quantile(arr, q, axis=0), index=y.index, columns=cols
        )
    out["n_valid"] = len(draws)
    out["n_discarded"] = n_bad
    out["quantiles"] = list(quantiles)
    return out
