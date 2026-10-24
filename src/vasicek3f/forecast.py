"""
Yield forecasting and an expanding-window (pseudo) out-of-sample evaluation.

Model forecast of yields h months ahead, using the exact P-measure VAR:

    E_t[x_{t+h}] = theta_P + F^h (x_{t|t} - theta_P),      F = exp(-K_P dt)
    E_t[y_{t+h}(tau)] = a^Q(tau) + b^Q(tau)' E_t[x_{t+h}]

Out-of-sample design
--------------------
* Start with an initial estimation window (e.g. the first half of the sample).
* At each forecast origin t, parameters are those estimated on data up to the most
  recent refit date <= t (refit every ``refit_every`` months, warm-started from the
  previous estimate), and the state is the *filtered* state x_{t|t}, which uses data
  up to t only. No future information enters a forecast.
* Benchmark: the random walk (no-change) forecast y_t. Beating it at 6-12 month
  horizons is notoriously hard for term structure models (Duffee 2002, 2011).
* Diebold-Mariano test on squared errors, with a Newey-West variance (h-1 lags).
  Negative DM statistics favour the model.

Note: "pseudo" out-of-sample because the yield data are final, not real-time
vintages, and the model specification was chosen with the full sample in view.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from .affine import q_loadings
from .discretize import discretize_ou
from .estimation import estimate
from .kalman import kalman_filter
from .model import build_state_space
from .params import VasicekParams


def forecast_states(params: VasicekParams, x_t: np.ndarray, h: int, dt: float) -> np.ndarray:
    """E_t[x_{t+h}] under the exact discrete P dynamics (h in periods)."""
    _, F, _ = discretize_ou(params.K_P, params.theta_P, params.Sigma, dt)
    return params.theta_P + np.linalg.matrix_power(F, h) @ (x_t - params.theta_P)


def forecast_yields(
    params: VasicekParams, x_t: np.ndarray, maturities: Sequence[float], h: int, dt: float
) -> np.ndarray:
    """Model-implied expected yields h periods ahead."""
    a, b = q_loadings(params, np.asarray(maturities, float))
    return a + b @ forecast_states(params, x_t, h, dt)


def _newey_west_var(d: np.ndarray, lags: int) -> float:
    d = d - d.mean()
    T = len(d)
    v = d @ d / T
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / T
    return v


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, h: int) -> float:
    """DM statistic for equal squared-error loss (negative => model better)."""
    d = e_model**2 - e_bench**2
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return np.nan
    v = _newey_west_var(d, max(h - 1, 0))
    return float(d.mean() / np.sqrt(v / len(d))) if v > 0 else np.nan


def expanding_window_oos(
    y: pd.DataFrame,
    maturities: Sequence[float],
    dt: float,
    initial: Optional[int] = None,
    refit_every: int = 12,
    horizons: Sequence[int] = (1, 6, 12),
    start: Optional[VasicekParams] = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Expanding-window pseudo out-of-sample forecast evaluation.

    Returns
    -------
    dict
        ``errors``: long DataFrame (origin, horizon, maturity, actual, model, rw);
        ``summary``: RMSE (bp) of model and random walk, ratio and DM statistic by
        horizon and maturity; ``refits``: refit dates with log-likelihood and kappa_Q.
    """
    tau = np.asarray(maturities, float)
    Y = y.values
    T = len(Y)
    t0 = initial if initial is not None else T // 2
    max(horizons)
    rows = []
    refit_info = []
    params = start
    origins = list(range(t0 - 1, T - 1))
    for k, t in enumerate(origins):
        if k % refit_every == 0:
            res = estimate(
                Y[: t + 1],
                tau,
                dt,
                start=params,
                compute_se=False,
                polish=False,
                ftol=1e-9,
                maxiter=300 if params is not None else 400,
                verbose=False,
            )
            params = res.params
            ss = build_state_space(params, tau, dt)
            kf = kalman_filter(Y, ss.a, ss.H, ss.R, ss.c, ss.F, ss.Q, ss.x0, ss.P0)
            refit_info.append(
                {
                    "refit_date": y.index[t],
                    "n_obs": t + 1,
                    "loglik": res.loglik,
                    **{f"kappaQ_{i + 1}": params.kappa_Q[i] for i in range(3)},
                }
            )
            if verbose:
                print(f"  refit at {y.index[t]:%Y-%m} (window {t + 1} months) loglik={res.loglik:,.1f}")
        x_t = kf.x_filt[t]  # causal: filtered with data up to t only
        for h in horizons:
            if t + h >= T:
                continue
            f = forecast_yields(params, x_t, tau, h, dt)
            for j, m in enumerate(tau):
                rows.append((y.index[t], h, m, Y[t + h, j], f[j], Y[t, j]))
    err = pd.DataFrame(rows, columns=["origin", "horizon", "maturity", "actual", "model", "rw"])
    summ = []
    for (h, m), g in err.groupby(["horizon", "maturity"]):
        em = (g["actual"] - g["model"]).values
        er = (g["actual"] - g["rw"]).values
        rm, rr = np.sqrt(np.mean(em**2)) * 1e4, np.sqrt(np.mean(er**2)) * 1e4
        summ.append(
            {
                "horizon_m": h,
                "maturity": m,
                "n": len(g),
                "rmse_model_bp": rm,
                "rmse_rw_bp": rr,
                "ratio": rm / rr,
                "dm_stat": diebold_mariano(em, er, h),
            }
        )
    return {
        "errors": err,
        "summary": pd.DataFrame(summ),
        "refits": pd.DataFrame(refit_info),
        "horizons": list(horizons),
        "initial_window": t0,
        "refit_every": refit_every,
    }
