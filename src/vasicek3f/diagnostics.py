"""
Model diagnostics: fit quality, factor interpretation, benchmark comparison.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def pca_explained(yields: pd.DataFrame, n: int = 5) -> pd.Series:
    """Share of yield-level variance explained by the first n principal components."""
    Y = yields.dropna().values
    Y = Y - Y.mean(0)
    s = np.linalg.svd(Y, compute_uv=False) ** 2
    share = s / s.sum()
    return pd.Series(share[:n], index=[f"PC{i+1}" for i in range(min(n, len(share)))])


def pricing_errors(observed: pd.DataFrame, fitted: pd.DataFrame) -> pd.DataFrame:
    """Per-maturity pricing error statistics in basis points."""
    e = (observed - fitted) * 1e4
    out = pd.DataFrame({
        "mean_bp": e.mean(),
        "mae_bp": e.abs().mean(),
        "rmse_bp": np.sqrt((e ** 2).mean()),
        "max_abs_bp": e.abs().max(),
        "ar1": e.apply(lambda s: s.autocorr(1)),
    })
    out.index.name = "maturity"
    return out


def level_slope_curvature(yields: pd.DataFrame) -> pd.DataFrame:
    """Empirical yield-curve factors from observed yields."""
    cols = yields.columns
    short, long = cols.min(), cols.max()
    mid = cols[np.argmin(np.abs(np.asarray(cols) - 0.5 * (short + long)))]
    return pd.DataFrame({
        "level": yields.mean(axis=1),
        "slope": yields[long] - yields[short],
        "curvature": 2 * yields[mid] - yields[short] - yields[long],
    })


def factor_correlations(states: pd.DataFrame, yields: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation of latent states with empirical level / slope / curvature.
    Latent factors are only identified up to the canonical normalisation, so they
    need not line up one-for-one with level/slope/curvature; this table shows how
    they relate.
    """
    lsc = level_slope_curvature(yields)
    df = pd.concat([states, lsc], axis=1).dropna()
    return df.corr().loc[states.columns, lsc.columns]


def benchmark_comparison(model_tp: pd.Series, benchmarks: Dict[str, pd.Series]) -> pd.DataFrame:
    """Compare the model term premium with external series on overlapping month-ends."""
    rows = {}
    for name, b in benchmarks.items():
        if b is None:
            continue
        df = pd.concat([model_tp.rename("model"), b.rename("bench")], axis=1).dropna()
        if len(df) < 12:
            continue
        d = (df["model"] - df["bench"]) * 1e4
        rows[name] = {
            "n_months": len(df),
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "corr_level": df["model"].corr(df["bench"]),
            "corr_12m_change": df["model"].diff(12).corr(df["bench"].diff(12)),
            "mean_model_pct": df["model"].mean() * 100,
            "mean_bench_pct": df["bench"].mean() * 100,
            "mean_diff_bp": d.mean(),
            "rmse_diff_bp": np.sqrt((d ** 2).mean()),
        }
    return pd.DataFrame(rows).T


def innovation_summary(innovations: np.ndarray, innov_var: np.ndarray,
                       maturities) -> pd.DataFrame:
    """Standardised one-step-ahead prediction errors: should be ~N(0,1), serially uncorrelated."""
    z = innovations / np.sqrt(innov_var)
    df = pd.DataFrame(z, columns=maturities)
    return pd.DataFrame({
        "mean": df.mean(), "std": df.std(),
        "ar1": df.apply(lambda s: s.autocorr(1)),
        "share_abs_gt_2": (df.abs() > 2).mean(),
    })
