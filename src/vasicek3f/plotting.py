"""
Static matplotlib figures for the report and the notebook.
(The dashboard draws its own interactive Plotly versions of the same charts.)
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C = {"obs": "#2b2b2b", "Q": "#1f5fa8", "P": "#c4622d", "TP": "#2e8b57",
     "ACM": "#7b4fa3", "KW": "#b8860b", "true": "#888888"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10, "legend.frameon": False,
})


def _save(fig, out_dir: Optional[str], name: str):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name), bbox_inches="tight")
    return fig


def plot_fit(observed: pd.DataFrame, fitted: pd.DataFrame, maturities=(1.0, 5.0, 10.0),
             out_dir: Optional[str] = None):
    mats = [m for m in maturities if m in observed.columns]
    fig, axes = plt.subplots(len(mats), 1, figsize=(10, 2.6 * len(mats)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, mats):
        ax.plot(observed.index, observed[m] * 100, color=C["obs"], lw=1.2, label="Observed")
        ax.plot(fitted.index, fitted[m] * 100, color=C["Q"], lw=1.0, ls="--", label="Model (Q)")
        ax.set_ylabel(f"{m:g}y yield, %")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Observed vs model-implied yields")
    return _save(fig, out_dir, "01_fit.png")


def plot_pricing_errors(err_table: pd.DataFrame, observed: pd.DataFrame, fitted: pd.DataFrame,
                        out_dir: Optional[str] = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    a1.bar([f"{m:g}" for m in err_table.index], err_table["rmse_bp"], color=C["Q"])
    a1.set_xlabel("maturity (years)"); a1.set_ylabel("RMSE, bp"); a1.set_title("Pricing error RMSE")
    e = (observed - fitted) * 1e4
    lim = np.nanpercentile(np.abs(e.values), 99)
    im = a2.imshow(e.T.values, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                   extent=[0, len(e), len(e.columns) - 0.5, -0.5])
    a2.set_yticks(range(len(e.columns))); a2.set_yticklabels([f"{m:g}y" for m in e.columns])
    ticks = np.linspace(0, len(e) - 1, 6).astype(int)
    a2.set_xticks(ticks); a2.set_xticklabels([e.index[i].year for i in ticks])
    a2.set_title("Pricing errors over time (observed - model), bp")
    fig.colorbar(im, ax=a2, fraction=0.03)
    return _save(fig, out_dir, "02_pricing_errors.png")


def plot_loadings(loadings: pd.DataFrame, out_dir: Optional[str] = None):
    tau = loadings.index.values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, ls in zip(range(1, 4), ["-", "--", ":"]):
        a1.plot(tau, loadings[f"b{i}_Q"], color=C["Q"], ls=ls, label=f"b{i} (Q)")
        a1.plot(tau, loadings[f"b{i}_P"], color=C["P"], ls=ls, label=f"b{i} (P)")
    a1.set_title("Factor loadings b(τ)"); a1.set_xlabel("maturity (years)"); a1.legend(ncol=2, fontsize=8)
    a2.plot(tau, loadings["a_Q"] * 100, color=C["Q"], label="a(τ) Q")
    a2.plot(tau, loadings["a_P"] * 100, color=C["P"], label="a(τ) P")
    a2.set_title("Intercepts a(τ), %"); a2.set_xlabel("maturity (years)"); a2.legend()
    return _save(fig, out_dir, "03_loadings.png")


def plot_states(states: pd.DataFrame, bands: Optional[np.ndarray] = None,
                out_dir: Optional[str] = None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
    for i, ax in enumerate(axes):
        col = states.columns[i]
        ax.plot(states.index, states[col] * 100, color=C["Q"], lw=1.1)
        if bands is not None:
            sd = np.sqrt(bands[:, i, i]) * 100
            ax.fill_between(states.index, states[col] * 100 - 2 * sd, states[col] * 100 + 2 * sd,
                            color=C["Q"], alpha=0.2, lw=0)
        ax.set_ylabel(f"{col}, %")
    axes[0].set_title("Smoothed latent factors (±2 s.d.)")
    return _save(fig, out_dir, "04_states.png")


def plot_decomposition(dec_tau: pd.DataFrame, maturity: float, out_dir: Optional[str] = None,
                       true_tp: Optional[pd.Series] = None):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    if "observed" in dec_tau:
        a1.plot(dec_tau.index, dec_tau["observed"] * 100, color=C["obs"], lw=0.8, alpha=0.6, label="Observed")
    a1.plot(dec_tau.index, dec_tau["fitted_Q"] * 100, color=C["Q"], lw=1.3, label="Pricing-measure fitted yield")
    a1.plot(dec_tau.index, dec_tau["affine_P"] * 100, color=C["P"], lw=1.3, label="Physical-measure affine yield")
    a1.set_ylabel("%"); a1.legend(loc="best", fontsize=8); a1.set_title(f"{maturity:g}-year yield decomposition")
    a2.plot(dec_tau.index, dec_tau["term_premium"] * 100, color=C["TP"], lw=1.4, label="Model term premium")
    if true_tp is not None:
        a2.plot(true_tp.index, true_tp * 100, color=C["true"], lw=1.0, ls="--", label="True term premium (simulation)")
    a2.axhline(0, color="k", lw=0.6)
    a2.set_ylabel("%"); a2.legend(); a2.set_title("Term premium = fitted (Q) − affine (P)")
    return _save(fig, out_dir, "05_decomposition.png")


def plot_benchmarks(model_tp: pd.Series, benchmarks: Dict[str, pd.Series], maturity: float,
                    out_dir: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(model_tp.index, model_tp * 100, color=C["TP"], lw=1.5, label="3F Vasicek (this model)")
    for name, s in benchmarks.items():
        if s is None:
            continue
        s = s.loc[model_tp.index.min():model_tp.index.max()]
        ax.plot(s.index, s * 100, color=C.get(name.split()[0], C["true"]), lw=1.1,
                ls="--" if name.startswith("True") else "-", label=name)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("%"); ax.legend(); ax.set_title(f"{maturity:g}-year term premium vs benchmarks")
    return _save(fig, out_dir, "06_benchmarks.png")


def plot_average_curve(observed: pd.DataFrame, dec: Dict[str, pd.DataFrame],
                       out_dir: Optional[str] = None):
    tau = observed.columns.values
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(tau, observed.mean() * 100, "o", color=C["obs"], label="Observed (mean)")
    ax.plot(tau, dec["fitted_Q"].mean() * 100, color=C["Q"], label="Fitted Q (mean)")
    ax.plot(tau, dec["affine_P"].mean() * 100, color=C["P"], label="Affine P (mean)")
    ax.plot(tau, dec["term_premium"].mean() * 100, color=C["TP"], label="Term premium (mean)")
    ax.set_xlabel("maturity (years)"); ax.set_ylabel("%"); ax.legend()
    ax.set_title("Average curves across the sample")
    return _save(fig, out_dir, "07_average_curve.png")


def make_all(bundle: dict, out_dir: str) -> None:
    obs = bundle["yields"]; dec = bundle["decomposition"]; tau_tp = bundle["tp_maturity"]
    plot_fit(obs, dec["fitted_Q"], out_dir=out_dir)
    plot_pricing_errors(bundle["pricing_errors"], obs, dec["fitted_Q"], out_dir=out_dir)
    plot_loadings(bundle["loadings"], out_dir=out_dir)
    plot_states(bundle["states"], bundle["states_cov"], out_dir=out_dir)
    true_tp = bundle["benchmarks"].get("True TP (simulation)")
    plot_decomposition(bundle["decomposition_tp"], tau_tp, out_dir=out_dir, true_tp=true_tp)
    plot_benchmarks(dec["term_premium"][tau_tp], bundle["benchmarks"], tau_tp, out_dir=out_dir)
    plot_average_curve(obs, dec, out_dir=out_dir)
    plt.close("all")
