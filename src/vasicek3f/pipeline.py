"""
End-to-end pipeline: data -> estimation -> filtering/smoothing -> decomposition
-> diagnostics -> saved results bundle (CSV/JSON for humans, pickle for the dashboard).
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import data as D
from .decomposition import term_premium_decomposition, decomposition_at, loadings_table
from .diagnostics import (pca_explained, pricing_errors, factor_correlations,
                          benchmark_comparison, innovation_summary)
from .estimation import estimate, EstimationResult
from .model import filter_and_smooth
from .params import VasicekParams
from .simulate import default_true_params, simulate_panel

BUNDLE_FILE = "bundle.pkl"


@dataclass
class RunConfig:
    source: str = "simulated"              # "simulated" | "gsw"
    gsw_path: Optional[str] = None
    acm_path: Optional[str] = None
    kw_path: Optional[str] = None
    start: Optional[str] = "1990-01"
    end: Optional[str] = None
    maturities: List[float] = field(default_factory=lambda: [float(m) for m in range(1, 11)])
    tp_maturity: float = 10.0
    dt: float = 1.0 / 12.0
    fast: bool = False                     # fast mode: no SEs, capped iterations
    compute_se: bool = True
    n_restarts: int = 1
    maxiter: int = 4000
    polish: bool = True
    ftol: float = 1e-14
    common_measurement_error: bool = False
    sim_periods: int = 480
    seed: int = 42
    out_dir: str = "results/simulated"
    make_plots: bool = True
    verbose: bool = True

    def effective(self) -> "RunConfig":
        if not self.fast:
            return self
        c = RunConfig(**asdict(self))
        c.compute_se = False
        c.maxiter = min(self.maxiter, 400)
        c.n_restarts = 1
        c.polish = False
        c.ftol = 1e-10
        return c


def load_inputs(cfg: RunConfig) -> Dict[str, object]:
    """Return observed yields plus any benchmark term premium series."""
    tau = np.asarray(cfg.maturities, float)
    bench: Dict[str, pd.Series] = {}
    extra: Dict[str, object] = {}
    if cfg.source == "simulated":
        true = default_true_params(tau.size)
        sim = simulate_panel(true, tau, cfg.sim_periods, cfg.dt, seed=cfg.seed)
        y = sim["yields"]
        if cfg.tp_maturity in sim["term_premium"].columns:
            bench["True TP (simulation)"] = sim["term_premium"][cfg.tp_maturity]
        extra.update(true_params=true, true_states=sim["states"])
    elif cfg.source == "gsw":
        if not cfg.gsw_path:
            raise ValueError("--gsw path to feds200628.csv is required for source=gsw")
        y = D.load_gsw(cfg.gsw_path, tau, cfg.start, cfg.end)
        y = y.dropna(how="any")  # keep a balanced panel (collapsed filter is exact & fast)
        if cfg.acm_path:
            bench["ACM"] = D.load_acm(cfg.acm_path, int(cfg.tp_maturity))
        if cfg.kw_path:
            bench["KW"] = D.load_kw(cfg.kw_path)
    else:
        raise ValueError(f"unknown source {cfg.source!r}")
    return {"yields": y, "benchmarks": bench, **extra}


def run(cfg: RunConfig) -> Dict[str, object]:
    t0 = time.time()
    cfg = cfg.effective()
    tau = np.asarray(cfg.maturities, float)
    inp = load_inputs(cfg)
    y: pd.DataFrame = inp["yields"]
    if cfg.verbose:
        print(f"Data: {cfg.source}, {len(y)} months x {len(tau)} maturities "
              f"({y.index.min():%Y-%m} to {y.index.max():%Y-%m})")

    est: EstimationResult = estimate(
        y.values, tau, cfg.dt, common_measurement_error=cfg.common_measurement_error,
        maxiter=cfg.maxiter, n_restarts=cfg.n_restarts, seed=cfg.seed,
        compute_se=cfg.compute_se, polish=cfg.polish, ftol=cfg.ftol, verbose=cfg.verbose)
    p = est.params

    fs = filter_and_smooth(p, y.values, tau, cfg.dt)
    states = pd.DataFrame(fs["x_smoothed"], index=y.index, columns=["x1", "x2", "x3"])
    states_filt = pd.DataFrame(fs["x_filtered"], index=y.index, columns=["x1", "x2", "x3"])
    dec = term_premium_decomposition(p, states.values, tau, y.index)
    dec_tp = decomposition_at(dec, cfg.tp_maturity, y)
    kf = fs["kalman"]

    bundle = {
        "config": asdict(cfg),
        "maturities": tau,
        "tp_maturity": cfg.tp_maturity,
        "yields": y,
        "params": p,
        "start_params": est.start_params,
        "estimation": {
            "loglik": est.loglik, "n_obs": est.n_obs, "n_params": est.n_params,
            "aic": est.aic, "bic": est.bic, "success": est.success, "message": est.message,
            "n_iter": est.n_iter, "elapsed_sec": est.elapsed_sec, "history": est.history,
            "hessian_pd": (est.std_errors.attrs.get("hessian_pd") if est.std_errors is not None else None),
        },
        "std_errors": est.std_errors,
        "states": states,
        "states_filtered": states_filt,
        "states_cov": fs["P_smoothed"],
        "decomposition": dec,
        "decomposition_tp": dec_tp,
        "loadings": loadings_table(p, np.linspace(0.25, max(10.0, tau.max()), 40)),
        "loadings_obs": loadings_table(p, tau),
        "pricing_errors": pricing_errors(y, dec["fitted_Q"]),
        "pca": pca_explained(y),
        "factor_corr": factor_correlations(states, y),
        "innovations": innovation_summary(kf.innovations, kf.innov_var_diag, tau),
        "benchmarks": inp["benchmarks"],
        "benchmark_table": benchmark_comparison(dec["term_premium"][cfg.tp_maturity],
                                                inp["benchmarks"]),
        "true_params": inp.get("true_params"),
        "true_states": inp.get("true_states"),
        "runtime_sec": None,
    }
    bundle["runtime_sec"] = time.time() - t0
    save_bundle(bundle, cfg.out_dir, make_plots=cfg.make_plots)
    if cfg.verbose:
        print(f"Done in {bundle['runtime_sec']:.0f}s. Results in {cfg.out_dir}/")
    return bundle


# ---------------------------------------------------------------------------
# Saving / loading
# ---------------------------------------------------------------------------
def _pct(df):
    return df * 100.0


def save_bundle(bundle: dict, out_dir: str, make_plots: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, BUNDLE_FILE), "wb") as f:
        pickle.dump(bundle, f)
    p: VasicekParams = bundle["params"]
    meta = {"estimation": bundle["estimation"], "params": p.to_dict(),
            "config": bundle["config"], "runtime_sec": bundle["runtime_sec"]}
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    if bundle["std_errors"] is not None:
        bundle["std_errors"].to_csv(os.path.join(out_dir, "std_errors.csv"))
    _pct(bundle["yields"]).to_csv(os.path.join(out_dir, "yields_observed_pct.csv"))
    for k, v in bundle["decomposition"].items():
        _pct(v).to_csv(os.path.join(out_dir, f"{k}_pct.csv"))
    _pct(bundle["decomposition_tp"]).to_csv(
        os.path.join(out_dir, f"decomposition_{bundle['tp_maturity']:g}y_pct.csv"))
    bundle["states"].to_csv(os.path.join(out_dir, "states_smoothed.csv"))
    bundle["pricing_errors"].to_csv(os.path.join(out_dir, "pricing_errors_bp.csv"))
    bundle["loadings_obs"].to_csv(os.path.join(out_dir, "loadings.csv"))
    if len(bundle["benchmark_table"]):
        bundle["benchmark_table"].to_csv(os.path.join(out_dir, "benchmark_comparison.csv"))
    if make_plots:
        from .plotting import make_all
        make_all(bundle, os.path.join(out_dir, "figures"))
    write_summary(bundle, os.path.join(out_dir, "summary.md"))


def load_bundle(out_dir: str) -> dict:
    with open(os.path.join(out_dir, BUNDLE_FILE), "rb") as f:
        return pickle.load(f)


def write_summary(b: dict, path: str) -> None:
    p: VasicekParams = b["params"]
    e = b["estimation"]
    tau_tp = b["tp_maturity"]
    tp = b["decomposition"]["term_premium"][tau_tp]
    lr = p.short_rate_long_run()
    hl = p.factor_half_lives()
    L = []
    L.append(f"# 3-Factor Vasicek term structure model: results\n")
    L.append(f"Source: **{b['config']['source']}**, {len(b['yields'])} months, "
             f"{b['yields'].index.min():%Y-%m} to {b['yields'].index.max():%Y-%m}, "
             f"maturities {', '.join(f'{m:g}y' for m in b['maturities'])}. "
             f"Runtime {b['runtime_sec']:.0f}s{' (fast mode)' if b['config']['fast'] else ''}.\n")
    L.append("## Estimation\n")
    L.append(f"- Log-likelihood: {e['loglik']:,.2f}  (AIC {e['aic']:,.1f}, BIC {e['bic']:,.1f}, "
             f"{e['n_params']} parameters, {e['n_obs']} observations)")
    L.append(f"- Optimiser: {e['message']} after {e['n_iter']} iterations")
    L.append(f"- kappa_Q: {np.round(p.kappa_Q, 4).tolist()}  (Q half-lives, years: "
             f"{np.round(hl['Q'], 2).tolist()})")
    L.append(f"- eigenvalues of K_P (real part): {np.round(np.real(np.linalg.eigvals(p.K_P)), 4).tolist()}"
             f"  (P half-lives, years: {np.round(hl['P'], 2).tolist()})")
    L.append(f"- Long-run short rate: Q {lr['Q']*100:.2f}%, P {lr['P']*100:.2f}%")
    L.append(f"- Measurement error s.d. (bp): {np.round(p.sigma_e*1e4, 2).tolist()}\n")
    L.append("## Fit\n")
    L.append("PCA of yield levels (share of variance): " +
             ", ".join(f"{k} {v:.2%}" for k, v in b["pca"].items()) + "\n")
    L.append(b["pricing_errors"].round(2).to_markdown() + "\n")
    L.append("## Term premium\n")
    L.append(f"{tau_tp:g}y term premium: mean {tp.mean()*100:.2f}%, s.d. {tp.std()*100:.2f}%, "
             f"min {tp.min()*100:.2f}%, max {tp.max()*100:.2f}%, latest {tp.iloc[-1]*100:.2f}% "
             f"({tp.index[-1]:%Y-%m}).\n")
    L.append("Decomposition: **Term Premium = Pricing-Measure Fitted Yield − Physical-Measure "
             "Affine Yield**. The P-affine yield is a no-risk-premium benchmark inside the model, "
             "not a strict Expectations-Hypothesis average of expected short rates.\n")
    if len(b["benchmark_table"]):
        L.append("### Benchmark comparison\n")
        L.append(b["benchmark_table"].to_markdown(floatfmt=".3f") + "\n")
    L.append("## Figures\n")
    for f in ["01_fit", "02_pricing_errors", "03_loadings", "04_states", "05_decomposition",
              "06_benchmarks", "07_average_curve"]:
        L.append(f"![{f}](figures/{f}.png)")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
