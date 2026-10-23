"""
Interactive dashboard for the 3-factor Vasicek term structure model.

    streamlit run dashboard/app.py

The dashboard reads a saved results bundle (results/<name>/bundle.pkl, written by
`python -m vasicek3f run ...`) or runs the estimator itself. Everything shown is
computed by the same functions in src/vasicek3f used by the CLI and the notebook;
the dashboard adds no model logic of its own except the sensitivity shocks.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from vasicek3f.affine import q_loadings, p_loadings  # noqa: E402
from vasicek3f.decomposition import term_premium_decomposition, loadings_table  # noqa: E402
from vasicek3f.model import filter_and_smooth, loglik  # noqa: E402
from vasicek3f.params import ParamTransform, VasicekParams  # noqa: E402
from vasicek3f.pipeline import RunConfig, run, load_bundle, BUNDLE_FILE  # noqa: E402

st.set_page_config(page_title="3-Factor Vasicek Term Structure", layout="wide")

COL = {"obs": "#3a3a3a", "Observed": "#3a3a3a", "Q": "#1f5fa8", "P": "#c4622d", "TP": "#2e8b57", "EH": "#8a6d3b",
       "ACM": "#7b4fa3", "KW": "#b8860b", "True TP (simulation)": "#8c8c8c", "shock": "#d62728"}
F_COLS = ["#1f5fa8", "#c4622d", "#2e8b57"]


# =============================================================================
# Data access
# =============================================================================
def saved_bundles() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", BUNDLE_FILE))):
        out[os.path.basename(os.path.dirname(p))] = os.path.dirname(p)
    return out


@st.cache_resource(show_spinner=False)
def cached_load(path: str, mtime: float) -> dict:
    return load_bundle(path)


DEMO_CFG = RunConfig(source="simulated", sim_periods=480, seed=42, fast=True,
                     out_dir=os.path.join(tempfile.gettempdir(), "vasicek3f_demo"),
                     make_plots=False, verbose=False)


def save_upload(upload, name: str):
    """Write a Streamlit upload to a temp file and return its path (None if no upload)."""
    if upload is None:
        return None
    d = os.path.join(tempfile.gettempdir(), "vasicek3f_uploads")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + os.path.splitext(upload.name)[1].lower())
    with open(path, "wb") as f:
        f.write(upload.getbuffer())
    return path


@st.cache_resource(show_spinner=False)
def cached_run(cfg_json: str) -> dict:
    cfg = RunConfig(**json.loads(cfg_json))
    return run(cfg)


def pct(x):
    return x * 100.0


def line_fig(series: dict, title: str, ytitle: str = "%", height: int = 360, dash=None):
    fig = go.Figure()
    for name, s in series.items():
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines",
                                 line=dict(color=COL.get(name.split(" ")[0], COL.get(name)),
                                           width=1.6, dash=(dash or {}).get(name))))
    fig.update_layout(title=title, height=height, yaxis_title=ytitle, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.15))
    return fig


def explain(text: str):
    with st.expander("How this chart connects to the model"):
        st.markdown(text)


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("3-Factor Vasicek")
st.sidebar.caption("Affine term structure model · Kalman filter · MLE · term premium")

bundles = saved_bundles()
modes = ["Load saved results", "Estimate: simulated data", "Estimate: GSW data (local files)"]
mode = st.sidebar.radio("Results source", modes, index=0 if bundles else 1)
fast = st.sidebar.toggle("Fast mode", value=True,
                         help="Skips standard errors, loosens the optimiser tolerance, skips the "
                              "BFGS polish step, holds states fixed in the sensitivity tab, and "
                              "uses a coarser maturity grid for loading charts.")

bundle = None
if mode == "Load saved results":
    if not bundles:
        st.sidebar.warning("No saved results found in results/. Run the CLI or choose an "
                           "estimation mode.")
    else:
        name = st.sidebar.selectbox("Saved run", list(bundles))
        path = bundles[name]
        try:
            bundle = cached_load(path, os.path.getmtime(os.path.join(path, BUNDLE_FILE)))
        except Exception:  # noqa: BLE001  (e.g. pickle written by different library versions)
            st.sidebar.info("Saved results could not be read with the installed library "
                            "versions, so a fresh fast-mode demo estimation is being run instead.")
            with st.spinner("Running the demo estimation on simulated data (about 10-20s, once) ..."):
                bundle = cached_run(json.dumps(asdict(DEMO_CFG)))
else:
    with st.sidebar.form("run_form"):
        if mode.startswith("Estimate: simulated"):
            periods = st.number_input("Months to simulate", 120, 1200, 480, step=60)
            seed = st.number_input("Random seed", 0, 10_000, 42)
            cfg = RunConfig(source="simulated", sim_periods=int(periods), seed=int(seed),
                            fast=fast,
                            out_dir=os.path.join(tempfile.gettempdir(), "vasicek3f_simulated"),
                            make_plots=False, verbose=False)
        else:
            st.caption("Upload the files (works on Streamlit Cloud) or give local paths.")
            up_gsw = st.file_uploader("GSW feds200628.csv", type=["csv"])
            up_acm = st.file_uploader("ACM term premium (optional)", type=["xls", "xlsx", "csv"])
            up_kw = st.file_uploader("Kim-Wright THREEFYTP10.csv (optional)", type=["csv"])
            gsw = st.text_input("…or local GSW path", os.path.join(ROOT, "data", "feds200628.csv"))
            acm = st.text_input("…or local ACM path", os.path.join(ROOT, "data", "ACMTermPremium.xls"))
            kw = st.text_input("…or local KW path", os.path.join(ROOT, "data", "THREEFYTP10.csv"))
            start = st.text_input("Start (YYYY-MM)", "1990-01")
            gsw = save_upload(up_gsw, "gsw") or gsw
            acm = save_upload(up_acm, "acm") or acm
            kw = save_upload(up_kw, "kw") or kw
            cfg = RunConfig(source="gsw", gsw_path=gsw,
                            acm_path=acm if os.path.exists(acm) else None,
                            kw_path=kw if os.path.exists(kw) else None, start=start, fast=fast,
                            out_dir=os.path.join(tempfile.gettempdir(), "vasicek3f_gsw"),
                            make_plots=False, verbose=False)
        go_btn = st.form_submit_button("Run estimation")
    if go_btn or "last_cfg" in st.session_state:
        if go_btn:
            st.session_state["last_cfg"] = json.dumps(asdict(cfg))
        t0 = time.time()
        with st.spinner("Estimating: starting values → Kalman-filter MLE"
                        + ("" if fast else " → numerical Hessian (standard errors)") + " ..."):
            try:
                bundle = cached_run(st.session_state["last_cfg"])
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"Estimation failed: {e}")

if bundle is None:
    st.title("3-Factor Vasicek Term Structure Model")
    st.info("Choose a saved run or start an estimation from the sidebar. To create saved "
            "results from the command line:\n\n"
            "`python -m vasicek3f run --source simulated --out results/simulated`")
    st.stop()

# -----------------------------------------------------------------------------
p: VasicekParams = bundle["params"]
y: pd.DataFrame = bundle["yields"]
tau = np.asarray(bundle["maturities"], float)
dec = bundle["decomposition"]
states: pd.DataFrame = bundle["states"]
est = bundle["estimation"]
cfg_b = bundle["config"]
is_sim = cfg_b["source"] == "simulated"
tp_mats = [m for m in tau]
tp_m = st.sidebar.selectbox("Term premium maturity (years)", tp_mats,
                            index=tp_mats.index(bundle["tp_maturity"]) if bundle["tp_maturity"] in tp_mats else len(tp_mats) - 1,
                            format_func=lambda m: f"{m:g}y")
st.sidebar.markdown("---")
st.sidebar.caption(f"Run: **{cfg_b['source']}**, {len(y)} months · "
                   f"estimated in {bundle['runtime_sec']:.0f}s"
                   f"{' (fast mode)' if cfg_b.get('fast') else ''}")

tabs = st.tabs(["Overview", "Yield curve fit", "Latent factors & loadings",
                "Term premium decomposition", "Benchmarks", "Sensitivity (sliders)",
                "Estimation & diagnostics"])

# =============================================================================
# 1. Overview
# =============================================================================
with tabs[0]:
    st.header("3-Factor Gaussian Vasicek term structure model")
    st.markdown(
        "An educational dashboard for a three-factor affine term structure model of government "
        "bond yields. It is written for **quant finance and MFE students, aspiring quant "
        "researchers, and practitioners preparing for interviews** who want to see how the "
        "theory (affine pricing, P vs Q, state space, Kalman filter, MLE) turns into numbers.")
    tp_series = dec["term_premium"][tp_m]
    c = st.columns(5)
    c[0].metric("Log-likelihood", f"{est['loglik']:,.0f}")
    c[1].metric("Avg pricing RMSE", f"{bundle['pricing_errors']['rmse_bp'].mean():.1f} bp")
    c[2].metric(f"{tp_m:g}y TP (latest)", f"{tp_series.iloc[-1]*100:.2f}%",
                f"{(tp_series.iloc[-1]-tp_series.iloc[-13])*1e4:+.0f}bp vs 12m ago" if len(tp_series) > 13 else None)
    c[3].metric(f"{tp_m:g}y TP (sample mean)", f"{tp_series.mean()*100:.2f}%")
    c[4].metric("Runtime", f"{bundle['runtime_sec']:.0f}s", "fast mode" if cfg_b.get("fast") else "full mode",
                delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Model at a glance")
        st.latex(r"\text{P: } dx_t = K^P(\theta^P - x_t)\,dt + \Sigma\, dW^P_t \qquad "
                 r"\text{Q: } dx_t = -K^Q x_t\,dt + \Sigma\, dW^Q_t")
        st.latex(r"r_t = \delta_0 + \iota' x_t,\qquad y_t(\tau) = a(\tau) + b(\tau)'x_t + e_t")
        st.latex(r"\underbrace{TP_t(\tau)}_{\text{term premium}} = "
                 r"\underbrace{a^Q(\tau)+b^Q(\tau)'x_t}_{\text{pricing-measure fitted yield}} - "
                 r"\underbrace{a^P(\tau)+b^P(\tau)'x_t}_{\text{physical-measure affine yield}}")
        st.markdown(
            "**Pipeline:** yields → closed-form Q loadings → state-space model (measurement from Q, "
            "transition from exact P discretisation) → Kalman filter likelihood → MLE → RTS "
            "smoother → affine decomposition.\n\n"
            "**Caution:** the P-affine yield is a *no-risk-premium benchmark inside the model*, "
            "not a strict Expectations-Hypothesis estimate. P-measure parameters are weakly "
            "identified, so the level of the term premium carries more uncertainty than the fit.")
    with right:
        st.subheader("Why three factors?")
        pca = bundle["pca"]
        fig = go.Figure(go.Bar(x=pca.index, y=pca.values * 100, marker_color=COL["Q"],
                               text=[f"{v:.2%}" for v in pca.values], textposition="outside"))
        fig.update_layout(height=300, yaxis_title="% of variance", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)
        st.caption("Principal components of yield levels: three factors (level, slope, "
                   "curvature) explain almost all cross-sectional variation, which is why a "
                   "three-factor model is the standard choice.")
    st.subheader("Observed yield curve over time")
    fig = go.Figure(go.Heatmap(z=pct(y.T.values), x=y.index, y=[f"{m:g}y" for m in tau],
                               colorscale="Viridis", colorbar=dict(title="%")))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)

# =============================================================================
# 2. Fit
# =============================================================================
with tabs[1]:
    st.header("Yield curve fit")
    a1, a2 = st.columns([2, 1])
    with a1:
        m_sel = st.selectbox("Maturity", tau, index=len(tau) - 1, format_func=lambda m: f"{m:g}y",
                             key="fit_m")
        st.plotly_chart(line_fig({"Observed": pct(y[m_sel]),
                                  "Q fitted (smoothed states)": pct(dec["fitted_Q"][m_sel])},
                                 f"{m_sel:g}y yield: observed vs model"))
    with a2:
        dates = list(y.index)
        d_sel = st.select_slider("Curve date", options=dates, value=dates[-1],
                                 format_func=lambda d: d.strftime("%Y-%m"))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tau, y=pct(y.loc[d_sel]), mode="markers", name="Observed",
                                 marker=dict(color=COL["obs"], size=8)))
        fig.add_trace(go.Scatter(x=tau, y=pct(dec["fitted_Q"].loc[d_sel]), mode="lines",
                                 name="Model", line=dict(color=COL["Q"])))
        fig.update_layout(title=f"Curve on {d_sel:%Y-%m}", xaxis_title="maturity (years)",
                          yaxis_title="%", height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    err = (y - dec["fitted_Q"]) * 1e4
    lim = float(np.nanpercentile(np.abs(err.values), 99))
    fig = go.Figure(go.Heatmap(z=err.T.values, x=err.index, y=[f"{m:g}y" for m in tau],
                               colorscale="RdBu_r", zmin=-lim, zmax=lim, colorbar=dict(title="bp")))
    fig.update_layout(title="Pricing errors (observed − model), bp", height=300,
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig)
    st.dataframe(bundle["pricing_errors"].style.format("{:.2f}"))
    explain(
        "Model yields are $\\hat y_t(\\tau) = a^Q(\\tau) + b^Q(\\tau)'\\hat x_{t|T}$: the closed-form "
        "Q-measure loadings (`affine.q_loadings`) applied to the RTS-smoothed states "
        "(`kalman.rts_smoother`). Errors are the measurement noise $e_t$ in the state-space model; "
        "their estimated standard deviations $\\sigma_e$ are MLE parameters. Errors of a few basis "
        "points with little autocorrelation mean three factors capture the cross-section. "
        "Persistent, maturity-patterned errors would suggest missing factors or misspecification.")

# =============================================================================
# 3. Factors & loadings
# =============================================================================
with tabs[2]:
    st.header("Latent factors and factor loadings")
    sd = np.sqrt(np.stack([np.diag(P) for P in bundle["states_cov"]]))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=["x1 (slowest under Q)", "x2", "x3 (fastest under Q)"])
    for i in range(3):
        s = pct(states.iloc[:, i])
        fig.add_trace(go.Scatter(x=s.index, y=s + 200 * sd[:, i], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=s - 200 * sd[:, i], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(31,95,168,0.18)", showlegend=False, hoverinfo="skip"),
                      row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=F_COLS[i]), name=f"x{i+1}"),
                      row=i + 1, col=1)
        if is_sim and bundle.get("true_states") is not None:
            ts = pct(bundle["true_states"].iloc[:, i])
            fig.add_trace(go.Scatter(x=ts.index, y=ts, line=dict(color="#999", dash="dot", width=1),
                                     name=f"true x{i+1}"), row=i + 1, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=40, b=10), yaxis_title="%")
    st.plotly_chart(fig)
    explain(
        "States are $\\hat x_{t|T}$ from the Rauch-Tung-Striebel smoother with ±2 s.d. bands from "
        "$P_{t|T}$. They are latent: identified only through the canonical normalisation "
        "($K^Q$ diagonal and ordered, $\\theta^Q=0$, $\\delta_1=\\iota$, $\\Sigma$ lower triangular), "
        "so they are *rotations* of level/slope/curvature rather than those factors exactly. In "
        "simulation the dotted lines show the true states.")

    lt = loadings_table(p, np.linspace(0.25, 10, 20 if fast else 60))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i in range(3):
            fig.add_trace(go.Scatter(x=lt.index, y=lt[f"b{i+1}_Q"], name=f"b{i+1} Q",
                                     line=dict(color=F_COLS[i])))
            fig.add_trace(go.Scatter(x=lt.index, y=lt[f"b{i+1}_P"], name=f"b{i+1} P",
                                     line=dict(color=F_COLS[i], dash="dash")))
        fig.update_layout(title="Factor loadings b(τ): Q (solid) vs P (dashed)", height=380,
                          xaxis_title="maturity (years)", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lt.index, y=pct(lt["a_Q"]), name="a(τ) Q", line=dict(color=COL["Q"])))
        fig.add_trace(go.Scatter(x=lt.index, y=pct(lt["a_P"]), name="a(τ) P", line=dict(color=COL["P"])))
        fig.add_trace(go.Scatter(x=lt.index, y=pct(lt["a_EH"]), name="a(τ) EH", line=dict(color=COL["EH"], dash="dot")))
        fig.update_layout(title="Intercepts a(τ), %", height=380, xaxis_title="maturity (years)",
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    st.markdown("**Correlation of latent factors with empirical level / slope / curvature**")
    st.dataframe(bundle["factor_corr"].style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1))
    explain(
        "Q loadings: $b^Q_i(\\tau) = (1-e^{-\\kappa_i\\tau})/(\\kappa_i\\tau)$, so a slow factor "
        "(small $\\kappa$) moves the whole curve (level-like) and fast factors only move the short "
        "end. $a^Q(\\tau) = \\delta_0 - \\text{convexity}(\\tau)$. P loadings come from the same "
        "pricing ODEs solved numerically with the physical drift $K^P(\\theta^P - x)$ "
        "(`affine.p_loadings`). The gap between the Q and P loadings is exactly what generates "
        "the term premium.")

# =============================================================================
# 4. Decomposition
# =============================================================================
with tabs[3]:
    st.header(f"{tp_m:g}-year term premium decomposition")
    st.markdown("**Term Premium = Pricing-Measure Fitted Yield − Physical-Measure Affine Yield**")
    ser = {"Observed": pct(y[tp_m]), "Q fitted yield": pct(dec["fitted_Q"][tp_m]),
           "P affine yield": pct(dec["affine_P"][tp_m])}
    show_eh = st.checkbox("Also show strict EH average expected short rate (reference)", value=False)
    if show_eh:
        ser["EH avg short rate"] = pct(dec["eh_short_rate"][tp_m])
    st.plotly_chart(line_fig(ser, "Yield and its components", dash={"EH avg short rate": "dot"}))
    tp_ser = {"TP (this model)": pct(dec["term_premium"][tp_m])}
    true_tp = bundle["benchmarks"].get("True TP (simulation)")
    if true_tp is not None and abs(tp_m - bundle["tp_maturity"]) < 1e-9:
        tp_ser["True TP (simulation)"] = pct(true_tp)
    fig = line_fig(tp_ser, "Term premium", dash={"True TP (simulation)": "dash"})
    fig.add_hline(y=0, line=dict(color="black", width=0.8))
    st.plotly_chart(fig)

    c1, c2 = st.columns(2)
    with c1:
        d_sel2 = st.select_slider("Date for term structure", options=list(y.index), value=y.index[-1],
                                  format_func=lambda d: d.strftime("%Y-%m"), key="tp_date")
        fig = go.Figure()
        for k, nm, col in (("fitted_Q", "Q fitted", COL["Q"]), ("affine_P", "P affine", COL["P"]),
                           ("term_premium", "Term premium", COL["TP"])):
            fig.add_trace(go.Scatter(x=tau, y=pct(dec[k].loc[d_sel2]), name=nm, line=dict(color=col)))
        fig.update_layout(title=f"Decomposition across maturities, {d_sel2:%Y-%m}", height=360,
                          xaxis_title="maturity (years)", yaxis_title="%", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    with c2:
        lo = bundle["loadings_obs"]
        fig = go.Figure()
        for i in range(3):
            fig.add_trace(go.Bar(x=[f"{m:g}y" for m in lo.index], y=lo[f"b{i+1}_TP"], name=f"x{i+1}",
                                 marker_color=F_COLS[i]))
        fig.update_layout(barmode="group", title="Term premium loadings b^Q(τ) − b^P(τ)", height=360,
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    conv = dec["convexity_P"][tp_m] * 1e4
    st.caption(f"P-measure convexity at {tp_m:g}y (EH average short rate − P affine yield): "
               f"mean {conv.mean():.1f}bp. This is the only difference between the benchmark used "
               "here and a strict EH expectations component.")
    explain(
        "Both components are affine in the same smoothed state: $y^Q = a^Q + b^{Q\\prime}x_t$ and "
        "$y^P = a^P + b^{P\\prime}x_t$, so the term premium is affine too, "
        "$TP_t = (a^Q-a^P) + (b^Q-b^P)'x_t$ (`decomposition.term_premium_decomposition`). "
        "The bar chart shows which latent factor drives variation in the premium. Under the market "
        "price of risk $\\Sigma\\lambda_t = K^P\\theta^P + (K^Q-K^P)x_t$, a zero term premium would "
        "require $K^P = K^Q$ and $\\theta^P = 0$. The P-affine yield equals the EH average short rate "
        "minus a small P-convexity term (shown in the caption).")

# =============================================================================
# 5. Benchmarks
# =============================================================================
with tabs[4]:
    st.header("Comparison with benchmark term premium series")
    bm = {k: v for k, v in bundle["benchmarks"].items() if v is not None}
    if not bm:
        st.info("No benchmark series in this run. For real data pass ACM (NY Fed) and Kim-Wright "
                "(FRED THREEFYTP10) files: `--acm data/ACMTermPremium.xls --kw data/THREEFYTP10.csv`.")
    else:
        mtp = dec["term_premium"][bundle["tp_maturity"]]
        ser = {"TP (this model)": pct(mtp)}
        for k, v in bm.items():
            ser[k] = pct(v.loc[mtp.index.min():mtp.index.max()])
        fig = line_fig(ser, f"{bundle['tp_maturity']:g}y term premium vs benchmarks", height=420,
                       dash={"True TP (simulation)": "dash"})
        fig.add_hline(y=0, line=dict(color="black", width=0.8))
        st.plotly_chart(fig)
        st.dataframe(bundle["benchmark_table"])
    explain(
        "ACM (Adrian-Crump-Moench) and KW (Kim-Wright) are published estimates from different "
        "models: ACM uses regression-based affine pricing on many yields; KW is a three-factor "
        "Gaussian model estimated with survey forecasts of short rates, which disciplines the P "
        "dynamics. This project has no survey information, so expect the *direction* of moves "
        "(change correlation) to agree better than the *level*. In simulation the benchmark is the "
        "true premium, which isolates estimation error from model error.")

# =============================================================================
# 6. Sensitivity
# =============================================================================
with tabs[5]:
    st.header("Sensitivity analysis")
    st.warning("Sliders **shock the estimated parameters and recompute model outputs at those "
               "values**. They do **not** re-estimate the model: nothing is re-optimised, and the "
               "log-likelihood change shows how much worse the shocked parameters fit the data.")
    c = st.columns(3)
    with c[0]:
        st.markdown("**Q measure (cross-section)**")
        kq = [st.slider(f"κQ{i+1} multiplier", 0.5, 2.0, 1.0, 0.05, key=f"kq{i}") for i in range(3)]
        d0 = st.slider("δ0 shift (bp)", -200, 200, 0, 10)
    with c[1]:
        st.markdown("**P measure (time series)**")
        kp = st.slider("K_P multiplier (speed of mean reversion)", 0.5, 2.0, 1.0, 0.05)
        lr = st.slider("P long-run short rate shift (bp)", -200, 200, 0, 10)
    with c[2]:
        st.markdown("**Volatility & states**")
        sg = st.slider("Σ multiplier", 0.5, 2.0, 1.0, 0.05)
        refilter = st.checkbox("Re-run Kalman filter/smoother at shocked parameters", value=not fast,
                               help="Off: states held at their estimated values (instant). On: the "
                                    "filter re-infers the states given the shocked parameters, "
                                    "still without re-optimising (slower).")

    q = p.copy()
    q.kappa_Q = p.kappa_Q * np.array(kq)
    q.delta0 = p.delta0 + d0 / 1e4
    q.K_P = p.K_P * kp
    q.theta_P = p.theta_P + np.array([lr / 1e4, 0, 0])
    q.Sigma = p.Sigma * sg
    if np.any(np.diff(q.kappa_Q) <= 0):
        st.info("κQ ordering changed. Outputs are still well defined, but this parameter vector is "
                "outside the identified region used for estimation.")
    dt = cfg_b.get("dt", 1 / 12)
    t0 = time.time()
    if refilter:
        fs = filter_and_smooth(q, y.values, tau, dt)
        st_shock = fs["x_smoothed"]
    else:
        st_shock = states.values
    dec_s = term_premium_decomposition(q, st_shock, tau, y.index)
    ll_s = loglik(q, y.values, tau, dt)
    elapsed = time.time() - t0

    m = st.columns(4)
    m[0].metric("Δ log-likelihood vs MLE", f"{ll_s - est['loglik']:,.1f}" if np.isfinite(ll_s) else "−∞ (non-stationary)")
    rm_b = np.sqrt(np.nanmean((y.values - dec["fitted_Q"].values) ** 2)) * 1e4
    rm_s = np.sqrt(np.nanmean((y.values - dec_s["fitted_Q"].values) ** 2)) * 1e4
    m[1].metric("Pricing RMSE", f"{rm_s:.1f} bp", f"{rm_s - rm_b:+.1f} bp", delta_color="inverse")
    tb, ts = dec["term_premium"][tp_m], dec_s["term_premium"][tp_m]
    m[2].metric(f"Mean {tp_m:g}y TP", f"{ts.mean()*100:.2f}%", f"{(ts.mean()-tb.mean())*1e4:+.0f} bp")
    m[3].metric(f"Latest {tp_m:g}y TP", f"{ts.iloc[-1]*100:.2f}%", f"{(ts.iloc[-1]-tb.iloc[-1])*1e4:+.0f} bp")
    st.caption(f"Recomputed in {elapsed*1000:.0f} ms ({'states re-filtered' if refilter else 'states held fixed'}).")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tb.index, y=pct(tb), name="Baseline TP", line=dict(color=COL["TP"])))
        fig.add_trace(go.Scatter(x=ts.index, y=pct(ts), name="Shocked TP", line=dict(color=COL["shock"], dash="dash")))
        fig.add_hline(y=0, line=dict(color="black", width=0.8))
        fig.update_layout(title=f"{tp_m:g}y term premium: baseline vs shocked", height=380,
                          yaxis_title="%", margin=dict(l=10, r=10, t=50, b=10), hovermode="x unified")
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tau, y=pct(y.mean()), mode="markers", name="Observed mean",
                                 marker=dict(color=COL["obs"])))
        for k, nm, col in (("fitted_Q", "Q fitted", COL["Q"]), ("affine_P", "P affine", COL["P"]),
                           ("term_premium", "TP", COL["TP"])):
            fig.add_trace(go.Scatter(x=tau, y=pct(dec[k].mean()), name=f"{nm} (base)", line=dict(color=col)))
            fig.add_trace(go.Scatter(x=tau, y=pct(dec_s[k].mean()), name=f"{nm} (shock)",
                                     line=dict(color=col, dash="dash")))
        fig.update_layout(title="Average curves: baseline (solid) vs shocked (dashed)", height=380,
                          xaxis_title="maturity (years)", yaxis_title="%", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig)
    explain(
        "Each slider edits one block of the parameter vector, then the same functions used after "
        "estimation are re-run: Q loadings (`q_loadings`), P loadings (`p_loadings`, ODE) and the "
        "decomposition. **Q sliders** (κQ, δ0, Σ via convexity) change how states map to yields, "
        "so with states held fixed the fit deteriorates immediately: the cross-section pins these "
        "parameters down tightly. **P sliders** (K_P, long-run rate) leave the fitted yields "
        "unchanged with states fixed and move only the P-affine yield and hence the term "
        "premium, while the likelihood changes relatively little: the P side is weakly "
        "identified, which is why term premium *levels* are model-dependent.\n\n"
        "This is sensitivity analysis, not re-estimation: a proper re-estimation would let every "
        "other parameter adjust to the shock by maximising the likelihood again (minutes, not "
        "milliseconds). Use the CLI or the sidebar 'Estimate' modes for that.")

# =============================================================================
# 7. Estimation & diagnostics
# =============================================================================
with tabs[6]:
    st.header("Estimation details and diagnostics")
    c = st.columns(4)
    c[0].metric("Log-likelihood", f"{est['loglik']:,.2f}")
    c[1].metric("AIC / BIC", f"{est['aic']:,.0f} / {est['bic']:,.0f}")
    c[2].metric("Parameters / obs", f"{est['n_params']} / {est['n_obs']}")
    c[3].metric("Optimiser iterations", f"{est['n_iter']}")
    st.caption(f"Optimiser message: {est['message']}")

    tr = ParamTransform(len(tau), cfg_b.get("common_measurement_error", False))
    tab = pd.DataFrame({"start (2-stage)": tr.natural_vector(bundle["start_params"]),
                        "MLE": tr.natural_vector(p)}, index=tr.natural_names)
    if bundle.get("std_errors") is not None:
        tab["std error"] = bundle["std_errors"]["std_error"].values
        tab["t-stat"] = bundle["std_errors"]["t_stat"].values
    else:
        st.info("Standard errors were not computed (fast mode or --no-se). Run in full mode to get "
                "inverse-Hessian / delta-method standard errors.")
    if is_sim and bundle.get("true_params") is not None:
        tab.insert(0, "true", tr.natural_vector(bundle["true_params"]))
    st.dataframe(tab.style.format("{:.5f}"), height=420)

    c1, c2 = st.columns(2)
    hl = p.factor_half_lives()
    lr_ = p.short_rate_long_run()
    with c1:
        st.markdown("**Persistence and long-run means**")
        st.dataframe(pd.DataFrame({"Q half-life (y)": hl["Q"], "P half-life (y)": hl["P"]},
                                  index=["factor/eig 1", "factor/eig 2", "factor/eig 3"]).style.format("{:.2f}"))
        st.write(f"Long-run short rate: Q **{lr_['Q']*100:.2f}%**, P **{lr_['P']*100:.2f}%**")
    with c2:
        mpr = p.market_price_of_risk()
        st.markdown("**Market prices of risk**  λ_t = λ0 + λ1 x_t")
        st.dataframe(pd.DataFrame(np.column_stack([mpr["lambda0"], mpr["lambda1"]]),
                                  columns=["λ0", "λ1·x1", "λ1·x2", "λ1·x3"]).style.format("{:.3f}"))
    st.markdown("**Standardised one-step-ahead prediction errors** (should be ≈ mean 0, s.d. 1, low AR(1))")
    st.dataframe(bundle["innovations"].style.format("{:.3f}"))
    if est.get("history") and len(est["history"]) > 1:
        st.write("Log-likelihood by restart:", [round(h, 2) for h in est["history"]])
    explain(
        "Estimation runs in three stages (`estimation.py`): (1) cross-sectional least squares for "
        "κQ and δ0 with per-date OLS states; (2) a VAR(1) on those states mapped to $K^P$, "
        "$\\theta^P$ and $\\Sigma$ via the matrix logarithm and an exact Van Loan inversion; "
        "(3) joint MLE of all parameters using the Kalman-filter prediction-error likelihood "
        "(computed with the collapsed Jungbacker-Koopman filter and a steady-state shortcut). "
        "Standard errors are from the inverse numerical Hessian in the unconstrained space, "
        "mapped back by the delta method. Large standard errors on $K^P$ relative to $\\kappa^Q$ "
        "are the quantitative statement of 'P is weakly identified'.")
