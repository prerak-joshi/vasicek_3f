"""
3-Factor Vasicek term structure dashboard.

    streamlit run streamlit_app.py        (or: streamlit run dashboard/app.py)

Layout and navigation live here; each tab is rendered by a module in dashboard/views/.
All model logic comes from the vasicek3f package in src/ — the dashboard only reads
results bundles (or runs the pipeline) and applies sensitivity shocks.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, os.path.join(ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

st.set_page_config(
    page_title="Vasicek 3F · Term Structure & Term Premium",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

import theme  # noqa: E402
from context import (  # noqa: E402
    DEMO_CFG,
    cached_run,
    demo_bundle,
    load_saved,
    make_ctx,
    run_label,
    save_upload,
    saved_runs,
)
from vasicek3f.pipeline import RunConfig  # noqa: E402
from views import (  # noqa: E402
    curve,
    factors,
    forecasting,
    methodology,
    model,
    sensitivity,
    summary,
    term_premium,
)

theme.inject_css()

# =============================================================================
# Sidebar: data source and global options
# =============================================================================
st.sidebar.markdown("### Vasicek 3F")
st.sidebar.caption("Gaussian affine term structure · Kalman filter MLE · term premium")

runs = saved_runs()
SOURCES = ["Saved results", "Estimate on simulated data", "Estimate on GSW data"]
source = st.sidebar.radio("Results source", SOURCES, index=0 if runs else 1)
fast = st.sidebar.toggle(
    "Fast mode",
    value=True,
    help="Estimation: no standard errors or bands, looser tolerance, no polish. "
    "Dashboard: states held fixed in sensitivity, coarser loading grid.",
)

bundle, note = None, None
if source == "Saved results":
    if not runs:
        st.sidebar.warning("No saved results in results/.")
    else:
        name = st.sidebar.selectbox("Run", list(runs), format_func=lambda n: run_label(n, runs[n]))
        try:
            bundle = load_saved(runs[name])
        except Exception:  # noqa: BLE001
            note = (
                "Saved results could not be read with the installed library versions; "
                "showing a fresh fast-mode demo estimation instead."
            )
            with st.spinner("Running the demo estimation (about 10–20 s, once) ..."):
                bundle = demo_bundle()
else:
    with st.sidebar.form("run"):
        if source == "Estimate on simulated data":
            periods = st.number_input("Months to simulate", 120, 1200, 480, step=60)
            seed = st.number_input("Random seed", 0, 10_000, 42)
            cfg = RunConfig(
                **{**asdict(DEMO_CFG), "sim_periods": int(periods), "seed": int(seed), "fast": fast}
            )
        else:
            st.caption("Upload files (works on Streamlit Cloud) or use local paths.")
            up_g = st.file_uploader("GSW feds200628.csv", type=["csv"])
            up_a = st.file_uploader("ACM term premium (optional)", type=["xls", "xlsx", "csv"])
            up_k = st.file_uploader("Kim-Wright THREEFYTP10.csv (optional)", type=["csv"])
            gsw = save_upload(up_g, "gsw") or os.path.join(ROOT, "data", "feds200628.csv")
            acm = save_upload(up_a, "acm") or os.path.join(ROOT, "data", "ACMTermPremium.xls")
            kw = save_upload(up_k, "kw") or os.path.join(ROOT, "data", "THREEFYTP10.csv")
            start = st.text_input("Sample start (YYYY-MM)", "1990-01")
            cfg = RunConfig(
                **{
                    **asdict(DEMO_CFG),
                    "source": "gsw",
                    "gsw_path": gsw,
                    "acm_path": acm if os.path.exists(acm) else None,
                    "kw_path": kw if os.path.exists(kw) else None,
                    "start": start,
                    "fast": fast,
                    "out_dir": DEMO_CFG.out_dir.replace("demo", "gsw"),
                }
            )
        go = st.form_submit_button("Run estimation", type="primary")
    if go:
        st.session_state["cfg"] = json.dumps(asdict(cfg))
    if "cfg" in st.session_state:
        with st.spinner(
            "Estimating: starting values → Kalman-filter MLE → decomposition"
            + ("" if fast else " → standard errors and bands")
            + " ..."
        ):
            try:
                bundle = cached_run(st.session_state["cfg"])
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"Estimation failed: {e}")

if bundle is None:
    theme.header(
        "3-Factor Vasicek Term Structure Model", "Choose a saved run or start an estimation from the sidebar."
    )
    st.stop()

tau = [float(t) for t in bundle["maturities"]]
default_m = bundle["tp_maturity"] if bundle["tp_maturity"] in tau else tau[-1]
tp_m = st.sidebar.selectbox(
    "Focus maturity", tau, index=tau.index(default_m), format_func=lambda v: f"{v:g}y"
)
has_bc = bundle.get("decomposition_bc") is not None
use_bc = st.sidebar.toggle(
    "Bias-corrected P dynamics",
    value=False,
    disabled=not has_bc,
    help="Show the term premium computed with small-sample bias-corrected "
    "P dynamics (Kilian bootstrap) instead of the MLE P dynamics.",
)
ctx = make_ctx(bundle, tp_m, use_bc, fast)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Run:** {ctx.cfg['source']} · {len(ctx.yields)} months · {len(ctx.tau)} maturities  \n"
    f"**Estimated in** {bundle['runtime_sec']:.0f}s{' (fast mode)' if ctx.cfg.get('fast') else ''}  \n"
    f"[Docs & code](https://github.com/prerak-joshi/vasicek_3f)"
)

# =============================================================================
# Header and tabs
# =============================================================================
src = {
    "gsw": "US Treasury zero-coupon yields (Gürkaynak-Sack-Wright)",
    "simulated": "Simulated yields from known parameters",
}.get(ctx.cfg["source"], ctx.cfg["source"])
chips = [
    (f"{ctx.yields.index[0]:%b %Y} – {ctx.yields.index[-1]:%b %Y}", False),
    (f"{len(ctx.tau)} maturities {ctx.tau.min():g}–{ctx.tau.max():g}y", False),
    ("3-factor Gaussian A0(3), canonical", False),
    ("bias-corrected P" if ctx.use_bc else "MLE P dynamics", ctx.use_bc),
]
if ctx.is_sim:
    chips.append(("simulation: truth known", True))
theme.header("Term Structure & Term Premium — 3-Factor Vasicek", src, chips)
if note:
    st.info(note)

TABS = [
    ("Summary", summary),
    ("Yield curve", curve),
    ("Term premium", term_premium),
    ("Factors", factors),
    ("Forecasting", forecasting),
    ("Sensitivity", sensitivity),
    ("Model & estimation", model),
    ("Methodology", methodology),
]
for tab, (_, view) in zip(st.tabs([t for t, _ in TABS]), TABS):
    with tab:
        view.render(ctx)

st.markdown(
    '<p class="vt-note" style="margin-top:2rem">Educational model. The term premium is a '
    "model-based decomposition (pricing-measure fitted yield minus physical-measure affine "
    "yield), sensitive to weakly identified P dynamics. Not investment advice.</p>",
    unsafe_allow_html=True,
)
