"""Summary: headline numbers, term premium history, and attribution of yield changes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from context import Ctx, csv_download, explain, pct
from theme import COL, takeaways, ts_figure

WINDOWS = {"1 month": 1, "3 months": 3, "6 months": 6, "12 months": 12, "24 months": 24}


def _chg(s: pd.Series, k: int) -> float:
    return float(s.iloc[-1] - s.iloc[-1 - k]) * 1e4 if len(s) > k else np.nan


def render(ctx: Ctx) -> None:
    y, tp, m = ctx.yields, ctx.tp, ctx.tp_m
    last = y.index[-1]
    pctile = float((tp <= tp.iloc[-1]).mean())

    c = st.columns(5)
    c[0].metric(
        f"{m:g}y yield", f"{y[m].iloc[-1] * 100:.2f}%", f"{_chg(y[m], 1):+.0f} bp 1m", delta_color="off"
    )
    c[1].metric(
        f"{m:g}y term premium", f"{tp.iloc[-1] * 100:.2f}%", f"{_chg(tp, 1):+.0f} bp 1m", delta_color="off"
    )
    c[2].metric("TP change 3m / 12m", f"{_chg(tp, 3):+.0f} / {_chg(tp, 12):+.0f} bp")
    c[3].metric(
        "Historical percentile",
        f"{pctile:.0%}",
        help=f"Share of months since {y.index[0]:%Y-%m} with a lower or equal term premium.",
    )
    if ctx.bands is not None:
        lo, hi = ctx.bands["q05"][m].iloc[-1], ctx.bands["q95"][m].iloc[-1]
        c[4].metric("90% band (MLE)", f"{lo * 100:.2f} – {hi * 100:.2f}%")
    elif ctx.dec_bc is not None:
        other = (ctx.dec_mle if ctx.use_bc else ctx.dec_bc)["term_premium"][m].iloc[-1]
        c[4].metric(
            "Other P estimate",
            f"{other * 100:.2f}%",
            help="MLE vs bias-corrected P dynamics; switch in the sidebar.",
        )
    else:
        c[4].metric("Sample", f"{y.index[0]:%Y-%m} – {last:%Y-%m}")

    # --- key takeaways (data-driven)
    lines = []
    k = 12 if len(y) > 12 else len(y) - 1
    dy = _chg(y[m], k)
    dP = _chg(ctx.dec["affine_P"][m], k)
    dT = _chg(tp, k)
    lines.append(
        f"The {m:g}-year term premium is <b>{tp.iloc[-1] * 100:.2f}%</b> at {last:%b %Y}, "
        f"the <b>{pctile:.0%}</b> percentile of its history since {y.index[0]:%Y}."
    )
    lines.append(
        f"Over the last {k} months the {m:g}y yield moved <b>{dy:+.0f} bp</b>: "
        f"<b>{dP:+.0f} bp</b> from the physical-measure (expectations) component and "
        f"<b>{dT:+.0f} bp</b> from the term premium."
    )
    if ctx.dec_bc is not None:
        mle, bc = ctx.dec_mle["term_premium"][m], ctx.dec_bc["term_premium"][m]
        lines.append(
            f"Correcting the P dynamics for small-sample bias moves the latest estimate "
            f"from {mle.iloc[-1] * 100:.2f}% to {bc.iloc[-1] * 100:.2f}% "
            f"(sample means {mle.mean() * 100:.2f}% vs {bc.mean() * 100:.2f}%). "
            "The level of the term premium is model-dependent; its changes are more robust."
        )
    for name, s in ctx.benchmarks.items():
        j = pd.concat([ctx.dec_mle["term_premium"][ctx.bundle["tp_maturity"]], s], axis=1).dropna()
        if len(j) > 24:
            cc = j.iloc[:, 0].diff(12).corr(j.iloc[:, 1].diff(12))
            lines.append(f"Correlation of 12-month changes with <b>{name}</b>: {cc:.2f}.")
    oos = ctx.bundle.get("oos")
    if oos is not None:
        s = oos["summary"]
        s12 = s[s["horizon_m"] == max(oos["horizons"])]
        lines.append(
            f"Out of sample, the model beats a random walk at the {int(s12['horizon_m'].iloc[0])}-month "
            f"horizon for {int((s12['ratio'] < 1).sum())} of {len(s12)} maturities "
            f"(median RMSE ratio {s12['ratio'].median():.2f})."
        )
    takeaways(lines)

    # --- term premium history
    series = {ctx.tp_label: pct(tp)}
    styles = {ctx.tp_label: {"color": COL["TP_BC"] if ctx.use_bc else COL["TP"], "width": 2.0}}
    if ctx.dec_bc is not None and not ctx.use_bc:
        series["TP (bias-corrected P)"] = pct(ctx.dec_bc["term_premium"][m])
        styles["TP (bias-corrected P)"] = {"color": COL["TP_BC"], "dash": "dot", "width": 1.4}
    if abs(m - ctx.bundle["tp_maturity"]) < 1e-9:
        for name, s in ctx.benchmarks.items():
            key = "True TP (simulation)" if name.startswith("True") else name
            series[key] = pct(s.loc[y.index.min() : y.index.max()])
            styles[key] = {
                "color": COL.get(name.split()[0], COL["True"]),
                "width": 1.3,
                "dash": "dash" if key.startswith("True") else None,
            }
    band = None
    if ctx.bands is not None and not ctx.use_bc:
        band = (pct(ctx.bands["q05"][m]), pct(ctx.bands["q95"][m]), "90% parameter band")
    fig = ts_figure(
        series,
        f"{m:g}-year term premium",
        styles=styles,
        band=band,
        recessions=ctx.is_us,
        zero_line=True,
        height=420,
    )
    st.plotly_chart(fig)

    # --- attribution of yield changes across the curve
    st.subheader("What moved the curve?")
    a, b = st.columns([1, 3])
    with a:
        win = st.radio("Window", list(WINDOWS), index=3, key="attr_win")
        kk = min(WINDOWS[win], len(y) - 1)
        st.caption(f"{y.index[-1 - kk]:%b %Y} → {last:%b %Y}")
    obs = (y.iloc[-1] - y.iloc[-1 - kk]) * 1e4
    dPc = (ctx.dec["affine_P"].iloc[-1] - ctx.dec["affine_P"].iloc[-1 - kk]) * 1e4
    dTc = (ctx.dec["term_premium"].iloc[-1] - ctx.dec["term_premium"].iloc[-1 - kk]) * 1e4
    res = obs - dPc - dTc
    lab = [f"{t:g}y" for t in ctx.tau]
    with b:
        fig = go.Figure()
        fig.add_bar(x=lab, y=dPc.values, name="Expectations (P affine)", marker_color=COL["P"])
        fig.add_bar(
            x=lab, y=dTc.values, name="Term premium", marker_color=COL["TP_BC"] if ctx.use_bc else COL["TP"]
        )
        fig.add_bar(x=lab, y=res.values, name="Pricing error", marker_color="#b8bec8")
        fig.add_scatter(
            x=lab,
            y=obs.values,
            mode="markers",
            name="Observed change",
            marker=dict(color="black", size=9, symbol="diamond"),
        )
        fig.update_layout(
            barmode="relative",
            height=360,
            yaxis_title="bp",
            title=f"Change in yields over {win}, decomposed",
            hovermode="x",
        )
        st.plotly_chart(fig)
    table = pd.DataFrame(
        {"observed_bp": obs, "expectations_bp": dPc, "term_premium_bp": dTc, "pricing_error_bp": res}
    )
    with st.expander("Attribution table"):
        st.dataframe(table.style.format("{:+.1f}"))
        csv_download(table, f"yield_change_attribution_{kk}m.csv")
    explain(
        "Every fitted yield splits exactly as $y^Q_t = y^P_t + TP_t$, so a change in the observed yield "
        "decomposes into $\\Delta y^P$ (the no-risk-premium, expectations-type component under the "
        "physical dynamics), $\\Delta TP$, and the change in the pricing error. All three are affine in "
        "the smoothed states. The percentile and band use the full sample; the band reflects parameter "
        "uncertainty from the inverse Hessian (MLE P dynamics)."
    )
    d = (
        pd.DataFrame(
            {
                "observed": y[m],
                "fitted_Q": ctx.dec["fitted_Q"][m],
                "affine_P": ctx.dec["affine_P"][m],
                "term_premium": tp,
            }
        )
        * 100
    )
    csv_download(
        d,
        f"decomposition_{m:g}y_{'bc' if ctx.use_bc else 'mle'}_pct.csv",
        f"Download {m:g}y decomposition (CSV, %)",
    )
