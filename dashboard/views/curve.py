"""Yield curve fit: time series, curve snapshots and pricing errors."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from context import Ctx, csv_download, explain, pct
from theme import COL, ts_figure


def render(ctx: Ctx) -> None:
    y, dec, tau = ctx.yields, ctx.dec_mle, ctx.tau
    a, b = st.columns([3, 2])
    with a:
        m = st.selectbox(
            "Maturity", list(tau), index=len(tau) - 1, format_func=lambda v: f"{v:g}y", key="fit_m"
        )
        fig = ts_figure(
            {"Observed": pct(y[m]), "Model (Q, smoothed states)": pct(dec["fitted_Q"][m])},
            f"{m:g}-year yield: observed vs model",
            styles={
                "Observed": {"color": COL["Observed"], "width": 1.3},
                "Model (Q, smoothed states)": {"color": COL["Q"], "dash": "dash"},
            },
            recessions=ctx.is_us,
        )
        st.plotly_chart(fig)
    with b:
        dates = list(y.index)
        d1, d2 = st.select_slider(
            "Compare curves",
            options=dates,
            value=(dates[max(0, len(dates) - 13)], dates[-1]),
            format_func=lambda d: d.strftime("%Y-%m"),
            key="curve_dates",
        )
        fig = go.Figure()
        for d, col, dash in ((d1, "#9aa4b2", "dot"), (d2, COL["Q"], None)):
            fig.add_scatter(
                x=tau,
                y=pct(y.loc[d]).values,
                mode="markers",
                name=f"Observed {d:%Y-%m}",
                marker=dict(color=col, size=8),
            )
            fig.add_scatter(
                x=tau,
                y=pct(dec["fitted_Q"].loc[d]).values,
                mode="lines",
                name=f"Model {d:%Y-%m}",
                line=dict(color=col, dash=dash),
            )
        fig.update_layout(
            title="Yield curve snapshots",
            height=380,
            xaxis_title="maturity (years)",
            yaxis_title="%",
            hovermode="x",
        )
        st.plotly_chart(fig)

    err = (y - dec["fitted_Q"]) * 1e4
    lim = float(np.nanpercentile(np.abs(err.values), 99))
    fig = go.Figure(
        go.Heatmap(
            z=err.T.values,
            x=err.index,
            y=[f"{t:g}y" for t in tau],
            colorscale="RdBu_r",
            zmin=-lim,
            zmax=lim,
            colorbar=dict(title="bp"),
        )
    )
    fig.update_layout(title="Pricing errors (observed − model), bp", height=300, hovermode="closest")
    st.plotly_chart(fig)
    pe = ctx.bundle["pricing_errors"]
    c1, c2 = st.columns([3, 2])
    with c1:
        st.dataframe(pe.style.format("{:.2f}"))
    with c2:
        st.metric("Average RMSE across maturities", f"{pe['rmse_bp'].mean():.1f} bp")
        st.metric("Estimated measurement error (avg s.d.)", f"{ctx.params.sigma_e.mean() * 1e4:.1f} bp")
        csv_download(err, "pricing_errors_bp.csv", "Download pricing errors (CSV, bp)")
    explain(
        "Model yields are $\\hat y_t(\\tau)=a^Q(\\tau)+b^Q(\\tau)'\\hat x_{t|T}$: closed-form Q loadings "
        "(`affine.q_loadings`) applied to RTS-smoothed states. The errors are the measurement noise "
        "$e_t$, whose standard deviations are MLE parameters. Small, weakly autocorrelated errors mean "
        "three factors span the cross-section; persistent maturity-specific patterns would indicate a "
        "missing factor (GSW yields are themselves fitted curves, which keeps errors small)."
    )
