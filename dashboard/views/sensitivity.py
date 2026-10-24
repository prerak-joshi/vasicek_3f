"""Sensitivity analysis: shock estimated parameters and recompute (no re-estimation)."""

from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from context import Ctx, explain, pct
from theme import COL
from vasicek3f.decomposition import term_premium_decomposition
from vasicek3f.model import filter_and_smooth, loglik


def render(ctx: Ctx) -> None:
    p, y, tau, m = ctx.params, ctx.yields, ctx.tau, ctx.tp_m
    st.warning(
        "Sliders **shock the estimated (MLE) parameters and recompute** the outputs. Nothing is "
        "re-optimised; the log-likelihood change shows how much worse the shocked parameters fit."
    )
    c = st.columns(3)
    with c[0]:
        st.markdown("**Q measure (cross-section)**")
        kq = [st.slider(f"κᵠ{i + 1} multiplier", 0.5, 2.0, 1.0, 0.05, key=f"kq{i}") for i in range(3)]
        d0 = st.slider("δ₀ shift (bp)", -200, 200, 0, 10)
    with c[1]:
        st.markdown("**P measure (time series)**")
        kp = st.slider("Kᴾ multiplier (mean-reversion speed)", 0.5, 2.0, 1.0, 0.05)
        lr = st.slider("P long-run short rate shift (bp)", -200, 200, 0, 10)
    with c[2]:
        st.markdown("**Volatility & states**")
        sg = st.slider("Σ multiplier", 0.5, 2.0, 1.0, 0.05)
        refilter = st.checkbox(
            "Re-run Kalman filter/smoother at shocked parameters",
            value=not ctx.fast,
            help="Off: states held at their estimated values (instant). On: states are "
            "re-inferred under the shocked parameters, still without re-optimising.",
        )
    q = p.copy()
    q.kappa_Q = p.kappa_Q * np.array(kq)
    q.delta0 = p.delta0 + d0 / 1e4
    q.K_P = p.K_P * kp
    q.theta_P = p.theta_P + np.array([lr / 1e4, 0, 0])
    q.Sigma = p.Sigma * sg
    if np.any(np.diff(q.kappa_Q) <= 0):
        st.info(
            "κᵠ ordering changed: outputs are still defined, but this point lies outside the "
            "identified region used for estimation."
        )
    t0 = time.time()
    states = filter_and_smooth(q, y.values, tau, ctx.dt)["x_smoothed"] if refilter else ctx.states.values
    dec_s = term_premium_decomposition(q, states, tau, y.index)
    ll_s = loglik(q, y.values, tau, ctx.dt)
    elapsed = time.time() - t0
    base = ctx.dec_mle

    k = st.columns(4)
    k[0].metric(
        "Δ log-likelihood vs MLE",
        f"{ll_s - ctx.est['loglik']:,.1f}" if np.isfinite(ll_s) else "−∞ (non-stationary)",
    )
    rb = np.sqrt(np.nanmean((y.values - base["fitted_Q"].values) ** 2)) * 1e4
    rs = np.sqrt(np.nanmean((y.values - dec_s["fitted_Q"].values) ** 2)) * 1e4
    k[1].metric("Pricing RMSE", f"{rs:.1f} bp", f"{rs - rb:+.1f} bp", delta_color="inverse")
    tb, ts = base["term_premium"][m], dec_s["term_premium"][m]
    k[2].metric(
        f"Mean {m:g}y TP",
        f"{ts.mean() * 100:.2f}%",
        f"{(ts.mean() - tb.mean()) * 1e4:+.0f} bp",
        delta_color="off",
    )
    k[3].metric(
        f"Latest {m:g}y TP",
        f"{ts.iloc[-1] * 100:.2f}%",
        f"{(ts.iloc[-1] - tb.iloc[-1]) * 1e4:+.0f} bp",
        delta_color="off",
    )
    st.caption(
        f"Recomputed in {elapsed * 1000:.0f} ms ({'states re-filtered' if refilter else 'states held fixed'})."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=tb.index, y=pct(tb), name="Baseline TP (MLE)", line=dict(color=COL["TP"]))
        fig.add_scatter(x=ts.index, y=pct(ts), name="Shocked TP", line=dict(color=COL["Shock"], dash="dash"))
        fig.add_hline(y=0, line=dict(color="#7d8592", width=0.9))
        fig.update_layout(title=f"{m:g}-year term premium: baseline vs shocked", height=380, yaxis_title="%")
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_scatter(
            x=tau,
            y=pct(y.mean()).values,
            mode="markers",
            name="Observed mean",
            marker=dict(color=COL["Observed"]),
        )
        for key, nm, col in (
            ("fitted_Q", "Q fitted", COL["Q"]),
            ("affine_P", "P affine", COL["P"]),
            ("term_premium", "TP", COL["TP"]),
        ):
            fig.add_scatter(x=tau, y=pct(base[key].mean()).values, name=f"{nm} (base)", line=dict(color=col))
            fig.add_scatter(
                x=tau,
                y=pct(dec_s[key].mean()).values,
                name=f"{nm} (shock)",
                line=dict(color=col, dash="dash"),
            )
        fig.update_layout(
            title="Average curves: baseline (solid) vs shocked (dashed)",
            height=420,
            xaxis_title="maturity (years)",
            yaxis_title="%",
            hovermode="x",
            legend=dict(orientation="h", y=-0.28, yanchor="top"),
        )
        st.plotly_chart(fig)
    explain(
        "Each slider edits one block of the MLE parameter vector and re-runs the same pricing "
        "(`q_loadings`, `p_loadings`) and optionally filtering functions used after estimation. **Q "
        "sliders** change how states map to yields, so with states fixed the fit deteriorates sharply: "
        "the cross-section pins these parameters down. **P sliders** leave fitted yields unchanged and "
        "move only the P affine yield and the term premium, at a small cost in likelihood: the P side "
        "is weakly identified, which is why term premium *levels* are model-dependent.\n\n"
        "This is sensitivity analysis, not re-estimation: re-estimation would let every other parameter "
        "adjust by maximising the likelihood again (minutes, not milliseconds)."
    )
