"""Latent factors and factor loadings."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from context import Ctx, explain, pct
from theme import COL, FACTOR, add_recessions
from vasicek3f.decomposition import loadings_table


def render(ctx: Ctx) -> None:
    states = ctx.states
    sd = np.sqrt(np.stack([np.diag(P) for P in ctx.bundle["states_cov"]]))
    hl = ctx.params.factor_half_lives()["Q"]
    titles = [f"x{i + 1}  ·  κᵠ={ctx.params.kappa_Q[i]:.3f} (Q half-life {hl[i]:.1f}y)" for i in range(3)]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=titles)
    for i in range(3):
        s = pct(states.iloc[:, i])
        fig.add_scatter(
            x=s.index,
            y=s + 200 * sd[:, i],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            row=i + 1,
            col=1,
        )
        fig.add_scatter(
            x=s.index,
            y=s - 200 * sd[:, i],
            fill="tonexty",
            line=dict(width=0),
            fillcolor="rgba(31,95,168,0.15)",
            showlegend=False,
            hoverinfo="skip",
            row=i + 1,
            col=1,
        )
        fig.add_scatter(
            x=s.index, y=s, line=dict(color=FACTOR[i], width=1.5), name=f"x{i + 1}", row=i + 1, col=1
        )
        ts = ctx.bundle.get("true_states")
        if ctx.is_sim and ts is not None:
            fig.add_scatter(
                x=ts.index,
                y=pct(ts.iloc[:, i]),
                line=dict(color="#999", dash="dot", width=1),
                name=f"true x{i + 1}",
                row=i + 1,
                col=1,
            )
        if ctx.is_us:
            add_recessions(fig, s.index, row=i + 1, col=1)
    fig.update_layout(height=640, title="Smoothed latent factors (±2 s.d.), %")
    st.plotly_chart(fig)

    lt = loadings_table(ctx.params, np.linspace(0.25, max(10.0, ctx.tau.max()), 20 if ctx.fast else 60))
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for i in range(3):
            fig.add_scatter(x=lt.index, y=lt[f"b{i + 1}_Q"], name=f"b{i + 1} Q", line=dict(color=FACTOR[i]))
            fig.add_scatter(
                x=lt.index, y=lt[f"b{i + 1}_P"], name=f"b{i + 1} P", line=dict(color=FACTOR[i], dash="dash")
            )
        fig.update_layout(
            title="Factor loadings b(τ): Q solid, P dashed",
            height=380,
            xaxis_title="maturity (years)",
            hovermode="x",
        )
        st.plotly_chart(fig)
    with c2:
        fig = go.Figure()
        fig.add_scatter(x=lt.index, y=pct(lt["a_Q"]), name="a(τ) Q", line=dict(color=COL["Q"]))
        fig.add_scatter(x=lt.index, y=pct(lt["a_P"]), name="a(τ) P", line=dict(color=COL["P"]))
        fig.add_scatter(x=lt.index, y=pct(lt["a_EH"]), name="a(τ) EH", line=dict(color=COL["EH"], dash="dot"))
        fig.update_layout(
            title="Intercepts a(τ), %", height=380, xaxis_title="maturity (years)", hovermode="x"
        )
        st.plotly_chart(fig)
    st.markdown("**Correlation of latent factors with empirical level / slope / curvature**")
    st.dataframe(
        ctx.bundle["factor_corr"].style.format("{:.2f}").background_gradient(cmap="RdBu", vmin=-1, vmax=1)
    )
    explain(
        "States are $\\hat x_{t|T}$ from the RTS smoother with bands from $P_{t|T}$. They are identified "
        "only through the canonical normalisation ($K^Q$ diagonal and ordered, $\\theta^Q=0$, "
        "$\\delta_1=\\iota$, $\\Sigma$ lower triangular), so they are rotations of level/slope/curvature. "
        "Q loadings: $b_i(\\tau)=(1-e^{-\\kappa_i\\tau})/(\\kappa_i\\tau)$. P loadings solve the same "
        "pricing ODEs with the physical drift $K^P(\\theta^P-x)$; the Q–P gap generates the term premium."
    )
