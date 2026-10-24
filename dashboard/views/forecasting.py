"""Forecasting: expected yield paths and the out-of-sample evaluation."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from context import Ctx, csv_download, explain, pct
from theme import COL
from vasicek3f.forecast import forecast_yields


def render(ctx: Ctx) -> None:
    p = ctx.params if not ctx.use_bc else ctx.bundle["params_bc"]
    x_t = ctx.bundle["states_filtered"].iloc[-1].values
    last = ctx.yields.index[-1]
    st.markdown(
        f"**Model-expected yield curves from {last:%b %Y}** "
        f"(filtered state, {'bias-corrected' if ctx.use_bc else 'MLE'} P dynamics)"
    )
    fig = go.Figure()
    fig.add_scatter(
        x=ctx.tau,
        y=pct(ctx.yields.iloc[-1]).values,
        mode="markers",
        name="Observed today",
        marker=dict(color=COL["Observed"], size=8),
    )
    shades = ["#1f5fa8", "#4f86c6", "#86afdc", "#b9d0ec"]
    for h, col in zip((0, 12, 36, 120), shades):
        fig.add_scatter(
            x=ctx.tau,
            y=pct(forecast_yields(p, x_t, ctx.tau, h, ctx.dt)),
            name="Fitted today" if h == 0 else f"Expected in {h // 12}y",
            mode="lines",
            line=dict(color=col, width=2.2 if h == 0 else 1.6),
        )
    fig.update_layout(
        height=380,
        xaxis_title="maturity (years)",
        yaxis_title="%",
        title="Expected future yield curves under P",
        hovermode="x",
    )
    st.plotly_chart(fig)

    oos = ctx.bundle.get("oos")
    st.subheader("Out-of-sample forecast evaluation")
    if oos is None:
        st.info(
            "This run has no out-of-sample evaluation. Re-run the pipeline with `--oos` "
            "(expanding window, refit every 12 months; takes a few minutes)."
        )
    else:
        s = oos["summary"]
        st.caption(
            f"Expanding window starting with the first {oos['initial_window']} months, "
            f"parameters re-estimated every {oos['refit_every']} months (warm start), forecasts "
            "from the filtered state at each origin. Benchmark: random walk (no change)."
        )
        piv = s.pivot(index="horizon_m", columns="maturity", values="ratio")
        fig = go.Figure(
            go.Heatmap(
                z=piv.values,
                x=[f"{c:g}y" for c in piv.columns],
                y=[f"{h}m" for h in piv.index],
                zmid=1.0,
                colorscale="RdYlGn_r",
                text=np.round(piv.values, 2),
                texttemplate="%{text}",
                colorbar=dict(title="RMSE ratio"),
            )
        )
        fig.update_layout(
            title="RMSE ratio, model / random walk (< 1 = model better)", height=260, hovermode="closest"
        )
        st.plotly_chart(fig)
        c1, c2 = st.columns([1, 2])
        err = oos["errors"]
        with c1:
            h = st.selectbox("Horizon (months)", oos["horizons"], index=len(oos["horizons"]) - 1)
            mm = st.selectbox(
                "Maturity",
                list(ctx.tau),
                index=len(ctx.tau) - 1,
                format_func=lambda v: f"{v:g}y",
                key="oos_m",
            )
            row = s[(s["horizon_m"] == h) & (s["maturity"] == mm)].iloc[0]
            st.metric(
                "RMSE model / RW",
                f"{row['rmse_model_bp']:.0f} / {row['rmse_rw_bp']:.0f} bp",
                f"ratio {row['ratio']:.2f}",
                delta_color="off",
            )
            st.metric(
                "Diebold-Mariano",
                f"{row['dm_stat']:.2f}",
                help="Negative favours the model; |DM| > 1.96 ≈ significant at 5%.",
            )
        with c2:
            g = err[(err["horizon"] == h) & (err["maturity"] == mm)].set_index("origin")
            d = ((g["actual"] - g["rw"]) ** 2 - (g["actual"] - g["model"]) ** 2).cumsum() * 1e8
            fig = go.Figure(go.Scatter(x=d.index, y=d.values, line=dict(color=COL["Q"])))
            fig.add_hline(y=0, line=dict(color="#7d8592", width=0.9))
            fig.update_layout(
                title="Cumulative squared-error advantage of the model over RW (bp²)",
                height=330,
                yaxis_title="bp²",
            )
            st.plotly_chart(fig)
        with st.expander("Full table"):
            st.dataframe(
                s.style.format(
                    {
                        "rmse_model_bp": "{:.1f}",
                        "rmse_rw_bp": "{:.1f}",
                        "ratio": "{:.3f}",
                        "dm_stat": "{:.2f}",
                    }
                )
            )
            csv_download(s.set_index(["horizon_m", "maturity"]), "oos_forecast_summary.csv")
        refits = oos.get("refits")
        if refits is not None and len(refits):
            with st.expander("Parameter stability across refits (κᵠ)"):
                r = refits.set_index("refit_date")[["kappaQ_1", "kappaQ_2", "kappaQ_3"]]
                fig = go.Figure([go.Scatter(x=r.index, y=r[c], name=c) for c in r.columns])
                fig.update_layout(height=300, yaxis_type="log", title="κᵠ estimates by refit date")
                st.plotly_chart(fig)
    explain(
        "Forecasts use the exact discrete P dynamics: $E_t[x_{t+h}]=\\theta^P+F^h(x_{t|t}-\\theta^P)$ and "
        "$E_t[y_{t+h}]=a^Q+b^{Q\\prime}E_t[x_{t+h}]$ (`forecast.py`). Out of sample, every forecast uses "
        "only parameters estimated on data up to the latest refit and the filtered (not smoothed) state. "
        "Beating the random walk at 6–12 months is hard for affine models on real data (Duffee 2002); "
        "this tab tells you honestly how the P dynamics perform, which matters because the term premium "
        "is built on them. Yield data are final vintages, so the exercise is pseudo out-of-sample."
    )
