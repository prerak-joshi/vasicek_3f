"""Term premium decomposition: components, cross-section, factor loadings, benchmarks."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from context import Ctx, csv_download, explain, pct
from theme import COL, FACTOR, ts_figure


def render(ctx: Ctx) -> None:
    y, dec, m = ctx.yields, ctx.dec, ctx.tp_m
    st.markdown(
        "**Term Premium = Pricing-Measure Fitted Yield − Physical-Measure Affine Yield**"
        f" &nbsp;·&nbsp; showing **{'bias-corrected' if ctx.use_bc else 'MLE'} P dynamics**"
    )
    show_eh = st.toggle("Overlay strict EH average expected short rate", value=False)
    ser = {
        "Observed": pct(y[m]),
        "Q fitted yield": pct(dec["fitted_Q"][m]),
        "P affine yield": pct(dec["affine_P"][m]),
    }
    sty = {
        "Observed": {"color": COL["Observed"], "width": 1.0},
        "Q fitted yield": {"color": COL["Q"]},
        "P affine yield": {"color": COL["P"]},
    }
    if show_eh:
        ser["EH avg short rate"] = pct(dec["eh_short_rate"][m])
        sty["EH avg short rate"] = {"color": COL["EH"], "dash": "dot"}
    st.plotly_chart(ts_figure(ser, f"{m:g}-year yield and its components", styles=sty, recessions=ctx.is_us))

    ser = {"TP (MLE)": pct(ctx.dec_mle["term_premium"][m])}
    sty = {"TP (MLE)": {"color": COL["TP"], "width": 2.0}}
    if ctx.dec_bc is not None:
        ser["TP (bias-corrected P)"] = pct(ctx.dec_bc["term_premium"][m])
        sty["TP (bias-corrected P)"] = {"color": COL["TP_BC"], "width": 1.6}
    band = None
    if ctx.bands is not None:
        band = (pct(ctx.bands["q05"][m]), pct(ctx.bands["q95"][m]), "90% parameter band (MLE)")
    st.plotly_chart(
        ts_figure(
            ser,
            f"{m:g}-year term premium: MLE vs bias-corrected P",
            styles=sty,
            band=band,
            recessions=ctx.is_us,
            zero_line=True,
        )
    )

    c1, c2 = st.columns(2)
    with c1:
        d = st.select_slider(
            "Date",
            options=list(y.index),
            value=y.index[-1],
            format_func=lambda v: v.strftime("%Y-%m"),
            key="tp_date",
        )
        fig = go.Figure()
        for k, nm, col in (
            ("fitted_Q", "Q fitted", COL["Q"]),
            ("affine_P", "P affine", COL["P"]),
            ("term_premium", "Term premium", COL["TP"]),
        ):
            fig.add_scatter(x=ctx.tau, y=pct(dec[k].loc[d]).values, name=nm, line=dict(color=col))
        if ctx.bands is not None and not ctx.use_bc:
            fig.add_scatter(
                x=ctx.tau,
                y=pct(ctx.bands["q95"].loc[d]).values,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_scatter(
                x=ctx.tau,
                y=pct(ctx.bands["q05"].loc[d]).values,
                fill="tonexty",
                fillcolor=COL["Band"],
                line=dict(width=0),
                name="90% band",
            )
        fig.update_layout(
            title=f"Decomposition across maturities, {d:%b %Y}",
            height=370,
            xaxis_title="maturity (years)",
            yaxis_title="%",
            hovermode="x",
        )
        st.plotly_chart(fig)
    with c2:
        lo = ctx.bundle["loadings_obs"]
        fig = go.Figure()
        for i in range(3):
            fig.add_bar(
                x=[f"{t:g}y" for t in lo.index],
                y=lo[f"b{i + 1}_TP"],
                name=f"x{i + 1}",
                marker_color=FACTOR[i],
            )
        fig.update_layout(
            barmode="group", title="Term premium loadings bᵠ(τ) − bᴾ(τ) (MLE)", height=370, hovermode="x"
        )
        st.plotly_chart(fig)

    conv = dec["convexity_P"][m] * 1e4
    st.caption(
        f"P-measure convexity at {m:g}y (EH average short rate − P affine yield): mean "
        f"{conv.mean():.1f} bp — the only difference between this benchmark and a strict EH "
        "expectations component."
    )

    # --- benchmarks
    st.subheader("Comparison with published term premium estimates")
    bm = ctx.benchmarks
    if not bm:
        st.info(
            "No benchmark series in this run. Pass the NY Fed ACM workbook and FRED THREEFYTP10 "
            "(Kim-Wright) when running on GSW data."
        )
    else:
        tm = ctx.bundle["tp_maturity"]
        mt = ctx.dec_mle["term_premium"][tm]
        ser = {"TP (MLE)": pct(mt)}
        sty = {"TP (MLE)": {"color": COL["TP"], "width": 2.0}}
        if ctx.dec_bc is not None:
            ser["TP (bias-corrected P)"] = pct(ctx.dec_bc["term_premium"][tm])
            sty["TP (bias-corrected P)"] = {"color": COL["TP_BC"], "width": 1.5}
        for name, s in bm.items():
            ser[name] = pct(s.loc[mt.index.min() : mt.index.max()])
            sty[name] = {
                "color": COL.get(name.split()[0], COL["True"]),
                "width": 1.4,
                "dash": "dash" if name.startswith("True") else None,
            }
        st.plotly_chart(
            ts_figure(
                ser,
                f"{tm:g}-year term premium vs benchmarks",
                styles=sty,
                recessions=ctx.is_us,
                zero_line=True,
                height=420,
            )
        )
        tab = ctx.bundle.get("benchmark_table")
        if tab is not None and len(tab):
            st.dataframe(tab)
        if ctx.dec_bc is not None:
            from vasicek3f.diagnostics import benchmark_comparison

            st.caption("Same statistics for the bias-corrected term premium:")
            st.dataframe(benchmark_comparison(ctx.dec_bc["term_premium"][tm], bm))
    explain(
        "Both components are affine in the same smoothed state, so $TP_t=(a^Q-a^P)+(b^Q-b^P)'x_t$ "
        "(`decomposition.term_premium_decomposition`). **Bias-corrected P** re-estimates $K^P,\\theta^P$ "
        "from the filtered states with a Kilian (1998) bootstrap correction for the downward bias in "
        "estimated persistence (Bauer, Rudebusch & Wu 2012), keeping all Q parameters at the MLE "
        "(`bias_correction.py`). The **band** simulates parameters from the MLE's asymptotic "
        "distribution and re-runs the smoother for each draw (`uncertainty.py`). ACM is regression-based; "
        "Kim-Wright adds survey forecasts to pin down P. Agreement in *changes* is the fair test; "
        "*levels* differ by construction."
    )
    out = pd.DataFrame({"tp_mle": ctx.dec_mle["term_premium"][m]})
    if ctx.dec_bc is not None:
        out["tp_bias_corrected"] = ctx.dec_bc["term_premium"][m]
    if ctx.bands is not None:
        out["band_q05"], out["band_q95"] = ctx.bands["q05"][m], ctx.bands["q95"][m]
    csv_download(out * 100, f"term_premium_{m:g}y_pct.csv", f"Download {m:g}y term premium series (CSV, %)")
