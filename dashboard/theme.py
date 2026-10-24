"""Visual identity for the dashboard: palette, Plotly template, CSS and chart helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- palette
INK = "#1b2430"
MUTED = "#5f6b7a"
GRID = "#e6e9ef"
COL = {
    "Observed": "#2b2b2b",
    "Q": "#1f5fa8",  # pricing measure
    "P": "#c4622d",  # physical measure
    "TP": "#2e8b57",  # term premium (MLE)
    "TP_BC": "#7a3e9d",  # term premium (bias-corrected P)
    "EH": "#8a6d3b",
    "ACM": "#d4a017",
    "KW": "#5a9bd5",
    "True": "#8c8c8c",
    "Shock": "#d62728",
    "Band": "rgba(46,139,87,0.16)",
}
FACTOR = ["#1f5fa8", "#c4622d", "#2e8b57"]

# NBER US business-cycle recessions (peak month to trough month)
NBER_RECESSIONS = [
    ("1969-12", "1970-11"),
    ("1973-11", "1975-03"),
    ("1980-01", "1980-07"),
    ("1981-07", "1982-11"),
    ("1990-07", "1991-03"),
    ("2001-03", "2001-11"),
    ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

pio.templates["vasicek"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif", size=12.5, color=INK
        ),
        title=dict(font=dict(size=14.5, color=INK), x=0.0, xanchor="left", y=0.97),
        colorway=[COL["Q"], COL["P"], COL["TP"], COL["TP_BC"], COL["ACM"], COL["KW"]],
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(gridcolor=GRID, linecolor="#c9ced6", ticks="outside", tickcolor="#c9ced6", zeroline=False),
        yaxis=dict(gridcolor=GRID, linecolor="#c9ced6", zeroline=False),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#c9ced6", font=dict(size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11.5),
        ),
        margin=dict(l=10, r=10, t=56, b=10),
    )
)
pio.templates.default = "vasicek"

CSS = """
<style>
.block-container {padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em; color: #1b2430;}
div[data-testid="stMarkdownContainer"] h3, h3 {font-size: 1.18rem !important; font-weight: 620 !important;
    padding-top: 0.9rem !important;}
div[data-testid="stMetric"] {
    background: #f7f9fc; border: 1px solid #e3e8ef; border-radius: 10px;
    padding: 0.7rem 0.9rem;
}
div[data-testid="stMetricLabel"] p {font-size: 0.80rem; color: #5f6b7a;}
div[data-testid="stMetricValue"] {font-size: 1.45rem;}
.vt-header {border-bottom: 1px solid #e3e8ef; padding-bottom: 0.6rem; margin-bottom: 0.8rem;}
.vt-title {font-size: 1.7rem !important; font-weight: 680; color: #1b2430; margin: 0;
           line-height: 1.25;}
.vt-sub {color: #5f6b7a; font-size: 0.95rem !important; margin: 0.2rem 0 0.45rem 0;}
.vt-chip {display: inline-block; background: #eef3fb; color: #1f5fa8; border-radius: 999px;
          padding: 0.08rem 0.6rem; font-size: 0.78rem; margin-right: 0.35rem; font-weight: 550;}
.vt-chip.warn {background: #fff4e5; color: #9a5b00;}
.vt-note {color: #5f6b7a; font-size: 0.86rem;}
.vt-take {background: #f7f9fc; border-left: 3px solid #1f5fa8; padding: 0.7rem 1rem;
          border-radius: 6px; margin: 0.3rem 0 0.8rem 0;}
.vt-take li {margin-bottom: 0.25rem;}
button[data-baseweb="tab"] p {font-size: 0.93rem; font-weight: 550;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str, chips: Iterable[tuple[str, bool]] = ()) -> None:
    chip_html = "".join(f'<span class="vt-chip{" warn" if w else ""}">{c}</span>' for c, w in chips)
    st.markdown(
        f'<div class="vt-header"><div class="vt-title">{title}</div>'
        f'<div class="vt-sub">{subtitle}</div><div>{chip_html}</div></div>',
        unsafe_allow_html=True,
    )


def takeaways(lines: Iterable[str]) -> None:
    items = "".join(f"<li>{ln}</li>" for ln in lines)
    st.markdown(
        f'<div class="vt-take"><ul style="margin:0;padding-left:1.1rem">{items}</ul></div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- chart helpers
RANGE_BUTTONS = dict(
    buttons=[
        dict(count=1, label="1Y", step="year", stepmode="backward"),
        dict(count=5, label="5Y", step="year", stepmode="backward"),
        dict(count=10, label="10Y", step="year", stepmode="backward"),
        dict(step="all", label="All"),
    ],
    bgcolor="#f1f4f9",
    activecolor="#dbe6f7",
    font=dict(size=11),
    x=1,
    xanchor="right",
    y=1.12,
)


def add_recessions(
    fig: go.Figure, index: pd.Index, row: Optional[int] = None, col: Optional[int] = None
) -> None:
    lo, hi = index.min(), index.max()
    for a, b in NBER_RECESSIONS:
        a_, b_ = pd.Timestamp(a), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
        if b_ < lo or a_ > hi:
            continue
        kw = dict(
            x0=max(a_, lo), x1=min(b_, hi), fillcolor="#9aa4b2", opacity=0.13, line_width=0, layer="below"
        )
        if row is not None:
            kw.update(row=row, col=col)
        fig.add_vrect(**kw)


def ts_figure(
    series: dict[str, pd.Series],
    title: str,
    ytitle: str = "%",
    height: int = 380,
    styles: Optional[dict[str, dict]] = None,
    band: Optional[tuple[pd.Series, pd.Series, str]] = None,
    recessions: bool = False,
    zero_line: bool = False,
    range_selector: bool = True,
) -> go.Figure:
    """Time-series chart with optional shaded band, recession shading and range selector."""
    fig = go.Figure()
    if band is not None:
        lo, hi, name = band
        fig.add_trace(
            go.Scatter(x=hi.index, y=hi.values, line=dict(width=0), showlegend=False, hoverinfo="skip")
        )
        fig.add_trace(
            go.Scatter(
                x=lo.index,
                y=lo.values,
                fill="tonexty",
                line=dict(width=0),
                fillcolor=COL["Band"],
                name=name,
                hoverinfo="skip",
            )
        )
    for name, s in series.items():
        st_ = (styles or {}).get(name, {})
        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                name=name,
                mode="lines",
                line=dict(width=st_.get("width", 1.7), color=st_.get("color"), dash=st_.get("dash")),
                hovertemplate="%{y:.2f}",
            )
        )
    if recessions and len(series):
        add_recessions(fig, next(iter(series.values())).index)
    if zero_line:
        fig.add_hline(y=0, line=dict(color="#7d8592", width=0.9))
    fig.update_layout(title=title, height=height, yaxis_title=ytitle)
    if range_selector:
        fig.update_xaxes(rangeselector=RANGE_BUTTONS)
    return fig


def curve_figure(
    tau, curves: dict[str, tuple[pd.Series, dict]], title: str, height: int = 360, ytitle: str = "%"
) -> go.Figure:
    fig = go.Figure()
    for name, (vals, style) in curves.items():
        mode = style.pop("mode", "lines")
        fig.add_trace(go.Scatter(x=list(tau), y=list(vals), name=name, mode=mode, **style))
    fig.update_layout(
        title=title, height=height, xaxis_title="maturity (years)", yaxis_title=ytitle, hovermode="x"
    )
    return fig
