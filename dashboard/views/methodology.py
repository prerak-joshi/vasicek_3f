"""Methodology: renders docs/methodology.md plus scope and caveats."""

from __future__ import annotations

import os

import streamlit as st

from context import ROOT, Ctx

CAVEATS = """
### Scope and caveats

- **Model-based term premium.** The physical-measure affine yield is a no-risk-premium benchmark inside the
  model, not an observed expectation, and differs from a strict Expectations-Hypothesis component by
  P-measure convexity.
- **Weak P identification.** The level of the term premium depends on Kᴾ and θᴾ, which a few decades of
  persistent monthly data pin down poorly. Compare the MLE and bias-corrected estimates and the
  parameter band before reading much into a level.
- **Gaussian dynamics.** Constant volatility; negative yields are possible; the effective lower bound
  period is not modelled.
- **No survey or macro information**, unlike Kim-Wright. Differences from ACM and KW are informative,
  not errors to be tuned away.
- **Educational, not official.** Not a production term premium model and not investment advice.
"""


def render(ctx: Ctx) -> None:
    st.markdown(CAVEATS)
    path = os.path.join(ROOT, "docs", "methodology.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
            st.markdown(text.split("\n", 1)[1] if text.startswith("# ") else text)
    st.markdown(
        "---\nFull documentation, API reference and the walkthrough notebook are in the "
        "[GitHub repository](https://github.com/prerak-joshi/vasicek_3f)."
    )
