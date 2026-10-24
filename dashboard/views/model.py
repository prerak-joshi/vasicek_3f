"""Model & estimation: parameters, standard errors, P dynamics, diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

from context import Ctx, csv_download, explain
from vasicek3f.params import ParamTransform


def render(ctx: Ctx) -> None:
    est, p = ctx.est, ctx.params
    c = st.columns(5)
    c[0].metric("Log-likelihood", f"{est['loglik']:,.1f}")
    c[1].metric("AIC / BIC (thousands)", f"{est['aic'] / 1e3:,.1f} / {est['bic'] / 1e3:,.1f}")
    c[2].metric("Parameters / obs", f"{est['n_params']} / {est['n_obs']}")
    c[3].metric("Optimiser iterations", f"{est['n_iter']}")
    hp = est.get("hessian_pd")
    c[4].metric("Hessian positive definite", "—" if hp is None else ("yes" if hp else "no"))
    st.caption(f"Optimiser: {est['message']}")

    tr = ParamTransform(len(ctx.tau), ctx.cfg.get("common_measurement_error", False))
    tab = pd.DataFrame(
        {"start (2-stage)": tr.natural_vector(ctx.bundle["start_params"]), "MLE": tr.natural_vector(p)},
        index=tr.natural_names,
    )
    se = ctx.bundle.get("std_errors")
    if se is not None:
        tab["std error"] = se["std_error"].values
        tab["t-stat"] = se["t_stat"].values
    else:
        st.info("No standard errors in this run (fast mode or --no-se).")
    if ctx.is_sim and ctx.bundle.get("true_params") is not None:
        tab.insert(0, "true", tr.natural_vector(ctx.bundle["true_params"]))
    st.dataframe(tab.style.format("{:.5f}"), height=430)
    csv_download(tab, "parameters.csv", "Download parameter table (CSV)")

    st.subheader("Physical dynamics: MLE vs bias-corrected")
    hl = p.factor_half_lives()
    rows = {"MLE": np.sort(hl["P"])[::-1]}
    bc = ctx.bundle.get("bias_correction")
    if bc is not None:
        rows["Bias-corrected"] = np.sort(np.asarray(bc["half_life_years"]))[::-1]
    rows["Q (for reference)"] = np.sort(hl["Q"])[::-1]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Half-lives of the factors (years), slowest first**")
        st.dataframe(pd.DataFrame(rows, index=["slowest", "middle", "fastest"]).T.style.format("{:.2f}"))
        lr = p.short_rate_long_run()
        txt = f"Long-run short rate: Q **{lr['Q'] * 100:.2f}%**, P (MLE) **{lr['P'] * 100:.2f}%**"
        if bc is not None:
            txt += f", P (bias-corrected) **{(p.delta0 + np.sum(bc['theta_P'])) * 100:.2f}%**"
        st.markdown(txt)
        if bc is not None:
            st.caption(
                f"Bias correction: largest VAR root {bc['max_root_ols']:.4f} → {bc['max_root_bc']:.4f}, "
                f"{bc['shrink']:.0%} of the bootstrap bias removed, {bc['n_boot']} draws; "
                f"log-likelihood at corrected parameters {bc['loglik'] - est['loglik']:+.1f} vs MLE."
            )
    with c2:
        mpr = p.market_price_of_risk()
        st.markdown("**Market prices of risk**  λₜ = λ₀ + λ₁ xₜ")
        st.dataframe(
            pd.DataFrame(
                np.column_stack([mpr["lambda0"], mpr["lambda1"]]),
                columns=["λ₀", "λ₁·x1", "λ₁·x2", "λ₁·x3"],
                index=["shock 1", "shock 2", "shock 3"],
            ).style.format("{:.3f}")
        )
    st.markdown(
        "**Standardised one-step-ahead prediction errors** (≈ mean 0, s.d. 1, low AR(1) if well specified)"
    )
    st.dataframe(ctx.bundle["innovations"].style.format("{:.3f}"))
    st.download_button(
        "Download parameters (JSON)",
        json.dumps(p.to_dict(), indent=2),
        file_name="params.json",
        mime="application/json",
    )
    st.caption(
        f"Run created {ctx.bundle.get('created', 'n/a')} with vasicek3f "
        f"{ctx.bundle.get('package_version', '1.0')}; runtime {ctx.bundle['runtime_sec']:.0f}s."
    )
    explain(
        "Three-stage estimation (`estimation.py`): (1) cross-sectional least squares for κᵠ and δ₀ with "
        "per-date OLS states; (2) VAR(1) on those states mapped to Kᴾ, θᴾ, Σ (matrix log, exact Van Loan "
        "inversion); (3) joint MLE of all parameters with the exact Kalman-filter likelihood (collapsed "
        "Jungbacker-Koopman filter, steady-state gain). Standard errors: inverse numerical Hessian in the "
        "unconstrained space, delta method to natural parameters. Large Kᴾ standard errors relative to "
        "κᵠ are the quantitative form of 'P is weakly identified'."
    )
