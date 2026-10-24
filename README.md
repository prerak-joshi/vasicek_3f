# vasicek3f — 3-Factor Vasicek Term Structure Model

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vasicek3f-5ygmrt2fr8rvg97zfofzlq.streamlit.app)
[![tests](https://github.com/prerak-joshi/vasicek_3f/actions/workflows/tests.yml/badge.svg)](https://github.com/prerak-joshi/vasicek_3f/actions/workflows/tests.yml)
[![docs](https://github.com/prerak-joshi/vasicek_3f/actions/workflows/docs.yml/badge.svg)](https://prerak-joshi.github.io/vasicek_3f/)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Ruff](https://img.shields.io/badge/code%20style-ruff-261230)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A three-factor Gaussian Vasicek (A0(3)) affine term structure model of government bond yields, estimated by **exact Kalman-filter maximum likelihood**, with a **model-based term premium decomposition**, **bias-corrected P dynamics**, **parameter-uncertainty bands** and an **out-of-sample forecast test**. It comes with an interactive dashboard, documentation site and walkthrough notebook.

**[Live dashboard](https://vasicek3f-5ygmrt2fr8rvg97zfofzlq.streamlit.app)** · **[Documentation](https://prerak-joshi.github.io/vasicek_3f/)** · **[Wiki](https://github.com/prerak-joshi/vasicek_3f/wiki)**

```
Term Premium(τ) = Pricing-Measure Fitted Yield(τ) − Physical-Measure Affine Yield(τ)
                = [aQ(τ) + bQ(τ)'x_t]            − [aP(τ) + bP(τ)'x_t]
```

The physical-measure affine yield is a no-risk-premium benchmark inside the model, not a strict Expectations-Hypothesis estimate; it differs from the average expected short rate by a small P-convexity term, which is reported.

## Highlights

| | |
|---|---|
| **Model** | Canonical Gaussian A0(3): diagonal ordered K^Q, θ^Q = 0, full K^P; closed-form Q loadings, ODE-based P loadings |
| **Estimation** | Exact discretisation (Van Loan), exact Kalman likelihood (collapsed Jungbacker-Koopman filter + steady state), two-stage starting values, MLE, delta-method standard errors |
| **Term premium** | MLE and small-sample bias-corrected (Kilian bootstrap) P dynamics; 90% parameter-uncertainty bands; yield-change attribution |
| **Validation** | Expanding-window out-of-sample forecasts vs random walk with Diebold-Mariano tests; simulation recovery; 28 tests, ~90% coverage |
| **Benchmarks** | ACM (NY Fed) and Kim-Wright (Fed) term premia |
| **Delivery** | Streamlit dashboard, MkDocs site with API reference, GitHub wiki, notebook, CLI, portable results bundles |

## Quick start

```bash
git clone https://github.com/prerak-joshi/vasicek_3f.git && cd vasicek_3f
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .

pytest                                                 # 28 tests
streamlit run streamlit_app.py                         # dashboard

# simulated data (known truth)
vasicek3f run --source simulated --oos --out results/simulated

# US Treasuries: GSW yields + ACM and Kim-Wright benchmarks
vasicek3f download --dest data                         # or save the three files manually
vasicek3f run --source gsw --gsw data/feds200628.csv \
    --acm data/ACMTermPremium.xls --kw data/THREEFYTP10.csv --start 1990-01 --oos --out results/gsw
```

Details, data sources and all options: **[Getting started](https://prerak-joshi.github.io/vasicek_3f/getting-started/)**.

## Repository

```
src/vasicek3f/     model, estimation, decomposition, bias correction, bands, forecasting, data, pipeline
dashboard/         Streamlit app (app.py, theme, context, one module per tab in views/)
streamlit_app.py   Streamlit Community Cloud entry point
notebooks/         guided walkthrough
tests/             unit, recovery, pipeline and loader tests
docs/              documentation site (MkDocs) — also generates the wiki via scripts/build_wiki.py
results/           published runs (portable bundle, CSVs, figures, summary.md)
```

How the pieces connect: **[Architecture](https://prerak-joshi.github.io/vasicek_3f/architecture/)**.

## Interpreting the output

Fitted yields and *changes* in the term premium are well determined. The *level* of the term premium depends on the weakly identified physical dynamics. Compare the MLE and bias-corrected estimates and the band before reading much into a level. See **[Interpreting the term premium](https://prerak-joshi.github.io/vasicek_3f/term-premium/)**.

## Deployment

Streamlit Community Cloud: main file `streamlit_app.py`, Python 3.12, no secrets required. Runtime dependencies are in `requirements.txt`, theme in `.streamlit/config.toml`. The app opens the published runs in `results/`.

### Secrets management

This app does not currently require credentials or other secrets. For a private GitHub
repository, authorize Streamlit Community Cloud to access the repository through its
GitHub integration; do not put a GitHub token in the app's secrets. If a future feature
needs a credential, add it in the app's **Settings → Secrets** in Community Cloud and
read it with `st.secrets` in the app. For local development, use
`.streamlit/secrets.toml`; that file is excluded from Git, so never force-add it or paste
secret values into source code, README files, or logs.

## Contributing, citation, licence

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md). Cite via [CITATION.cff](CITATION.cff). MIT licence.

> **Scope.** Educational and research project. Not an official or production term premium model, not a Kim-Wright implementation, and not investment advice.
