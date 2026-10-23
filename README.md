# 3-Factor Vasicek Term Structure Model

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR-APP-NAME.streamlit.app)
[![tests](https://github.com/YOUR-GITHUB-USERNAME/vasicek3f/actions/workflows/tests.yml/badge.svg)](https://github.com/YOUR-GITHUB-USERNAME/vasicek3f/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

A complete Python implementation of a **three-factor Gaussian Vasicek (A0(3)) affine term structure model** for government bond yields:

- latent factors estimated with the **Kalman filter**
- parameters fitted by **maximum likelihood**
- model-implied yields across the curve
- **term premium decomposition** into a pricing-measure fitted yield and a physical-measure affine yield
- comparison with the **ACM** and **Kim-Wright** term premium series

It includes a Python package, a command-line pipeline, an interactive **Streamlit dashboard**, a guided **Jupyter notebook**, unit tests with a parameter-recovery experiment, and documentation.

> **Educational project.** This is a learning and interview-preparation project, not a production or official term premium model. It is not a Kim-Wright implementation and has no survey or macro information. See [Where to be careful](#where-the-model-is-strong-and-where-to-be-careful).

---

## The decomposition used here

```
Term Premium(τ) = Pricing-Measure Fitted Yield(τ) − Physical-Measure Affine Yield(τ)
                = [aQ(τ) + bQ(τ)'x_t]            − [aP(τ) + bP(τ)'x_t]
```

The physical-measure affine yield is the yield the model would produce if bonds were priced with the P dynamics, i.e. with a zero market price of risk. It is a tractable **no-risk-premium benchmark inside the model, not a strict Expectations-Hypothesis estimate**. It equals the average expected short rate minus a small P-measure convexity term. The strict EH series is also computed and reported as a reference, so the size of that gap is visible.

---

## Quick start

```bash
git clone https://github.com/YOUR-GITHUB-USERNAME/vasicek3f.git && cd vasicek3f
pip install -r requirements-dev.txt      # runtime + tests + notebook (requirements.txt = runtime only)

pytest -q                                # 20 tests, ~15s

# 1) simulated data (no downloads; known true parameters)
python -m vasicek3f run --source simulated --out results/simulated

# 2) US Treasuries (GSW) with ACM and Kim-Wright benchmarks
python -m vasicek3f download --dest data         # or download the three files manually
python -m vasicek3f run --source gsw --gsw data/feds200628.csv \
    --acm data/ACMTermPremium.xls --kw data/THREEFYTP10.csv \
    --start 1990-01 --out results/gsw

# 3) dashboard and notebook
streamlit run streamlit_app.py
jupyter lab notebooks/01_vasicek3f_walkthrough.ipynb
```

If the automatic download fails (some sites block scripted requests), save these files manually into `data/`:

| Series | Source | File |
|---|---|---|
| GSW zero-coupon yields | federalreserve.gov/data/yield-curve-tables/feds200628.csv | `feds200628.csv` |
| ACM term premium | newyorkfed.org → ACM Term Premia → `ACMTermPremium.xls` | `ACMTermPremium.xls` |
| Kim-Wright 10y term premium | FRED series `THREEFYTP10` (download CSV) | `THREEFYTP10.csv` |

Useful flags for `run`: `--fast` (no standard errors, looser tolerance), `--no-se`, `--restarts N`, `--maturities 1 2 3 5 7 10`, `--tp-maturity 10`, `--common-error`, `--end 2026-09`.

---

## Deploy the dashboard on Streamlit Community Cloud

The repository is laid out for Streamlit Community Cloud out of the box:

| File | Role on Streamlit Cloud |
|---|---|
| `streamlit_app.py` | default entry point Streamlit looks for (runs `dashboard/app.py`) |
| `requirements.txt` | lean runtime dependencies Streamlit installs (no Jupyter / pytest) |
| `.streamlit/config.toml` | theme and server settings (upload limit for GSW files) |
| `results/simulated/bundle.pkl` | pre-computed demo so the app opens instantly |

Steps:

1. Push this repository to GitHub (public, or private if your Streamlit account is linked to it).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → **Deploy a public app from GitHub**.
3. Choose the repository, branch `main`, main file path `streamlit_app.py`.
4. Under **Advanced settings**, pick Python **3.12** (3.11 also works). No secrets are needed.
5. Deploy. Replace the `YOUR-APP-NAME` / `YOUR-GITHUB-USERNAME` placeholders in the badges above with your app URL and username.

How the cloud version behaves:

- **Opens on the saved simulated run.** If the pickled demo can't be read with the library versions Streamlit installs, the app automatically runs a fresh fast-mode demo estimation instead (about 10–20 s, once, then cached for all visitors).
- **Real data via upload.** Streamlit Cloud can't see your local `data/` folder, so the *Estimate: GSW data* mode accepts uploads of `feds200628.csv`, the ACM workbook and the FRED Kim-Wright CSV. Uploaded files and dashboard runs are written to a temporary folder, never to the repo.
- **Keep fast mode on** for estimation in the cloud. Full mode adds the numerical-Hessian standard errors and is noticeably slower on the shared free-tier CPU.
- To show real-data results without uploading each time, run the GSW pipeline locally (`--out results/gsw`) and commit `results/gsw/` (remove the `results/*/` line from `.gitignore` or force-add it). It then appears under *Load saved results*. Check the data providers' terms before redistributing their series.

GitHub Actions (`.github/workflows/tests.yml`) runs the test suite and a headless dashboard smoke test on every push, so a broken deploy is caught before Streamlit picks it up.

---

## Repository structure

```
vasicek3f/
├── README.md                     ← this file: overview, structure, how files connect
├── streamlit_app.py              ← Streamlit Cloud entry point (runs dashboard/app.py)
├── requirements.txt              runtime deps (what Streamlit Cloud installs)
├── requirements-dev.txt          + pytest, Jupyter
├── pyproject.toml, LICENSE
├── .streamlit/config.toml        dashboard theme and server settings
├── .github/workflows/tests.yml   CI: tests + dashboard smoke test
├── src/vasicek3f/                ← the model (all logic lives here)
│   ├── params.py                 identification scheme, parameter container, unconstrained transforms
│   ├── affine.py                 bond pricing: closed-form Q loadings, ODE solver for P, EH reference
│   ├── discretize.py             exact OU discretisation (Van Loan) and its inverse
│   ├── kalman.py                 Kalman filter (missing data), collapsed fast likelihood, RTS smoother
│   ├── model.py                  builds the state-space model from parameters; log-likelihood
│   ├── estimation.py             2-stage starting values, MLE, numerical-Hessian standard errors
│   ├── decomposition.py          term premium decomposition and loadings tables
│   ├── diagnostics.py            pricing errors, PCA, factor correlations, benchmark comparison
│   ├── data.py                   GSW / ACM / Kim-Wright loaders and downloader
│   ├── simulate.py               realistic true parameters and panel simulator
│   ├── plotting.py               static matplotlib figures for reports and the notebook
│   ├── pipeline.py               end-to-end run, results bundle, summary report
│   └── cli.py, __main__.py       `python -m vasicek3f download | run`
├── dashboard/app.py              ← Streamlit dashboard (presentation + sensitivity only)
├── notebooks/
│   ├── build_notebook.py         reproducible source of the notebook
│   └── 01_vasicek3f_walkthrough.ipynb   executed walkthrough, theory next to code
├── tests/
│   ├── test_model.py             pricing, discretisation, filter vs brute force, identities
│   ├── test_recovery.py          simulate → estimate → recover parameters and term premium
│   └── test_data.py              loaders on files mimicking the real layouts
├── docs/
│   ├── THEORY.md                 the mathematics, derivations and identification
│   └── DASHBOARD_GUIDE.md        tabs, chart-to-model mapping, sliders, runtime
├── results/simulated/            example output (bundle.pkl, CSVs, figures, summary.md)
└── data/                         put downloaded data files here
```

### How the files connect

```
             data.py / simulate.py
                     │  yields (T×N, decimals)
                     ▼
 params.py ──► affine.py (Q loadings a, H) ─┐
     │                                      ├─► model.py: state-space model ─► kalman.py: log L
     └────► discretize.py (c, F, Q, P0) ────┘                                     │
                                                                                  ▼
                     estimation.py: starting values ─► MLE ─► standard errors
                                                                                  │ params
                                                                                  ▼
            model.filter_and_smooth ─► smoothed states ─► decomposition.py (affine.py again)
                                                                                  │
                                                     diagnostics.py, plotting.py  ▼
                                    pipeline.py ─► results/<run>/ (bundle.pkl, CSV, PNG, summary.md)
                                                         │                │
                                              dashboard/app.py    notebook (calls the package directly)
```

- **One source of truth.** All model logic is in `src/vasicek3f`. The CLI, notebook, tests and dashboard import it; none of them re-implements pricing or estimation.
- **`pipeline.py` is the hand-off point.** It runs everything and writes a results bundle: `bundle.pkl` for the dashboard, plus human-readable CSV/JSON/PNG and `summary.md`.
- **The dashboard reads bundles** (instant) or calls `pipeline.run` itself (sidebar "Estimate" modes). Its only extra logic is the sensitivity shocks, which reuse the same pricing and filtering functions.
- **The notebook** walks through the same steps one module at a time, so it doubles as documentation of the code path.

---

## The dashboard

Full guide: [docs/DASHBOARD_GUIDE.md](docs/DASHBOARD_GUIDE.md). In short:

**Audience.** Quant finance and MFE students, aspiring quant researchers, and practitioners preparing for interviews. It assumes the prerequisites below and aims to show how each equation becomes a number.

**Tabs.**

| Tab | Shows | Built from |
|---|---|---|
| Overview | key metrics, model equations, PCA (why 3 factors), yield heatmap | `diagnostics.pca_explained`, estimation summary |
| Yield curve fit | observed vs fitted by maturity and by date, pricing-error heatmap and table | `affine.q_loadings` × smoothed states; measurement errors |
| Latent factors & loadings | smoothed states with ±2 s.d. bands, Q vs P loadings, intercepts, factor correlations | `kalman.rts_smoother`, `affine.q_loadings/p_loadings` |
| Term premium decomposition | Q fitted yield, P affine yield, TP; TP across maturities; TP factor loadings | `decomposition.term_premium_decomposition` |
| Benchmarks | model TP vs ACM / KW (or the true TP in simulation) with statistics | `diagnostics.benchmark_comparison` |
| Sensitivity (sliders) | shock parameters, see TP, fit and likelihood change | same pricing/filter functions at shocked parameters |
| Estimation & diagnostics | start vs MLE (vs truth), standard errors, half-lives, market prices of risk, innovation tests | `estimation.py`, `params.py` |

Every chart has a "How this chart connects to the model" note naming the equations and functions behind it.

**Sliders are sensitivity analysis, not re-estimation.** They multiply or shift blocks of the *estimated* parameters (κQ, δ0, K_P, the P long-run short rate, Σ) and recompute the outputs. Nothing is re-optimised. The dashboard reports the log-likelihood change so you can see how much worse the shocked parameters fit. Proper re-estimation would let every other parameter adjust and takes seconds to minutes; the sliders respond in milliseconds. You can either hold the states fixed (instant) or re-run the Kalman filter at the shocked parameters (slower, still no optimisation).

**Runtime and fast mode.**

| Step | Typical time (simulated, 480 months × 10 maturities) |
|---|---|
| Load a saved bundle | ~1–2 s |
| Estimation, fast mode (no SEs, ftol 1e-10, no polish) | ~10 s |
| Estimation, full mode (with numerical-Hessian SEs) | ~25 s |
| Sensitivity, states held fixed / re-filtered | ~15 ms / ~50–100 ms |

Real data usually takes longer than simulation because the optimiser has a harder surface. The slow parts are the MLE (each gradient needs ~33 likelihood evaluations by finite differences) and the standard errors (a 32×32 numerical Hessian is ~2,000 likelihood evaluations). P-measure loadings require solving ODEs numerically, which is why loading charts use a coarser grid in fast mode. Estimation results are cached, so re-opening tabs or moving unrelated widgets does not re-run them.

---

## What the project covers

- building a 3-factor Gaussian Vasicek model in canonical form
- affine bond prices and yields, closed-form loadings and convexity
- separating physical-measure (P) and pricing-measure (Q) dynamics; market prices of risk
- exact discretisation and state-space form
- Kalman filtering, smoothing and the prediction-error likelihood
- maximum likelihood with an unconstrained parameterisation and good starting values
- fitted yields, pricing-error diagnostics and factor interpretation
- the model-based term premium and comparison with ACM and KW

Theory and derivations: [docs/THEORY.md](docs/THEORY.md).

## Validation

- Closed-form Q loadings match a numerical ODE solution to ~1e-12.
- Van Loan discretisation matches direct quadrature; the diffusion inversion recovers Σ exactly.
- The Kalman likelihood matches a brute-force multivariate normal density; the fast collapsed filter matches the plain filter.
- Identities: P = Q ⇒ zero term premium; zero volatility ⇒ P-affine yield equals the EH rate.
- **Parameter recovery** (simulation, 40 years monthly): MLE log-likelihood exceeds the true-parameter likelihood, Q parameters and measurement errors are recovered, fitted yields are at the noise level, and the estimated 10y term premium has a correlation above 0.99 with the true one (the level differs by around 10bp, see below).

## Where the model is strong, and where to be careful

**Strong:** exact and internally consistent pricing; no-arbitrage by construction; fits the cross-section to a few basis points; an exact likelihood that can be validated end-to-end by simulation.

**Careful:**
- **P parameters are weakly identified.** The cross-section of yields pins down Q tightly; a few decades of persistent monthly data say much less about K_P and θ_P. Small-sample bias typically makes estimated mean reversion too fast, which shifts long yields into the term premium. In the simulation example the TP *dynamics* match the truth closely while the *level* is biased. Check the K_P standard errors in the diagnostics tab.
- **Gaussian dynamics:** constant volatility, yields can go negative, the zero lower bound is not modelled.
- **No survey or macro information**, unlike Kim-Wright. Differences from ACM/KW are expected and informative rather than errors to be tuned away.
- **The decomposition benchmark is model-based.** The P-affine yield is not an observed expectation and differs from a strict EH component by P-convexity.

## Prerequisites

Fixed income and zero-coupon pricing, continuously compounded yields, P vs Q measures, mean-reverting processes and the Vasicek model, matrix algebra, affine term structure models, state-space models, the Kalman filter, MLE, and Python with NumPy, Pandas, SciPy and Matplotlib.

## References

- Vasicek (1977); Duffie & Kan (1996); Dai & Singleton (2000); Duffee (2002)
- Joslin, Singleton & Zhu (2011), *A New Perspective on Gaussian Dynamic Term Structure Models*
- Kim & Wright (2005), FEDS 2005-33; Adrian, Crump & Moench (2013), *JFE*
- Gürkaynak, Sack & Wright (2007), *The U.S. Treasury Yield Curve: 1961 to the Present*
- Durbin & Koopman, *Time Series Analysis by State Space Methods*; Jungbacker & Koopman (2015)
- Van Loan (1978), *Computing Integrals Involving the Matrix Exponential*
