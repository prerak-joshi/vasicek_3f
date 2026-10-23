# Dashboard guide

```bash
streamlit run streamlit_app.py      # or: streamlit run dashboard/app.py
```

## Who it is for

Quant finance and Master of Financial Engineering students, aspiring quant researchers, and practitioners preparing for quant interviews. It assumes you know fixed income basics, P vs Q, the Vasicek model and the Kalman filter at a textbook level, and want to see how those pieces produce actual numbers and charts. It is a teaching and exploration tool, not a monitoring tool for official term premium estimates.

## Sidebar

| Control | What it does |
|---|---|
| **Results source** | *Load saved results* reads `results/<run>/bundle.pkl` written by the CLI (instant; falls back to a fresh demo estimation if the file can't be read). *Estimate: simulated data* simulates a panel from known parameters and estimates it. *Estimate: GSW data* estimates on the Fed GSW file, uploaded or from a local path, adding ACM/KW benchmarks if provided. |
| **Fast mode** | Skips standard errors, loosens the optimiser tolerance, skips the BFGS polish, defaults the sensitivity tab to fixed states, and uses a coarser maturity grid for loading charts. |
| **Term premium maturity** | Which maturity the decomposition and sensitivity tabs focus on. |

Estimation results are cached by configuration, so switching tabs or moving sliders never re-runs the estimator.

## Tabs, and how each chart connects to the model

### 1. Overview
- **Metrics:** log-likelihood, average pricing RMSE, latest and mean term premium, runtime and mode.
- **Model at a glance:** P and Q dynamics, short rate, measurement equation, decomposition.
- **PCA bar chart:** variance explained by principal components of yields. Three components capture almost everything, which motivates three latent factors. *(`diagnostics.pca_explained`)*
- **Yield heatmap:** the raw input data.

### 2. Yield curve fit
- **Observed vs fitted (time series):** $\hat y_t(\tau)=a^Q(\tau)+b^Q(\tau)'\hat x_{t|T}$, i.e. closed-form Q loadings applied to smoothed states. *(`affine.q_loadings`, `kalman.rts_smoother`)*
- **Curve on a date:** the same, across maturities on one date.
- **Pricing-error heatmap and table:** observed minus fitted in bp, which are the measurement errors $e_t$. Small, serially uncorrelated errors mean three factors capture the cross-section.

### 3. Latent factors & loadings
- **Smoothed states with ±2 s.d. bands** from $P_{t|T}$. In simulation, dotted lines show the true states. Factors are identified only up to the canonical normalisation, so they are rotations of level/slope/curvature.
- **Q vs P loadings $b(\tau)$:** Q in closed form, P from the pricing ODEs with the physical drift. The gap between them generates the term premium. *(`affine.p_loadings`)*
- **Intercepts $a(\tau)$:** Q (δ0 minus convexity), P, and the strict EH intercept.
- **Factor correlation table** with empirical level, slope and curvature.

### 4. Term premium decomposition
- **Yield and components:** observed, Q fitted yield, P affine yield; optionally the strict EH average expected short rate.
- **Term premium:** $TP=y^Q-y^P$, with the true premium overlaid in simulation.
- **Decomposition across maturities** on a chosen date.
- **TP factor loadings** $b^Q-b^P$: which latent factor moves the premium.
- **Caption:** the P-convexity gap between the benchmark used here and a strict EH component.
*(`decomposition.term_premium_decomposition`)*

### 5. Benchmarks
Model term premium vs ACM and Kim-Wright on real data, or vs the true premium in simulation, with correlation of levels and 12-month changes, mean difference and RMSE. *(`diagnostics.benchmark_comparison`)* Expect better agreement in direction than in level: KW uses survey forecasts to discipline P, which this model does not.

### 6. Sensitivity (sliders)
| Slider | Parameter block | Main effect with states held fixed |
|---|---|---|
| κQ1–3 multipliers | Q mean reversion | Q loadings change → fitted yields move away from data; likelihood drops sharply |
| δ0 shift | Q long-run short rate | shifts all fitted yields; poor fit |
| K_P multiplier | P mean-reversion speed | fit unchanged; the time variation of the P affine yield and TP changes (the sample-mean TP moves little because the P long-run mean is unchanged) |
| P long-run short rate shift | θ_P (first factor) | fit unchanged; TP level shifts, more so at long maturities |
| Σ multiplier | volatility | convexity under both measures; small effect on fit and TP |

Outputs: Δ log-likelihood vs the MLE, pricing RMSE, mean and latest TP, baseline vs shocked TP series, and average curves.

**Why this is sensitivity analysis rather than re-estimation.** The sliders take the *estimated* parameter vector, shock one block, and re-run the same pricing (and optionally filtering) functions. No other parameter is allowed to adjust, and no likelihood is maximised. Re-estimation would re-optimise all 32 parameters, which takes seconds to minutes, and would generally offset part of the shock through other parameters. The Δ log-likelihood readout makes the distinction explicit. The pattern it reveals is the key lesson: Q shocks are punished heavily by the data, while P shocks change the term premium a lot at little cost in likelihood. The P side is weakly identified.

**Re-run Kalman filter at shocked parameters** (checkbox): instead of holding states at their estimated values, the filter and smoother re-infer the states under the shocked parameters (still no optimisation). For Q shocks this lets the states partially absorb the change and restore some of the fit.

### 7. Estimation & diagnostics
- Parameter table: two-stage start, MLE, true values (simulation) and standard errors / t-stats (full mode).
- Factor half-lives under Q and P, long-run short rate under each measure.
- Market prices of risk $\lambda_0,\lambda_1$.
- Standardised one-step-ahead prediction errors (should be ≈ N(0,1) with low autocorrelation).

## Runtime, fast mode, and why some parts are slower

| Operation | Typical time | Why |
|---|---|---|
| Load saved bundle | ~1–2 s | unpickling plus first render |
| Fast-mode estimation (simulated, 480×10) | ~10 s | L-BFGS-B with finite-difference gradients: ~33 likelihood evaluations per gradient at ~3 ms each |
| Full-mode estimation | ~25 s | adds a BFGS polish and a 32×32 numerical Hessian (~2,000 likelihood evaluations) for standard errors |
| Real GSW data | usually longer | harder likelihood surface, more iterations |
| Sensitivity, states fixed | ~15 ms | closed-form Q loadings, ODE solve for P loadings |
| Sensitivity, re-filtered | ~50–100 ms | full Kalman filter and smoother with stored histories |

The likelihood itself is fast because of two exact shortcuts (collapsed filter, steady-state gain); see `docs/THEORY.md` §6. P-measure loadings need a numerical ODE solve, which is why fast mode uses a coarser maturity grid for loading charts. For the quickest experience, produce bundles once with the CLI and load them in the dashboard.

## Educational scope

The dashboard displays results from an educational model. The term premium shown is a model-based decomposition with a P-affine benchmark, sensitive to weakly identified P parameters, from a Gaussian model with no survey information. Use it to understand affine term structure modelling, not as an official or tradable term premium measure.
