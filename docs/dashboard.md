# Dashboard guide

**Live app:** <https://vasicek3f-5ygmrt2fr8rvg97zfofzlq.streamlit.app> · locally: `streamlit run streamlit_app.py`

## Who it is for

Fixed income and quant practitioners who want to inspect a transparent affine term structure model: how well it fits, what it says about the term premium, how robust that answer is, and how the estimates connect to the equations. It also works as a teaching tool for students who know the theory.

## Sidebar

| Control | What it does |
|---|---|
| **Results source** | *Saved results* opens a published run from `results/` (US Treasuries first if available, otherwise the simulation). *Estimate on simulated data* simulates a panel from known parameters and estimates it. *Estimate on GSW data* estimates on an uploaded (or local) Fed GSW file, with optional ACM and Kim-Wright files. |
| **Fast mode** | For estimation: no standard errors or bands, looser tolerance, no polish step. For the dashboard: sensitivity holds states fixed by default and loading charts use a coarser grid. |
| **Focus maturity** | The maturity used for headline numbers, decomposition and sensitivity charts. |
| **Bias-corrected P dynamics** | Switches the decomposition from MLE P dynamics to bootstrap bias-corrected P dynamics (if the run includes them). |

The header shows the data source, sample, maturities, model specification and which P dynamics are active.

## Tabs

### Summary
- **Headline metrics:** latest yield and term premium, 1/3/12-month changes, historical percentile, 90% parameter band.
- **Key takeaways:** sentences generated from the numbers — level and percentile, how much of the last year's yield change came from expectations vs term premium, the gap between MLE and bias-corrected estimates, agreement with benchmarks, and the out-of-sample result.
- **Term premium history** with the band, the alternative P estimate, benchmarks (or the true premium in simulation) and NBER recession shading for US data.
- **What moved the curve?** Attribution of yield changes over 1–24 months into expectations, term premium and pricing error, for every maturity, with a downloadable table.

### Yield curve
Observed vs fitted yields over time and on two chosen dates, the pricing-error heatmap and per-maturity error statistics. *Model yields are closed-form Q loadings applied to smoothed states.*

### Term premium
Components of the chosen yield (Q fitted, P affine, optional strict EH rate), MLE vs bias-corrected term premium with the band, the decomposition across maturities on any date, term-premium factor loadings, the P-convexity gap, and the comparison with ACM and Kim-Wright (statistics for both P estimates). CSV download of the series.

### Factors
Smoothed latent factors with ±2 s.d. bands (true factors overlaid in simulation), Q vs P loadings, intercepts, and correlations with empirical level, slope and curvature.

### Forecasting
Expected future yield curves under P from the latest filtered state, and the out-of-sample evaluation: RMSE-ratio heatmap by horizon and maturity, Diebold-Mariano statistics, the cumulative squared-error advantage over the random walk through time, and κᵠ stability across refits.

### Sensitivity
Sliders shock blocks of the estimated parameters — κᵠ multipliers, δ₀, Kᴾ speed, the P long-run short rate, Σ — and recompute fit, likelihood and term premium.

**This is sensitivity analysis, not re-estimation.** Nothing is re-optimised: the shocked parameter vector is fed through the same pricing (and optionally filtering) functions, and the change in log-likelihood shows how much worse it fits. The typical pattern is the key lesson of the model: Q shocks wreck the fit (likelihood falls by hundreds of points), while P shocks move the term premium substantially at a cost of only a few points. Optionally the Kalman filter can re-infer the states under the shocked parameters, which lets the states absorb part of a Q shock.

### Model & estimation
Estimation statistics, the parameter table (two-stage start, MLE, truth in simulation, standard errors and t-statistics), factor half-lives under Q, MLE-P and bias-corrected P, long-run short rates, market prices of risk, standardised prediction-error diagnostics, and JSON/CSV downloads.

### Methodology
Scope, caveats and the full mathematical documentation.

Every chart has a *How this connects to the model* note naming the equations and package functions behind it.

## Runtime

| Operation | Typical time |
|---|---|
| Open a saved run | 1–3 s |
| Estimate, fast mode (simulated, 480 months × 10 maturities) | ~10 s |
| Estimate, full mode with standard errors and bands | ~40 s |
| Out-of-sample evaluation (CLI `--oos`) | 1–3 min |
| Sensitivity, states fixed / re-filtered | ~15 ms / ~50–100 ms |

The slow parts are numerical: the MLE uses finite-difference gradients over 32 parameters, standard errors need a numerical Hessian, bands re-run the smoother for 200 parameter draws, and the out-of-sample test re-estimates the model every 12 months. Physical-measure loadings require solving ODEs, which is why loading charts are coarser in fast mode. Estimation results are cached, so moving between tabs never re-runs them. For the best experience, generate runs with the CLI and open them as saved results.

## Downloads

CSV downloads are available for the decomposition at the focus maturity, the yield-change attribution, the term premium series with bands, pricing errors, the out-of-sample table and the parameter table; parameters are also available as JSON.
