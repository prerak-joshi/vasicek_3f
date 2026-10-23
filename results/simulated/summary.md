# 3-Factor Vasicek term structure model: results

Source: **simulated**, 480 months, 1986-01 to 2025-12, maturities 1y, 2y, 3y, 4y, 5y, 6y, 7y, 8y, 9y, 10y. Runtime 26s.

## Estimation

- Log-likelihood: 28,455.51  (AIC -56,847.0, BIC -56,639.8, 32 parameters, 4800 observations)
- Optimiser: CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH after 129 iterations
- kappa_Q: [0.0531, 0.4294, 1.5393]  (Q half-lives, years: [13.05, 1.61, 0.45])
- eigenvalues of K_P (real part): [0.4748, 1.4968, 1.4968]  (P half-lives, years: [1.46, 0.46, 0.46])
- Long-run short rate: Q 6.20%, P 3.88%
- Measurement error s.d. (bp): [4.39, 3.5, 4.07, 4.29, 4.1, 4.02, 4.19, 3.94, 4.07, 3.82]

## Fit

PCA of yield levels (share of variance): PC1 96.29%, PC2 3.14%, PC3 0.30%, PC4 0.04%, PC5 0.04%

|   maturity |   mean_bp |   mae_bp |   rmse_bp |   max_abs_bp |   ar1 |
|-----------:|----------:|---------:|----------:|-------------:|------:|
|          1 |     -0.06 |     2.02 |      2.51 |         9.16 | -0.31 |
|          2 |      0.17 |     2.26 |      2.8  |         7.7  | -0.07 |
|          3 |     -0.25 |     2.82 |      3.61 |        11.36 | -0.07 |
|          4 |     -0.08 |     3.14 |      3.97 |        12.97 | -0    |
|          5 |      0.11 |     2.99 |      3.83 |        14.92 | -0.03 |
|          6 |      0.09 |     3.06 |      3.76 |        10.92 | -0.04 |
|          7 |     -0.02 |     3.13 |      3.91 |        13.42 | -0.01 |
|          8 |     -0.03 |     2.9  |      3.58 |         9.57 |  0.02 |
|          9 |      0    |     2.89 |      3.66 |        11.54 | -0.07 |
|         10 |     -0.01 |     2.65 |      3.31 |         9.95 | -0.1  |

## Term premium

10y term premium: mean 0.56%, s.d. 0.43%, min -0.51%, max 1.57%, latest 0.57% (2025-12).

Decomposition: **Term Premium = Pricing-Measure Fitted Yield − Physical-Measure Affine Yield**. The P-affine yield is a no-risk-premium benchmark inside the model, not a strict Expectations-Hypothesis average of expected short rates.

### Benchmark comparison

|                      |   n_months | start      | end        |   corr_level |   corr_12m_change |   mean_model_pct |   mean_bench_pct |   mean_diff_bp |   rmse_diff_bp |
|:---------------------|-----------:|:-----------|:-----------|-------------:|------------------:|-----------------:|-----------------:|---------------:|---------------:|
| True TP (simulation) |        480 | 1986-01-31 | 2025-12-31 |        0.999 |             0.998 |            0.563 |            0.454 |         10.834 |         17.450 |

## Figures

![01_fit](figures/01_fit.png)
![02_pricing_errors](figures/02_pricing_errors.png)
![03_loadings](figures/03_loadings.png)
![04_states](figures/04_states.png)
![05_decomposition](figures/05_decomposition.png)
![06_benchmarks](figures/06_benchmarks.png)
![07_average_curve](figures/07_average_curve.png)
