# Validation

The test suite (`pytest --cov`, about a minute, ~90% line coverage) checks the implementation against independent calculations and against simulations with known truth.

## Exactness checks

| Check | Test |
|---|---|
| Closed-form Q loadings equal the numerical ODE solution (≈1e-12) | `test_closed_form_q_loadings_match_ode` |
| Short-end limit: $a\to\delta_0$, $b\to 1$ as $\tau\to 0$ | `test_short_end_limit` |
| Convexity lowers long yields; zero volatility removes it | `test_convexity_lowers_long_yields` |
| $K^P=K^Q$, $\theta^P=0$ ⇒ zero term premium | `test_term_premium_zero_when_P_equals_Q` |
| Zero volatility ⇒ P-affine yield equals the EH average short rate | `test_p_affine_equals_eh_minus_convexity` |
| Van Loan covariance equals direct quadrature | `test_van_loan_matches_quadrature` |
| Diffusion inversion recovers $\Sigma$ exactly | `test_diffusion_inversion` |
| Kalman likelihood equals brute-force multivariate normal density | `test_kalman_matches_brute_force_gaussian` |
| Collapsed fast likelihood equals the plain filter | `test_collapsed_filter_equals_full_filter` |
| Market-price-of-risk identities | `test_market_price_of_risk_identity` |
| Bundle serialisation round-trips frames, series, arrays, parameters | `test_bundle_roundtrip` |

## Simulation recovery (40 years, monthly, 10 maturities)

| Result | Outcome |
|---|---|
| MLE log-likelihood ≥ likelihood at the true parameters | yes |
| κᵠ, δ₀, measurement errors recovered | within 10–15% |
| Fitted-yield RMSE vs noise level | ≈3.5bp vs 4bp noise |
| Correlation of estimated and true 10y term premium | 0.999 |
| Level of the 10y term premium | biased up by ~10bp (mean 0.56% vs 0.45%) |
| Largest P root: true / MLE / bias-corrected | half-life 3.5y / 1.5y / 2.0y |
| 90% parameter band coverage of the true 10y term premium | 100% of months (avg width ≈70bp) |
| Out-of-sample 12m RMSE ratio vs random walk | 0.79–0.88 (model better, as expected in a stationary simulation) |

The pattern — excellent fit and tracking, a biased level, too-fast estimated mean reversion that the bootstrap partly corrects — is exactly the identification lesson discussed in [Interpreting the term premium](term-premium.md). On real data the out-of-sample comparison is typically much less favourable to the model.

## Continuous integration

GitHub Actions runs on every push and pull request: ruff lint and format checks, the test suite on Python 3.11, 3.12 and 3.13 with coverage, and a headless smoke test of the dashboard.
