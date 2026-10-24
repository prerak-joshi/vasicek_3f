# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [1.1.0] — 2026-10-24

### Added
- **Bias-corrected P dynamics** (`bias_correction.py`): Kilian (1998) bootstrap correction of the
  VAR persistence on filtered states, in the spirit of Bauer, Rudebusch & Wu (2012). Reported
  alongside the MLE term premium.
- **Parameter-uncertainty bands** (`uncertainty.py`): 90% pointwise bands for the term premium from
  draws of the MLE's asymptotic distribution, re-running the smoother for each draw.
- **Out-of-sample evaluation** (`forecast.py`): expanding-window forecasts vs a random walk, with
  Diebold-Mariano tests (`--oos`).
- **Portable results bundles** (`serialization.py`): gzipped JSON instead of pickle, so saved runs load
  under any pandas/numpy version (e.g. on Streamlit Community Cloud).
- Rebuilt **dashboard** for practitioners: summary page with data-driven takeaways, yield-change
  attribution, MLE vs bias-corrected toggle, bands, recession shading, range selectors, forecasting
  tab, methodology tab, CSV downloads, consistent visual theme; modular code (`dashboard/views/`).
- **Documentation site** (MkDocs Material + API reference) published to GitHub Pages, and a `wiki/`
  folder generated from the same sources.
- Tooling: ruff lint/format, pre-commit, CI matrix (3.11–3.13) with coverage and a dashboard smoke test,
  issue/PR templates, `CITATION.cff`, `CONTRIBUTING.md`.

### Changed
- `standard_errors` now also returns the covariance of the unconstrained estimates.
- Package metadata (URLs, classifiers, optional extras `docs` and `dev`).

## [1.0.0] — 2026-10-23

### Added
- Three-factor Gaussian Vasicek model in canonical form, closed-form Q loadings, ODE-based P loadings.
- Exact discretisation (Van Loan), Kalman filter with collapsed likelihood and RTS smoother.
- Two-stage starting values, maximum likelihood, numerical-Hessian standard errors.
- Term premium decomposition (pricing-measure fitted yield minus physical-measure affine yield).
- GSW / ACM / Kim-Wright loaders, CLI pipeline, Streamlit dashboard, walkthrough notebook, tests.
