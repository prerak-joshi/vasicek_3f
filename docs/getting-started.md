# Getting started

## Install

Python 3.11–3.13.

```bash
git clone https://github.com/prerak-joshi/vasicek_3f.git
cd vasicek_3f
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt  # runtime + tests + docs + notebook
pip install -e .                     # installs the `vasicek3f` command
```

`requirements.txt` holds only the runtime dependencies (what Streamlit Cloud installs).

## Run on simulated data

No downloads needed; the true parameters are known, so you can check recovery.

```bash
vasicek3f run --source simulated --oos --out results/simulated
```

## Run on US Treasuries

### 1. Get the data

```bash
vasicek3f download --dest data
```

If a site blocks the scripted download, save the files manually into `data/`:

| Series | Where | File name |
|---|---|---|
| GSW zero-coupon yields | federalreserve.gov → Data → *Treasury yield curve* → `feds200628.csv` | `feds200628.csv` |
| ACM term premia | newyorkfed.org → Research → *ACM Term Premia* → `ACMTermPremium.xls` | `ACMTermPremium.xls` |
| Kim-Wright 10y term premium | FRED series `THREEFYTP10` → *Download CSV* | `THREEFYTP10.csv` |

### 2. Estimate

```bash
vasicek3f run --source gsw \
    --gsw data/feds200628.csv --acm data/ACMTermPremium.xls --kw data/THREEFYTP10.csv \
    --start 1990-01 --oos --out results/gsw
```

Typical runtime is a few minutes; the out-of-sample test is the slowest step. Results land in `results/gsw/`:

| File | Contents |
|---|---|
| `summary.md` | estimation summary, fit, term premium, OOS and benchmark tables, figures |
| `bundle.json.gz` | everything the dashboard needs (portable across library versions) |
| `params.json`, `std_errors.csv` | parameter estimates and standard errors |
| `term_premium_pct.csv`, `term_premium_bias_corrected_pct.csv` | term premium by maturity, % |
| `term_premium_10y_bands_pct.csv` | 90% parameter-uncertainty band, % |
| `fitted_Q_pct.csv`, `affine_P_pct.csv`, `eh_short_rate_pct.csv` | decomposition components, % |
| `oos_forecast_summary.csv` | out-of-sample RMSE vs random walk and DM statistics |
| `figures/*.png` | static charts |

### Useful options

| Flag | Effect |
|---|---|
| `--fast` | no standard errors or bands, looser tolerance (quick look) |
| `--oos` | expanding-window out-of-sample evaluation |
| `--maturities 1 2 3 5 7 10` | choose maturities (integer years for GSW) |
| `--tp-maturity 5` | maturity used for headline term premium outputs |
| `--start 1990-01 --end 2019-12` | sample period |
| `--no-bias-correction`, `--no-bands` | skip those steps |
| `--restarts 3` | extra randomly perturbed MLE starts |
| `--common-error` | one measurement-error s.d. for all maturities |

## Dashboard

```bash
streamlit run streamlit_app.py
```

It opens the saved runs in `results/` (US Treasuries first if present). See the [Dashboard guide](dashboard.md).

## Notebook

`notebooks/01_vasicek3f_walkthrough.ipynb` walks through the model one module at a time.

## Tests and quality checks

```bash
pytest --cov
ruff check . && ruff format --check .
mkdocs serve        # preview this documentation at http://127.0.0.1:8000
```

## Deploying the dashboard

The repository is ready for Streamlit Community Cloud: entry point `streamlit_app.py`, runtime dependencies in `requirements.txt`, theme in `.streamlit/config.toml`. On share.streamlit.io choose *Create app*, select the repository, branch `main`, file `streamlit_app.py`, and Python 3.12. No secrets are required. To publish real-data results, commit `results/gsw/` after running the pipeline locally (check the data providers' terms first).
