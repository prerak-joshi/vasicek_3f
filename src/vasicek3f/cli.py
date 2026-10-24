"""
Command-line interface.

    python -m vasicek3f download --dest data
    python -m vasicek3f run --source simulated --out results/simulated
    python -m vasicek3f run --source gsw --gsw data/feds200628.csv \
        --acm data/ACMTermPremium.xls --kw data/THREEFYTP10.csv \
        --start 1990-01 --out results/gsw
"""

from __future__ import annotations

import argparse

from .data import download_all
from .pipeline import RunConfig, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="vasicek3f", description="3-factor Vasicek term structure model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="download GSW, ACM and Kim-Wright data")
    d.add_argument("--dest", default="data")

    r = sub.add_parser("run", help="estimate the model and write results")
    r.add_argument("--source", choices=["simulated", "gsw"], default="simulated")
    r.add_argument("--gsw", dest="gsw_path")
    r.add_argument("--acm", dest="acm_path")
    r.add_argument("--kw", dest="kw_path")
    r.add_argument("--start", default="1990-01")
    r.add_argument("--end", default=None)
    r.add_argument("--maturities", type=float, nargs="+", default=[float(m) for m in range(1, 11)])
    r.add_argument("--tp-maturity", type=float, default=10.0)
    r.add_argument("--fast", action="store_true", help="skip standard errors, cap iterations")
    r.add_argument("--no-se", action="store_true", help="skip numerical-Hessian standard errors")
    r.add_argument("--restarts", type=int, default=1)
    r.add_argument("--maxiter", type=int, default=4000)
    r.add_argument(
        "--common-error", action="store_true", help="one measurement error s.d. for all maturities"
    )
    r.add_argument("--sim-periods", type=int, default=480)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--no-plots", action="store_true")
    r.add_argument("--no-bias-correction", action="store_true", help="skip bootstrap bias correction of P")
    r.add_argument("--n-boot", type=int, default=500)
    r.add_argument("--no-bands", action="store_true", help="skip parameter-uncertainty bands")
    r.add_argument("--band-draws", type=int, default=200)
    r.add_argument("--oos", action="store_true", help="run the expanding-window out-of-sample test (slow)")
    r.add_argument("--oos-refit-every", type=int, default=12)
    r.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "download":
        download_all(a.dest)
        return

    cfg = RunConfig(
        source=a.source,
        gsw_path=a.gsw_path,
        acm_path=a.acm_path,
        kw_path=a.kw_path,
        start=a.start,
        end=a.end,
        maturities=a.maturities,
        tp_maturity=a.tp_maturity,
        fast=a.fast,
        compute_se=not a.no_se,
        n_restarts=a.restarts,
        maxiter=a.maxiter,
        common_measurement_error=a.common_error,
        sim_periods=a.sim_periods,
        seed=a.seed,
        out_dir=a.out or f"results/{a.source}",
        make_plots=not a.no_plots,
        bias_correct=not a.no_bias_correction,
        n_boot=a.n_boot,
        bands=not a.no_bands,
        n_band_draws=a.band_draws,
        oos=a.oos,
        oos_refit_every=a.oos_refit_every,
    )
    run(cfg)


if __name__ == "__main__":
    main()
