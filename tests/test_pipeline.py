"""End-to-end pipeline test (fast mode, small simulated panel) and a short OOS run."""

import os

import numpy as np

from vasicek3f.forecast import expanding_window_oos
from vasicek3f.pipeline import BUNDLE_FILE, RunConfig, load_bundle, run
from vasicek3f.simulate import default_true_params, simulate_panel


def test_pipeline_end_to_end(tmp_path):
    out = tmp_path / "run"
    cfg = RunConfig(
        source="simulated",
        sim_periods=180,
        fast=True,
        out_dir=str(out),
        make_plots=True,
        verbose=False,
        n_boot=50,
    )
    b = run(cfg)
    assert os.path.exists(out / BUNDLE_FILE) and os.path.exists(out / "summary.md")
    assert os.path.exists(out / "figures" / "05_decomposition.png")
    loaded = load_bundle(str(out))
    assert np.allclose(loaded["params"].kappa_Q, b["params"].kappa_Q)
    tp = loaded["decomposition"]["term_premium"]
    assert tp.shape == (180, 10) and np.isfinite(tp.values).all()
    assert loaded["decomposition_bc"] is not None
    assert loaded["pricing_errors"]["rmse_bp"].mean() < 10
    assert "True TP (simulation)" in loaded["benchmarks"]


def test_expanding_window_oos_short():
    tau = np.arange(1.0, 11.0)
    sim = simulate_panel(default_true_params(10), tau, 150, seed=5)
    o = expanding_window_oos(
        sim["yields"], tau, 1 / 12, initial=120, refit_every=30, horizons=(1, 6), verbose=False
    )
    s = o["summary"]
    assert set(s["horizon_m"]) == {1, 6} and len(s) == 20
    assert (s["rmse_model_bp"] > 0).all() and len(o["refits"]) == 1
    # origins never use data beyond themselves: first origin is the end of the initial window
    assert o["errors"]["origin"].min() == sim["yields"].index[119]
