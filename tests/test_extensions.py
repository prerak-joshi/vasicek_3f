"""Tests for bias correction, forecasting, uncertainty bands and bundle serialisation."""

import numpy as np
import pandas as pd
import pytest

from vasicek3f.affine import q_loadings
from vasicek3f.bias_correction import bias_correct_P, ols_P
from vasicek3f.discretize import discretize_ou
from vasicek3f.estimation import estimate
from vasicek3f.forecast import diebold_mariano, forecast_states, forecast_yields
from vasicek3f.serialization import dump_bundle, load_bundle_file
from vasicek3f.simulate import default_true_params, simulate_panel
from vasicek3f.uncertainty import term_premium_bands

TAU = np.arange(1.0, 11.0)
DT = 1.0 / 12.0


def _persistent_var(T=300, seed=0):
    """Simulate a highly persistent 3-dim VAR(1) from known OU parameters."""
    p = default_true_params()
    c, F, Q = discretize_ou(p.K_P, p.theta_P, p.Sigma, DT)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Q)
    X = np.empty((T, 3))
    x = p.theta_P.copy()
    for t in range(T):
        x = c + F @ x + L @ rng.standard_normal(3)
        X[t] = x
    return p, X


def test_bias_correction_increases_persistence():
    """OLS persistence is biased down; the bootstrap correction should raise the largest root."""
    _, X = _persistent_var()
    bc = bias_correct_P(X, DT, n_boot=150, seed=1)
    assert bc.max_root_bc > bc.max_root_ols
    assert bc.max_root_bc < 1.0
    K_ols, _ = ols_P(X, DT)
    assert np.min(np.real(np.linalg.eigvals(bc.K_P))) < np.min(np.real(np.linalg.eigvals(K_ols)))


def test_bias_correction_apply_keeps_Q_block():
    p, X = _persistent_var()
    bc = bias_correct_P(X, DT, n_boot=50)
    q = bc.apply(p)
    assert np.allclose(q.kappa_Q, p.kappa_Q) and np.allclose(q.Sigma, p.Sigma)
    assert q.delta0 == p.delta0 and np.allclose(q.K_P, bc.K_P)


def test_forecast_h0_and_long_horizon():
    p = default_true_params()
    x = np.array([0.01, -0.005, 0.002])
    assert np.allclose(forecast_states(p, x, 0, DT), x)
    far = forecast_states(p, x, 12 * 200, DT)
    assert np.allclose(far, p.theta_P, atol=1e-8)
    a, b = q_loadings(p, TAU)
    assert np.allclose(forecast_yields(p, x, TAU, 0, DT), a + b @ x)


def test_diebold_mariano_sign():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1, 500)
    e_bad = rng.normal(0, 2, 500)
    assert diebold_mariano(e_good, e_bad, 1) < -5
    assert diebold_mariano(e_bad, e_good, 1) > 5


@pytest.fixture(scope="module")
def small_fit():
    p = default_true_params(TAU.size)
    sim = simulate_panel(p, TAU, 240, DT, seed=3)
    res = estimate(sim["yields"].values, TAU, DT, compute_se=True, polish=False, ftol=1e-10, verbose=False)
    return sim, res


def test_bands_bracket_point_estimate(small_fit):
    sim, res = small_fit
    from vasicek3f.decomposition import term_premium_decomposition
    from vasicek3f.model import filter_and_smooth

    b = term_premium_bands(res.u_hat, res.cov_u, res.transform, sim["yields"], TAU, DT, n_draws=40, seed=0)
    fs = filter_and_smooth(res.params, sim["yields"].values, TAU, DT)
    tp = term_premium_decomposition(res.params, fs["x_smoothed"], TAU, sim["yields"].index)["term_premium"]
    inside = ((tp[10.0] >= b["q05"][10.0]) & (tp[10.0] <= b["q95"][10.0])).mean()
    assert b["n_valid"] >= 30 and inside > 0.9
    assert (b["q95"][10.0] >= b["q05"][10.0]).all()


def test_bundle_roundtrip(tmp_path, small_fit):
    sim, res = small_fit
    bundle = {
        "params": res.params,
        "yields": sim["yields"],
        "series": sim["term_premium"][10.0],
        "arr": np.arange(6.0).reshape(2, 3),
        "nested": {"a": [1, 2.5, None], 3.0: "x"},
        "nan": float("nan"),
        "ts": pd.Timestamp("2020-01-31"),
        "maturities": TAU,
    }
    path = tmp_path / "b.json.gz"
    dump_bundle(bundle, str(path))
    out = load_bundle_file(str(path))
    pd.testing.assert_frame_equal(out["yields"], sim["yields"], check_freq=False)
    pd.testing.assert_series_equal(out["series"], sim["term_premium"][10.0], check_freq=False)
    assert np.allclose(out["params"].K_P, res.params.K_P)
    assert np.allclose(out["arr"], bundle["arr"]) and out["nested"][3.0] == "x"
    assert np.isnan(out["nan"]) and out["ts"] == bundle["ts"]
