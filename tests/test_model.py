"""
Unit tests for pricing, discretisation, filtering and the decomposition identities.
Run:  pytest -q
"""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from vasicek3f.affine import eh_loadings, p_loadings, q_loadings, q_loadings_ode
from vasicek3f.discretize import diffusion_from_discrete_cov, discretize_ou
from vasicek3f.kalman import kalman_filter, kalman_loglik_collapsed
from vasicek3f.model import build_state_space, loglik
from vasicek3f.params import ParamTransform
from vasicek3f.simulate import default_true_params, simulate_panel

TAU = np.arange(1.0, 11.0)
DT = 1.0 / 12.0


@pytest.fixture(scope="module")
def p():
    return default_true_params(TAU.size)


@pytest.fixture(scope="module")
def y(p):
    return simulate_panel(p, TAU, 240, DT, seed=1)["yields"].values


# ---------------------------------------------------------------- pricing
def test_closed_form_q_loadings_match_ode(p):
    a, b = q_loadings(p, TAU)
    a2, b2 = q_loadings_ode(p, TAU)
    assert np.allclose(a, a2, atol=1e-12)
    assert np.allclose(b, b2, atol=1e-9)


def test_short_end_limit(p):
    """As tau -> 0 the yield converges to the short rate: a -> delta0, b -> 1."""
    a, b = q_loadings(p, np.array([1e-6]))
    assert abs(a[0] - p.delta0) < 1e-10
    assert np.allclose(b[0], 1.0, atol=1e-5)


def test_convexity_lowers_long_yields(p):
    a, _ = q_loadings(p, TAU)
    assert np.all(np.diff(a) < 0)
    zero_vol = p.copy()
    zero_vol.Sigma = np.zeros((3, 3))
    a0, _ = q_loadings(zero_vol, TAU)
    assert np.allclose(a0, p.delta0)


def test_term_premium_zero_when_P_equals_Q(p):
    """If P dynamics equal Q dynamics, the P-affine yield equals the Q yield exactly."""
    q = p.copy()
    q.K_P = q.K_Q.copy()
    q.theta_P = np.zeros(3)
    aQ, bQ = q_loadings(q, TAU)
    aP, bP = p_loadings(q, TAU)
    assert np.allclose(aQ, aP, atol=1e-12) and np.allclose(bQ, bP, atol=1e-9)


def test_p_affine_equals_eh_minus_convexity(p):
    """With zero volatility there is no convexity, so the P-affine yield equals the EH rate."""
    q = p.copy()
    q.Sigma = np.zeros((3, 3))
    aP, bP = p_loadings(q, TAU)
    aE, bE = eh_loadings(q, TAU)
    assert np.allclose(aP, aE, atol=1e-10) and np.allclose(bP, bE, atol=1e-9)
    # with volatility, EH exceeds the P-affine yield (positive convexity term)
    aP, _ = p_loadings(p, TAU)
    aE, _ = eh_loadings(p, TAU)
    assert np.all(aE - aP > 0)


# ---------------------------------------------------------------- discretisation
def test_van_loan_matches_quadrature(p):
    _, F, Q = discretize_ou(p.K_P, p.theta_P, p.Sigma, DT)
    Qn = quad_vec(lambda s: expm(-p.K_P * s) @ p.SigmaSigmaT @ expm(-p.K_P.T * s), 0, DT)[0]
    assert np.allclose(F, expm(-p.K_P * DT))
    assert np.allclose(Q, Qn, rtol=1e-10, atol=1e-16)


def test_diffusion_inversion(p):
    _, _, Q = discretize_ou(p.K_P, p.theta_P, p.Sigma, DT)
    assert np.allclose(diffusion_from_discrete_cov(p.K_P, Q, DT), p.Sigma, atol=1e-12)


# ---------------------------------------------------------------- filtering
def test_kalman_matches_brute_force_gaussian(p, y):
    T = 5
    ys = y[:T]
    ss = build_state_space(p, TAU, DT)
    d = 3
    Px = np.zeros((T * d, T * d))
    for i in range(T):
        for j in range(T):
            blk = (
                np.linalg.matrix_power(ss.F, i - j) @ ss.P0
                if i >= j
                else ss.P0 @ np.linalg.matrix_power(ss.F, j - i).T
            )
            Px[i * d : (i + 1) * d, j * d : (j + 1) * d] = blk
    Hb = np.kron(np.eye(T), ss.H)
    cov = Hb @ Px @ Hb.T + np.kron(np.eye(T), ss.R)
    mu = np.tile(ss.a + ss.H @ ss.x0, T)
    brute = multivariate_normal(mu, cov).logpdf(ys.reshape(-1))
    kf = kalman_filter(ys, ss.a, ss.H, ss.R, ss.c, ss.F, ss.Q, ss.x0, ss.P0).loglik
    assert abs(brute - kf) < 1e-7


def test_collapsed_filter_equals_full_filter(p, y):
    ss = build_state_space(p, TAU, DT)
    full = kalman_filter(y, ss.a, ss.H, ss.R, ss.c, ss.F, ss.Q, ss.x0, ss.P0, store=False).loglik
    col = kalman_loglik_collapsed(y, ss.a, ss.H, np.diag(ss.R), ss.c, ss.F, ss.Q, ss.x0, ss.P0)
    assert abs(full - col) < 1e-6


def test_missing_values_handled(p, y):
    ym = y.copy()
    ym[10, 3] = np.nan
    ym[20, :] = np.nan
    ll = loglik(p, ym, TAU, DT)
    assert np.isfinite(ll) and ll < loglik(p, y, TAU, DT) + 1e3


def test_nonstationary_K_P_rejected(p, y):
    q = p.copy()
    q.K_P = -np.eye(3) * 0.1
    assert loglik(q, y, TAU, DT) == -np.inf


# ---------------------------------------------------------------- parameters
def test_transform_roundtrip(p):
    tr = ParamTransform(TAU.size)
    q = tr.to_params(tr.to_unconstrained(p))
    for k in ("kappa_Q", "Sigma", "K_P", "theta_P", "sigma_e"):
        assert np.allclose(getattr(p, k), getattr(q, k))
    assert abs(p.delta0 - q.delta0) < 1e-14


def test_market_price_of_risk_identity(p):
    m = p.market_price_of_risk()
    assert np.allclose(p.Sigma @ m["lambda1"], p.K_Q - p.K_P)
    assert np.allclose(p.Sigma @ m["lambda0"], p.K_P @ p.theta_P)


def test_transform_kappa_floor(p):
    tr = ParamTransform(TAU.size, common_measurement_error=True, kappa_min=0.02)
    u = tr.to_unconstrained(p)
    q = tr.to_params(u)
    assert np.allclose(q.kappa_Q, p.kappa_Q)
    assert np.allclose(q.sigma_e, np.mean(p.sigma_e))
    u[0] = -50.0  # pushing kappa_1 towards zero stops at the floor
    assert tr.to_params(u).kappa_Q[0] >= 0.02


def test_kp_min_penalty_only_when_binding(p, y):
    from vasicek3f.estimation import make_objective

    tr = ParamTransform(TAU.size)
    u = tr.to_unconstrained(p)
    base = make_objective(y, TAU, DT, tr)(u)
    assert make_objective(y, TAU, DT, tr, kp_min=0.1)(u) == base  # slowest P rate 0.2 > 0.1
    assert make_objective(y, TAU, DT, tr, kp_min=0.5)(u) > base
