"""
Parameter-recovery experiment: simulate from known parameters, estimate, compare.

This is the most important end-to-end test: it checks that the estimator finds a
likelihood at least as high as the truth, recovers the Q-measure (cross-sectional)
parameters well, and tracks the true term premium closely in its time variation.
The P-measure parameters are deliberately NOT tested tightly - they are weakly
identified in samples of realistic length, which is a central lesson of the project.
"""

import numpy as np
import pytest

from vasicek3f.decomposition import term_premium_decomposition
from vasicek3f.estimation import estimate
from vasicek3f.model import filter_and_smooth, loglik
from vasicek3f.simulate import default_true_params, simulate_panel

TAU = np.arange(1.0, 11.0)
DT = 1.0 / 12.0


@pytest.fixture(scope="module")
def fitted():
    true = default_true_params(TAU.size)
    sim = simulate_panel(true, TAU, 480, DT, seed=7)
    res = estimate(sim["yields"].values, TAU, DT, compute_se=False, polish=False, ftol=1e-10, verbose=False)
    return true, sim, res


def test_mle_beats_true_parameters(fitted):
    true, sim, res = fitted
    assert res.loglik >= loglik(true, sim["yields"].values, TAU, DT) - 1e-6


def test_q_parameters_recovered(fitted):
    true, _, res = fitted
    assert np.allclose(res.params.kappa_Q, true.kappa_Q, rtol=0.25)
    assert abs(res.params.delta0 - true.delta0) < 0.01
    assert np.allclose(res.params.sigma_e, true.sigma_e, rtol=0.25)


def test_fitted_yields_at_noise_level(fitted):
    true, sim, res = fitted
    fs = filter_and_smooth(res.params, sim["yields"].values, TAU, DT)
    rmse = np.sqrt(np.mean((fs["fitted_smoothed"] - sim["yields"].values) ** 2))
    assert rmse < 1.2 * true.sigma_e.mean()


def test_term_premium_tracks_truth(fitted):
    true, sim, res = fitted
    fs = filter_and_smooth(res.params, sim["yields"].values, TAU, DT)
    dec = term_premium_decomposition(res.params, fs["x_smoothed"], TAU, sim["yields"].index)
    est_tp = dec["term_premium"][10.0]
    true_tp = sim["term_premium"][10.0]
    assert est_tp.corr(true_tp) > 0.9
