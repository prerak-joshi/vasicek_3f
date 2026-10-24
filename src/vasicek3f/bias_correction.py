"""
Small-sample bias correction of the physical (P) dynamics.

Why
---
The P-measure mean-reversion matrix K_P is estimated from the time series of
highly persistent factors. In samples of a few decades, least-squares and
maximum-likelihood estimates of persistence are biased towards *faster* mean
reversion (Kendall/Hurwicz bias). Faster mean reversion pulls expected future
short rates back to their long-run mean too quickly, which flattens the
physical-measure affine yield and pushes more of the long yield into the term
premium. Bauer, Rudebusch & Wu (2012, JBES) show this materially changes
estimated term premia.

What this module does (a transparent two-step adjustment)
--------------------------------------------------------
1. Take the Kalman-filtered states x_{t|t} from the MLE.
2. Fit a demeaned VAR(1) by OLS: x_t - m = F (x_{t-1} - m) + u_t, with m = sample mean.
3. Estimate the OLS bias in F with a residual bootstrap (Kilian 1998):
       bias = mean_b(F*_b) - F_hat
   and set F_bc = F_hat - bias, shrinking the correction if needed so that the
   largest root stays inside the unit circle (Kilian's stationarity adjustment).
4. Map to continuous time: K_P^bc = -logm(F_bc)/dt, theta_P^bc = m.
5. Keep all Q-measure parameters and Sigma at their MLE values.

The result is a *post-estimation* adjustment for the P dynamics only, not a joint
re-estimation. Both the plain-MLE and the bias-corrected term premia are reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import logm

from .params import N_FACTORS, VasicekParams


@dataclass
class BiasCorrection:
    """Output of :func:`bias_correct_P`."""

    K_P: np.ndarray
    theta_P: np.ndarray
    F_ols: np.ndarray
    F_bc: np.ndarray
    bias: np.ndarray
    shrink: float  # fraction of the estimated bias actually removed (Kilian adjustment)
    max_root_ols: float
    max_root_bc: float
    n_boot: int

    def half_lives(self, dt: float = 1 / 12) -> np.ndarray:
        """Half-lives (years) implied by the bias-corrected K_P eigenvalues."""
        return np.log(2.0) / np.real(np.linalg.eigvals(self.K_P))

    def apply(self, params: VasicekParams) -> VasicekParams:
        """Return a copy of ``params`` with the bias-corrected P dynamics."""
        q = params.copy()
        q.K_P = self.K_P.copy()
        q.theta_P = self.theta_P.copy()
        return q


def _ols_demeaned(Z: np.ndarray) -> np.ndarray:
    """OLS F in Z_t = F Z_{t-1} + u_t for demeaned data Z (T, d)."""
    X, Y = Z[:-1], Z[1:]
    return np.linalg.solve(X.T @ X, X.T @ Y).T


def bias_correct_P(
    states: np.ndarray, dt: float, n_boot: int = 500, seed: int = 0, max_root: float = 0.9995
) -> BiasCorrection:
    """
    Kilian (1998) bootstrap bias correction of a VAR(1) fitted to ``states``.

    Parameters
    ----------
    states : (T, 3) filtered states (decimals)
    dt     : sampling interval in years
    n_boot : bootstrap replications
    max_root : cap on the modulus of the corrected VAR's largest root
    """
    X = np.asarray(states, float)
    m = X.mean(axis=0)
    Z = X - m
    F_hat = _ols_demeaned(Z)
    U = Z[1:] - Z[:-1] @ F_hat.T
    U = U - U.mean(axis=0)
    T = len(Z)
    rng = np.random.default_rng(seed)

    F_star = np.zeros_like(F_hat)
    for _ in range(n_boot):
        idx = rng.integers(0, len(U), size=T)
        Zb = np.empty_like(Z)
        Zb[0] = Z[rng.integers(0, T)]
        for t in range(1, T):
            Zb[t] = F_hat @ Zb[t - 1] + U[idx[t]]
        Zb -= Zb.mean(axis=0)
        F_star += _ols_demeaned(Zb)
    bias = F_star / n_boot - F_hat

    shrink = 1.0
    F_bc = F_hat - bias
    while np.max(np.abs(np.linalg.eigvals(F_bc))) >= max_root and shrink > 0:
        shrink = max(shrink - 0.01, 0.0)
        F_bc = F_hat - shrink * bias

    K = -np.real(logm(F_bc)) / dt
    if np.any(np.real(np.linalg.eigvals(K)) <= 1e-4):
        raise RuntimeError("bias-corrected K_P is not stationary")
    return BiasCorrection(
        K_P=K,
        theta_P=m,
        F_ols=F_hat,
        F_bc=F_bc,
        bias=bias,
        shrink=shrink,
        max_root_ols=float(np.max(np.abs(np.linalg.eigvals(F_hat)))),
        max_root_bc=float(np.max(np.abs(np.linalg.eigvals(F_bc)))),
        n_boot=n_boot,
    )


def ols_P(states: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Plain (uncorrected) OLS P dynamics from the same states, for comparison."""
    X = np.asarray(states, float)
    m = X.mean(axis=0)
    F = _ols_demeaned(X - m)
    return -np.real(logm(F)) / dt, m


__all__ = ["BiasCorrection", "bias_correct_P", "ols_P", "N_FACTORS"]
