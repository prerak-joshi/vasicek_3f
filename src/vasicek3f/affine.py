"""
Affine bond pricing.

For a Gaussian state with drift  k0 + K1 x  (here K1 = -K) and diffusion Sigma,
and short rate r = delta0 + delta1' x, zero-coupon bond prices are exponential
affine:

    P(t, tau) = exp( A(tau) + B(tau)' x_t )

with A, B solving the Riccati (here linear) ODEs

    dB/dtau = -delta1 - K' B,                         B(0) = 0
    dA/dtau = -delta0 + k0' B + 1/2 B' Sigma Sigma' B, A(0) = 0

Continuously compounded yields are therefore affine in the state:

    y(tau) = -(A + B'x)/tau = a(tau) + b(tau)' x

This module provides
  * closed-form Q loadings for the canonical diagonal-K_Q model (used in the likelihood),
  * a generic ODE solver used for the P-measure affine benchmark and for testing,
  * the strict Expectations-Hypothesis average expected short rate (reference only).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .params import VasicekParams, N_FACTORS


def _phi(k: np.ndarray, tau: float) -> np.ndarray:
    """(1 - exp(-k tau)) / k, numerically stable for small k."""
    return -np.expm1(-k * tau) / k


def q_loadings(params: VasicekParams, maturities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Closed-form yield loadings under Q for K_Q = diag(kappa), theta_Q = 0, delta1 = iota.

        b_i(tau) = (1 - e^{-kappa_i tau}) / (kappa_i tau)
        a(tau)   = delta0 - (1 / (2 tau)) * sum_ij (Sigma Sigma')_ij * I_ij(tau)

        I_ij(tau) = int_0^tau phi_i(s) phi_j(s) ds
                  = [tau - phi_i(tau) - phi_j(tau) + phi_{i+j}(tau)] / (kappa_i kappa_j)

    The second term in a(tau) is the (negative) convexity adjustment.

    Returns
    -------
    a : (N,) intercepts,  b : (N, 3) factor loadings
    """
    tau = np.asarray(maturities, dtype=float)
    k = params.kappa_Q
    SS = params.SigmaSigmaT
    N = tau.size
    a = np.empty(N)
    b = np.empty((N, N_FACTORS))
    ksum = k[:, None] + k[None, :]
    kprod = k[:, None] * k[None, :]
    for n, t in enumerate(tau):
        ph = _phi(k, t)
        b[n] = ph / t
        I = (t - ph[:, None] - ph[None, :] + _phi(ksum, t)) / kprod
        a[n] = params.delta0 - 0.5 * np.sum(SS * I) / t
    return a, b


def ode_loadings(
    K: np.ndarray,
    k0: np.ndarray,
    SigmaSigmaT: np.ndarray,
    delta0: float,
    delta1: np.ndarray,
    maturities: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic yield loadings for drift  k0 - K x  by numerically integrating the ODEs.

    Used for the physical-measure affine benchmark (full K_P, non-zero mean) and
    as an independent check of the closed-form Q loadings.
    """
    tau = np.asarray(maturities, dtype=float)
    order = np.argsort(tau)
    t_sorted = tau[order]
    d = K.shape[0]

    def rhs(_t, z):
        B = z[1:]
        dB = -delta1 - K.T @ B
        dA = -delta0 + k0 @ B + 0.5 * B @ SigmaSigmaT @ B
        return np.concatenate([[dA], dB])

    sol = solve_ivp(rhs, (0.0, float(t_sorted[-1])), np.zeros(d + 1),
                    t_eval=t_sorted, method="DOP853", rtol=1e-11, atol=1e-14)
    if not sol.success:  # pragma: no cover
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    A_sorted = sol.y[0]
    B_sorted = sol.y[1:].T
    A = np.empty_like(A_sorted)
    B = np.empty_like(B_sorted)
    A[order] = A_sorted
    B[order] = B_sorted
    a = -A / tau
    b = -B / tau[:, None]
    return a, b


def q_loadings_ode(params: VasicekParams, maturities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Q loadings via the generic ODE solver (for testing the closed form)."""
    return ode_loadings(params.K_Q, np.zeros(N_FACTORS), params.SigmaSigmaT,
                        params.delta0, params.delta1, maturities)


def p_loadings(params: VasicekParams, maturities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Physical-measure affine yield loadings: the yield the model would produce if
    bonds were priced with the P dynamics, i.e. with zero market price of risk.

        drift = K_P theta_P - K_P x

    This is the 'no-risk-premium benchmark' used in the decomposition. It equals the
    average expected short rate under P minus a P-measure convexity term.
    """
    k0 = params.K_P @ params.theta_P
    return ode_loadings(params.K_P, k0, params.SigmaSigmaT,
                        params.delta0, params.delta1, maturities)


def eh_loadings(params: VasicekParams, maturities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Strict Expectations-Hypothesis component (reference series only):

        (1/tau) int_0^tau E^P_t[r_{t+s}] ds = a_EH(tau) + b_EH(tau)' x_t

    with  M(tau) = (K_P tau)^{-1} (I - e^{-K_P tau}),
          b_EH = M' iota,
          a_EH = delta0 + iota' (I - M) theta_P.
    """
    tau = np.asarray(maturities, dtype=float)
    K = params.K_P
    I = np.eye(N_FACTORS)
    iota = params.delta1
    a = np.empty(tau.size)
    b = np.empty((tau.size, N_FACTORS))
    for n, t in enumerate(tau):
        M = np.linalg.solve(K * t, I - expm(-K * t))
        b[n] = M.T @ iota
        a[n] = params.delta0 + iota @ (I - M) @ params.theta_P
    return a, b


def model_yields(a: np.ndarray, b: np.ndarray, states: np.ndarray) -> np.ndarray:
    """Yields (T, N) given loadings and a (T, 3) array of states."""
    return a[None, :] + states @ b.T
