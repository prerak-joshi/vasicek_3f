"""
Exact discretisation of the multivariate Ornstein-Uhlenbeck process under P.

    dx = K (theta - x) dt + Sigma dW

sampled every dt years gives the VAR(1)

    x_{t+dt} = c + F x_t + eta_t,   eta_t ~ N(0, Q)

    F = exp(-K dt)
    c = (I - F) theta
    Q = int_0^dt exp(-K s) Sigma Sigma' exp(-K' s) ds

Q is computed exactly with Van Loan's (1978) block matrix-exponential method,
which works for any K (not just diagonal or diagonalisable).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov


def van_loan(K: np.ndarray, SS: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Van Loan: for dx = A x dt + B dW, expm([[-A, BB'], [0, A']] dt) = [[., G], [0, H]]
    gives Phi = H' and Q = Phi G.  Here A = -K and BB' = SS.  Returns (F, Q).
    """
    d = K.shape[0]
    M = np.zeros((2 * d, 2 * d))
    M[:d, :d] = K
    M[:d, d:] = SS
    M[d:, d:] = -K.T
    E = expm(M * dt)
    F = E[d:, d:].T
    Q = F @ E[:d, d:]
    return F, 0.5 * (Q + Q.T)


def discretize_ou(K: np.ndarray, theta: np.ndarray, Sigma: np.ndarray,
                  dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (c, F, Q) of the exact VAR(1) representation."""
    F, Q = van_loan(K, Sigma @ Sigma.T, dt)
    c = (np.eye(K.shape[0]) - F) @ theta
    return c, F, Q


def diffusion_from_discrete_cov(K: np.ndarray, Q: np.ndarray, dt: float) -> np.ndarray:
    """
    Invert the map SS -> Q for a given K. Q is linear in SS, so we evaluate it on a
    basis of symmetric matrices and solve the resulting linear system. The result is
    projected onto the PSD cone and returned as a lower-triangular Cholesky factor.
    """
    d = K.shape[0]
    iu = np.triu_indices(d)
    cols = []
    for i, j in zip(*iu):
        E = np.zeros((d, d))
        E[i, j] = E[j, i] = 1.0
        cols.append(van_loan(K, E, dt)[1][iu])
    A = np.column_stack(cols)
    s = np.linalg.lstsq(A, Q[iu], rcond=None)[0]
    SS = np.zeros((d, d))
    SS[iu] = s
    SS = SS + SS.T - np.diag(np.diag(SS))
    w, V = np.linalg.eigh(SS)
    w = np.maximum(w, 1e-10 * max(w.max(), 1e-12))
    SS = (V * w) @ V.T
    return np.linalg.cholesky(0.5 * (SS + SS.T))


def unconditional_moments(theta: np.ndarray, F: np.ndarray, Q: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Stationary mean and covariance of the discrete VAR(1)."""
    P0 = solve_discrete_lyapunov(F, Q)
    return theta.copy(), 0.5 * (P0 + P0.T)
