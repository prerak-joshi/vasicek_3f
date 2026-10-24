"""
Linear Gaussian state-space model, Kalman filter and Rauch-Tung-Striebel smoother.

State (transition) equation, from the exact P-measure discretisation:
    x_t = c + F x_{t-1} + eta_t,         eta_t ~ N(0, Q)

Measurement equation, from the Q-measure affine pricing formula:
    y_t = a + H x_t + e_t,               e_t ~ N(0, R),  R = diag(sigma_e^2)

So the cross-section of yields identifies the Q parameters (through a, H), while
the time-series behaviour of the filtered states identifies the P parameters
(through c, F, Q). Both enter one Gaussian likelihood, evaluated by the
prediction-error decomposition:

    log L = -1/2 sum_t [ n_t log 2pi + log|S_t| + v_t' S_t^{-1} v_t ]

Missing observations (NaN) are handled by dropping the corresponding rows of the
measurement equation for that date.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

_LOG2PI = np.log(2.0 * np.pi)


@dataclass
class KalmanResult:
    loglik: float
    x_pred: np.ndarray  # (T, d)   x_{t|t-1}
    P_pred: np.ndarray  # (T, d, d)
    x_filt: np.ndarray  # (T, d)   x_{t|t}
    P_filt: np.ndarray  # (T, d, d)
    innovations: np.ndarray  # (T, N)   v_t (NaN where missing)
    innov_var_diag: np.ndarray  # (T, N) diag(S_t)
    loglik_t: np.ndarray  # (T,)


def kalman_filter(
    y: np.ndarray,
    a: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    c: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    store: bool = True,
) -> KalmanResult:
    """
    Run the Kalman filter.

    Parameters
    ----------
    y : (T, N) observed yields (decimal), may contain NaN
    a, H, R : measurement intercept (N,), loading (N, d), noise covariance (N, N)
    c, F, Q : transition intercept (d,), matrix (d, d), noise covariance (d, d)
    x0, P0  : prior mean / covariance for x_1 (usually the stationary moments)
    store   : keep full histories (False for faster likelihood-only evaluation)
    """
    T, N = y.shape
    d = F.shape[0]
    I_d = np.eye(d)
    obs_mask = ~np.isnan(y)
    full_obs = obs_mask.all(axis=1)

    if store:
        x_pred = np.empty((T, d))
        P_pred = np.empty((T, d, d))
        x_filt = np.empty((T, d))
        P_filt = np.empty((T, d, d))
        innov = np.full((T, N), np.nan)
        svar = np.full((T, N), np.nan)
    ll_t = np.zeros(T)

    x_p, P_p = x0.astype(float).copy(), P0.astype(float).copy()
    for t in range(T):
        if store:
            x_pred[t] = x_p
            P_pred[t] = P_p
        if full_obs[t]:
            idx = slice(None)
            n_t = N
        else:
            idx = np.where(obs_mask[t])[0]
            n_t = idx.size
        if n_t == 0:
            x_f, P_f = x_p, P_p
        else:
            Ht = H[idx]
            Rt = R if full_obs[t] else R[np.ix_(idx, idx)]
            vt = y[t, idx] - a[idx] - Ht @ x_p
            PHt = P_p @ Ht.T
            S = Ht @ PHt + Rt
            try:
                cf = cho_factor(S, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return _failed(T, N, d, store)
            Sinv_v = cho_solve(cf, vt, check_finite=False)
            logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
            ll_t[t] = -0.5 * (n_t * _LOG2PI + logdet + vt @ Sinv_v)
            Kg = cho_solve(cf, PHt.T, check_finite=False).T  # P H' S^{-1}
            x_f = x_p + Kg @ vt
            # Joseph form keeps P_f symmetric positive semi-definite
            IKH = I_d - Kg @ Ht
            P_f = IKH @ P_p @ IKH.T + Kg @ Rt @ Kg.T
            if store:
                innov[t, idx] = vt
                svar[t, idx] = np.diag(S)
        if store:
            x_filt[t] = x_f
            P_filt[t] = P_f
        x_p = c + F @ x_f
        P_p = F @ P_f @ F.T + Q
        P_p = 0.5 * (P_p + P_p.T)

    ll = float(ll_t.sum())
    if not np.isfinite(ll):
        return _failed(T, N, d, store)
    if store:
        return KalmanResult(ll, x_pred, P_pred, x_filt, P_filt, innov, svar, ll_t)
    return KalmanResult(ll, None, None, None, None, None, None, ll_t)  # type: ignore[arg-type]


def kalman_loglik_collapsed(
    y: np.ndarray,
    a: np.ndarray,
    H: np.ndarray,
    r_diag: np.ndarray,
    c: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    x0: np.ndarray,
    P0: np.ndarray,
    steady_tol: float = 1e-11,
) -> float:
    """
    Fast exact log-likelihood via the collapsed filter of Jungbacker & Koopman (2015).

    With N yields and only d = 3 states, the N-dimensional observation can be reduced
    to its GLS projection on the state space,

        y*_t = C^{-1} H' R^{-1} (y_t - a) = x_t + e*_t,   C = H' R^{-1} H,  Var(e*) = C^{-1},

    and the Kalman filter is run on the 3-dimensional y*_t. The likelihood of the
    discarded orthogonal part is available in closed form:

        log L = log L(y*) - T/2 [ (N-d) log 2pi + log|R| + log|C| ] - 1/2 sum_t u_t' R^{-1} u_t

    with u_t = y_t - a - H y*_t the GLS residual. Requires no missing values.
    """
    T, N = y.shape
    d = F.shape[0]
    Rinv = 1.0 / r_diag
    C = H.T @ (H * Rinv[:, None])
    try:
        cC = cho_factor(C, lower=True)
    except np.linalg.LinAlgError:
        return -np.inf
    Cinv = cho_solve(cC, np.eye(d))
    z = y - a[None, :]
    ystar = cho_solve(cC, (H * Rinv[:, None]).T @ z.T).T  # (T, d)
    u = z - ystar @ H.T
    logdetC = 2.0 * np.sum(np.log(np.diag(cC[0])))
    ll_orth = -0.5 * (
        T * ((N - d) * _LOG2PI + np.sum(np.log(r_diag)) + logdetC) + np.sum(u * u * Rinv[None, :])
    )

    ll = 0.0
    x_p, P_p = x0.astype(float).copy(), P0.astype(float).copy()
    t = 0
    while t < T:
        S = P_p + Cinv
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return -np.inf
        v = ystar[t] - x_p
        w = np.linalg.solve(L, v)
        Linv_P = np.linalg.solve(L, P_p)  # L^{-1} P
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        ll += -0.5 * (d * _LOG2PI + logdet + w @ w)
        x_f = x_p + Linv_P.T @ w  # P S^{-1} v
        P_f = P_p - Linv_P.T @ Linv_P  # P - P S^{-1} P
        x_p = c + F @ x_f
        P_new = F @ P_f @ F.T + Q
        P_new = 0.5 * (P_new + P_new.T)
        t += 1
        # Steady state: once the predicted covariance stops changing, the gain and
        # innovation covariance are constant, so only the mean recursion remains.
        if np.max(np.abs(P_new - P_p)) <= steady_tol * np.max(np.abs(P_p)):
            P_p = P_new
            break
        P_p = P_new
    if t < T:
        S = P_p + Cinv
        L = np.linalg.cholesky(S)
        Linv = np.linalg.solve(L, np.eye(d))
        Sinv = Linv.T @ Linv
        Kg = P_p @ Sinv
        A = F - F @ Kg  # x_{t+1|t} = c + A x_{t|t-1} + F K y*_t
        FK = F @ Kg
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        n_rem = T - t
        V = np.empty((n_rem, d))
        for s in range(n_rem):
            V[s] = ystar[t + s] - x_p
            x_p = c + A @ x_p + FK @ ystar[t + s]
        W = V @ Linv.T
        ll += -0.5 * (n_rem * (d * _LOG2PI + logdet) + np.sum(W * W))
    total = ll + ll_orth
    return float(total) if np.isfinite(total) else -np.inf


def _failed(T: int, N: int, d: int, store: bool) -> KalmanResult:
    return KalmanResult(-np.inf, None, None, None, None, None, None, np.full(T, -np.inf))  # type: ignore[arg-type]


def rts_smoother(kf: KalmanResult, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel fixed-interval smoother.

        J_t     = P_{t|t} F' P_{t+1|t}^{-1}
        x_{t|T} = x_{t|t} + J_t (x_{t+1|T} - x_{t+1|t})
        P_{t|T} = P_{t|t} + J_t (P_{t+1|T} - P_{t+1|t}) J_t'
    """
    T, d = kf.x_filt.shape
    xs = kf.x_filt.copy()
    Ps = kf.P_filt.copy()
    for t in range(T - 2, -1, -1):
        Pp = kf.P_pred[t + 1]
        J = np.linalg.solve(Pp.T, (kf.P_filt[t] @ F.T).T).T
        xs[t] = kf.x_filt[t] + J @ (xs[t + 1] - kf.x_pred[t + 1])
        Ps[t] = kf.P_filt[t] + J @ (Ps[t + 1] - Pp) @ J.T
        Ps[t] = 0.5 * (Ps[t] + Ps[t].T)
    return xs, Ps
