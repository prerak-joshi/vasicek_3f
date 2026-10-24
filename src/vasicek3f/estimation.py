"""
Estimation: two-stage starting values, then full Kalman-filter maximum likelihood.

Stage 1 (cross-section, "Q step")
    For trial kappa_Q, the Q loadings b(tau) are known in closed form. At every date
    the latent state is then an OLS regression of yields on b(tau), and delta0 has a
    closed-form pooled solution. We search over kappa_Q (3 numbers) to minimise the
    total squared pricing error. This is fast and robust.

Stage 2 (time series, "P step")
    The cross-sectional states are treated as observed and a VAR(1) is fitted.
    K_P = -logm(F)/dt, theta_P = (I-F)^{-1} c, and Sigma is recovered by inverting the
    exact Van Loan discrete covariance map. Stages 1-2 are iterated once so that the
    convexity term uses the updated Sigma.

Stage 3 (joint MLE)
    All parameters are re-estimated jointly by maximising the exact Kalman-filter
    likelihood in an unconstrained parameterisation (see params.ParamTransform).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import logm
from scipy.optimize import minimize

from .affine import q_loadings
from .discretize import diffusion_from_discrete_cov
from .model import loglik
from .params import N_FACTORS, ParamTransform, VasicekParams

# ---------------------------------------------------------------------------
# Stage 1: cross-sectional fit
# ---------------------------------------------------------------------------
KAPPA_MIN = 0.02  # lower bound on kappa_1 for starting values (half-life <= ~35 years)
KAPPA_RATIO = 1.5  # minimum ratio between successive kappas for starting values


@dataclass
class CrossSectionFit:
    kappa_Q: np.ndarray
    delta0: float
    states: np.ndarray  # (T, 3)
    residuals: np.ndarray  # (T, N)
    sse: float


def cross_section_given_kappa(
    y: np.ndarray, tau: np.ndarray, kappa: np.ndarray, SS: np.ndarray, delta0_prior: float = 0.05
) -> CrossSectionFit:
    """Closed-form delta0 and per-date OLS states for fixed kappa_Q and SigmaSigma'."""
    Sigma = np.linalg.cholesky(SS + 1e-14 * np.eye(N_FACTORS))
    tmp = VasicekParams(kappa, 0.0, Sigma, np.eye(3), np.zeros(3), np.ones(tau.size))
    a0, b = q_loadings(tmp, tau)  # a0 = minus convexity, delta0 = 0
    btb_inv_bt = np.linalg.solve(b.T @ b, b.T)
    Mb = np.eye(tau.size) - b @ btb_inv_bt
    iota = np.ones(tau.size)
    z = y - a0[None, :]
    num = np.mean(z @ Mb @ iota)
    den = iota @ Mb @ iota
    ridge = 1e-8  # guards near-collinearity of b_1 and iota
    delta0 = (num + ridge * delta0_prior) / (den + ridge)
    X = (z - delta0) @ btb_inv_bt.T
    resid = z - delta0 - X @ b.T
    return CrossSectionFit(kappa, float(delta0), X, resid, float(np.sum(resid**2)))


def fit_kappa(
    y: np.ndarray, tau: np.ndarray, SS: np.ndarray, starts: Optional[Sequence[Sequence[float]]] = None
) -> CrossSectionFit:
    """Search over ordered kappa_Q minimising cross-sectional SSE (multi-start Nelder-Mead)."""
    if starts is None:
        starts = [(0.04, 0.4, 1.5), (0.08, 0.7, 2.5), (0.15, 1.0, 4.0), (0.03, 0.25, 0.9)]
    prior = float(np.mean(y[:, -1]))

    # On real data the cross-section alone barely separates a near-unit-root level factor
    # (kappa_1 -> 0) from delta0, and two factors can collapse onto the same kappa. Starting
    # values therefore keep kappa_1 >= KAPPA_MIN and successive kappas at least RATIO apart;
    # the subsequent MLE is unrestricted apart from the ordering.
    def unpack(v):
        v = np.clip(v, -10, 3)
        k1 = KAPPA_MIN + np.exp(v[0])
        k2 = k1 * (KAPPA_RATIO + np.exp(v[1]))
        k3 = k2 * (KAPPA_RATIO + np.exp(v[2]))
        return np.array([k1, k2, k3])

    def pack(k):
        return np.log(
            [
                max(k[0] - KAPPA_MIN, 1e-4),
                max(k[1] / k[0] - KAPPA_RATIO, 1e-4),
                max(k[2] / k[1] - KAPPA_RATIO, 1e-4),
            ]
        )

    def obj(v):
        try:
            return cross_section_given_kappa(y, tau, unpack(v), SS, prior).sse
        except np.linalg.LinAlgError:
            return 1e6

    best = None
    for s in starts:
        v0 = pack(np.asarray(s, float))
        r = minimize(obj, v0, method="Nelder-Mead", options={"xatol": 1e-7, "fatol": 1e-14, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    return cross_section_given_kappa(y, tau, unpack(best.x), SS, prior)


# ---------------------------------------------------------------------------
# Stage 2: VAR(1) on states -> continuous-time P parameters
# ---------------------------------------------------------------------------
def fit_var_to_ou(X: np.ndarray, dt: float, max_root: float = 0.998):
    """OLS VAR(1) on states, mapped to (K_P, theta_P, Sigma)."""
    Y, Z = X[1:], X[:-1]
    Zc = np.column_stack([np.ones(len(Z)), Z])
    B = np.linalg.lstsq(Zc, Y, rcond=None)[0]
    c, F = B[0], B[1:].T
    roots = np.abs(np.linalg.eigvals(F))
    if roots.max() >= max_root:  # enforce stationarity for a usable start
        F = F * (max_root / roots.max())
        c = Y.mean(0) - F @ Z.mean(0)
    K = -np.real(logm(F)) / dt
    if np.any(np.real(np.linalg.eigvals(K)) <= 1e-3):
        K = K + (1e-2 - min(np.real(np.linalg.eigvals(K)).min(), 0.0)) * np.eye(N_FACTORS)
    theta = np.linalg.solve(np.eye(N_FACTORS) - F, c)
    U = Y - Zc @ np.vstack([c, F.T])
    Qhat = np.cov(U.T)
    Sigma = diffusion_from_discrete_cov(K, Qhat, dt)
    return K, theta, Sigma


def starting_values(
    y: np.ndarray, tau: np.ndarray, dt: float, n_iter: int = 2, verbose: bool = False
) -> VasicekParams:
    """Two-stage (cross-section + VAR) starting values for the MLE."""
    rows = ~np.isnan(y).any(axis=1)
    yc = y[rows]
    SS = np.diag([1e-4, 1e-4, 1e-4])
    for it in range(n_iter):
        cs = fit_kappa(yc, tau, SS)
        K_P, theta_P, Sigma = fit_var_to_ou(cs.states, dt)
        SS = Sigma @ Sigma.T
        if verbose:
            rmse = np.sqrt(np.mean(cs.residuals**2)) * 1e4
            print(
                f"  [start {it + 1}] kappa_Q={np.round(cs.kappa_Q, 4)} "
                f"delta0={cs.delta0:.4f} cross-section RMSE={rmse:.2f}bp"
            )
    sigma_e = np.maximum(np.std(cs.residuals, axis=0), 2e-5)
    return VasicekParams(cs.kappa_Q, cs.delta0, Sigma, K_P, theta_P, sigma_e)


# ---------------------------------------------------------------------------
# Stage 3: maximum likelihood
# ---------------------------------------------------------------------------
@dataclass
class EstimationResult:
    params: VasicekParams
    start_params: VasicekParams
    loglik: float
    n_obs: int
    n_params: int
    success: bool
    message: str
    n_iter: int
    elapsed_sec: float
    u_hat: np.ndarray
    transform: ParamTransform
    std_errors: Optional[pd.DataFrame] = None
    cov_u: Optional[np.ndarray] = None  # asymptotic covariance of u_hat (unconstrained space)
    history: list[float] = field(default_factory=list)

    @property
    def aic(self) -> float:
        return -2 * self.loglik + 2 * self.n_params

    @property
    def bic(self) -> float:
        return -2 * self.loglik + np.log(self.n_obs) * self.n_params


def make_objective(
    y: np.ndarray, tau: np.ndarray, dt: float, tr: ParamTransform, kp_min: float = 0.0
) -> Callable[[np.ndarray], float]:
    """
    Mean negative log-likelihood per observation in the unconstrained space.

    ``kp_min`` > 0 adds a smooth penalty when the slowest P mean-reversion rate (smallest
    real part of the eigenvalues of K_P) falls below ``kp_min``. On real data the MLE
    otherwise tends to a unit root in the level factor, which makes the long-run P mean,
    and therefore the level of the term premium, meaningless.
    """
    n_obs = int(np.sum(~np.isnan(y)))

    def obj(u: np.ndarray) -> float:
        p = tr.to_params(u)
        ll = loglik(p, y, tau, dt)
        if not np.isfinite(ll):
            return 1e3
        pen = 0.0
        if kp_min > 0:
            gap = kp_min - float(np.min(np.real(np.linalg.eigvals(p.K_P))))
            if gap > 0:
                pen = 1e4 * (gap / kp_min) ** 2
        return (-ll + pen) / n_obs

    return obj


def estimate(
    y: np.ndarray,
    tau: np.ndarray,
    dt: float,
    start: Optional[VasicekParams] = None,
    common_measurement_error: bool = False,
    maxiter: int = 4000,
    n_restarts: int = 1,
    seed: int = 0,
    compute_se: bool = True,
    kappa_min: float = 0.0,
    kp_min: float = 0.0,
    polish: bool = True,
    ftol: float = 1e-14,
    verbose: bool = True,
) -> EstimationResult:
    """
    Maximum likelihood estimation of the 3-factor Vasicek model.

    Parameters
    ----------
    y   : (T, N) yields in decimals (NaN allowed)
    tau : (N,) maturities in years
    dt  : sampling interval in years (1/12 for monthly)
    n_restarts : extra randomly perturbed starts (best likelihood is kept)
    kappa_min  : floor on the slowest Q mean-reversion rate (0 = unrestricted)
    kp_min     : soft floor on the slowest P mean-reversion rate (0 = unrestricted)
    polish     : follow L-BFGS-B with a BFGS polish step (off in fast mode)
    ftol       : L-BFGS-B relative function tolerance (looser in fast mode)
    """
    t0 = time.time()
    tau = np.asarray(tau, float)
    tr = ParamTransform(tau.size, common_measurement_error, kappa_min=kappa_min)
    if start is None:
        if verbose:
            print("Stage 1-2: cross-sectional fit + VAR starting values")
        start = starting_values(y, tau, dt, verbose=verbose)
    obj = make_objective(y, tau, dt, tr, kp_min=kp_min)
    n_obs = int(np.sum(~np.isnan(y)))

    if start.kappa_Q[0] < 1.25 * kappa_min:  # move the start off the floor
        start = start.copy()
        start.kappa_Q = start.kappa_Q + (1.25 * kappa_min - start.kappa_Q[0])
    u0 = tr.to_unconstrained(start)
    rng = np.random.default_rng(seed)
    starts = [u0] + [u0 + rng.normal(scale=0.15, size=u0.size) for _ in range(n_restarts - 1)]

    best = None
    history: list[float] = []
    for k, us in enumerate(starts):
        if verbose:
            print(f"Stage 3: MLE run {k + 1}/{len(starts)}  start loglik={-obj(us) * n_obs:,.1f}")
        r = minimize(
            obj,
            us,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "maxfun": maxiter * 40, "ftol": ftol, "gtol": 1e-8, "maxcor": 30},
        )
        cand, cand_nit = r, r.nit
        if polish:
            r2 = minimize(obj, r.x, method="BFGS", options={"maxiter": 400, "gtol": 1e-7})
            if r2.fun < r.fun:
                cand, cand_nit = r2, r.nit + r2.nit
        history.append(-cand.fun * n_obs)
        if verbose:
            print(f"         -> loglik={-cand.fun * n_obs:,.2f}  ({r.message})")
        if best is None or cand.fun < best[0].fun:
            best = (cand, cand_nit, r.message)

    res, nit, msg = best
    params = tr.to_params(res.x)
    out = EstimationResult(
        params=params,
        start_params=start,
        loglik=-res.fun * n_obs,
        n_obs=n_obs,
        n_params=tr.size,
        success=bool(np.isfinite(res.fun) and res.fun < 1e3),
        message=str(msg),
        n_iter=int(nit),
        elapsed_sec=0.0,
        u_hat=res.x,
        transform=tr,
        history=history,
    )
    if compute_se:
        if verbose:
            print("Computing numerical Hessian for standard errors ...")
        out.std_errors, out.cov_u = standard_errors(out, y, tau, dt)
    out.elapsed_sec = time.time() - t0
    return out


# ---------------------------------------------------------------------------
# Standard errors (inverse Hessian + delta method)
# ---------------------------------------------------------------------------
def numerical_hessian(f: Callable[[np.ndarray], float], x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    n = x.size
    h = rel_step * np.maximum(1.0, np.abs(x))
    H = np.zeros((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4 * h[i] * h[j]
            )
    return H


def standard_errors(
    res: EstimationResult, y: np.ndarray, tau: np.ndarray, dt: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Inverse-Hessian standard errors mapped to natural parameters by the delta method.

    Returns the table of estimates/SEs/t-stats and the covariance matrix of the
    unconstrained estimates (used for parameter-uncertainty bands).
    """
    tr = res.transform
    n_obs = res.n_obs
    base = make_objective(y, tau, dt, tr)

    def negll(u):
        return base(u) * n_obs

    H = numerical_hessian(negll, res.u_hat)
    eig = np.linalg.eigvalsh(0.5 * (H + H.T))
    cov_u = np.linalg.pinv(0.5 * (H + H.T))

    # Jacobian of natural parameters w.r.t. u
    g0 = tr.natural_vector(tr.to_params(res.u_hat))
    J = np.zeros((g0.size, res.u_hat.size))
    for k in range(res.u_hat.size):
        step = 1e-6 * max(1.0, abs(res.u_hat[k]))
        up = res.u_hat.copy()
        up[k] += step
        dn = res.u_hat.copy()
        dn[k] -= step
        J[:, k] = (tr.natural_vector(tr.to_params(up)) - tr.natural_vector(tr.to_params(dn))) / (2 * step)
    cov = J @ cov_u @ J.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    df = pd.DataFrame({"estimate": g0, "std_error": se}, index=tr.natural_names)
    df["t_stat"] = df["estimate"] / df["std_error"].replace(0, np.nan)
    df.attrs["hessian_min_eig"] = float(eig.min())
    df.attrs["hessian_pd"] = bool(eig.min() > 0)
    return df, cov_u
