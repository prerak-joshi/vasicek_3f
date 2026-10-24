"""
Parameter container, identification scheme and unconstrained transforms.

Model (continuous time, annualised units, yields in decimals)
-------------------------------------------------------------
Physical measure P:
    dx_t = K_P (theta_P - x_t) dt + Sigma dW^P_t

Pricing measure Q:
    dx_t = -K_Q x_t dt + Sigma dW^Q_t

Short rate:
    r_t = delta0 + iota' x_t

Identification (Joslin-Singleton-Zhu style canonical form for a Gaussian A0(3) model)
    * K_Q = diag(kappa_1, kappa_2, kappa_3) with 0 < kappa_1 < kappa_2 < kappa_3
    * theta_Q = 0          (state has zero long-run mean under Q)
    * delta1  = iota       (short rate loads one-for-one on each factor)
    * Sigma lower triangular with positive diagonal

Under this normalisation delta0 is the Q long-run mean of the short rate
(r_inf^Q). Everything about the P dynamics (K_P, theta_P) is left free:
K_P is a full 3x3 matrix, which is what allows a non-trivial term premium.

The implied market prices of risk (essentially affine, Duffee 2002) satisfy
    Sigma lambda_t = Sigma (lambda0 + lambda1 x_t)
                   = (K_P theta_P) + (K_Q - K_P) x_t
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

N_FACTORS = 3
_TRIL = np.tril_indices(N_FACTORS)
_DIAG_IN_TRIL = [k for k, (i, j) in enumerate(zip(*_TRIL)) if i == j]

# Scale factors so that every element of the optimiser vector is O(1).
_LEVEL_SCALE = 100.0  # delta0 and theta_P are optimised in percent


@dataclass
class VasicekParams:
    """Natural (economic) parameters of the 3-factor Vasicek model."""

    kappa_Q: np.ndarray  # (3,) ordered, positive  -> K_Q = diag(kappa_Q)
    delta0: float  # Q long-run short rate (decimal)
    Sigma: np.ndarray  # (3,3) lower triangular, positive diagonal
    K_P: np.ndarray  # (3,3) full P mean-reversion matrix
    theta_P: np.ndarray  # (3,) P long-run mean of the state
    sigma_e: np.ndarray  # (N,) measurement error std per maturity (decimal)

    # ------------------------------------------------------------------ helpers
    def __post_init__(self) -> None:
        self.kappa_Q = np.asarray(self.kappa_Q, dtype=float).reshape(N_FACTORS)
        self.Sigma = np.tril(np.asarray(self.Sigma, dtype=float))
        self.K_P = np.asarray(self.K_P, dtype=float).reshape(N_FACTORS, N_FACTORS)
        self.theta_P = np.asarray(self.theta_P, dtype=float).reshape(N_FACTORS)
        self.sigma_e = np.atleast_1d(np.asarray(self.sigma_e, dtype=float))
        self.delta0 = float(self.delta0)

    @property
    def K_Q(self) -> np.ndarray:
        return np.diag(self.kappa_Q)

    @property
    def delta1(self) -> np.ndarray:
        return np.ones(N_FACTORS)

    @property
    def SigmaSigmaT(self) -> np.ndarray:
        return self.Sigma @ self.Sigma.T

    def is_stationary_P(self) -> bool:
        """P dynamics are stationary iff every eigenvalue of K_P has positive real part."""
        return bool(np.all(np.real(np.linalg.eigvals(self.K_P)) > 1e-6))

    def market_price_of_risk(self) -> dict[str, np.ndarray]:
        """lambda0, lambda1 such that lambda_t = lambda0 + lambda1 x_t."""
        lam0 = np.linalg.solve(self.Sigma, self.K_P @ self.theta_P)
        lam1 = np.linalg.solve(self.Sigma, self.K_Q - self.K_P)
        return {"lambda0": lam0, "lambda1": lam1}

    def short_rate_long_run(self) -> dict[str, float]:
        """Long-run mean of the short rate under each measure."""
        return {
            "Q": self.delta0,  # theta_Q = 0
            "P": self.delta0 + float(self.delta1 @ self.theta_P),
        }

    def factor_half_lives(self) -> dict[str, np.ndarray]:
        """Half-lives (years) implied by the eigenvalues of K_Q and K_P."""
        eig_p = np.linalg.eigvals(self.K_P)
        return {
            "Q": np.log(2.0) / self.kappa_Q,
            "P": np.log(2.0) / np.real(eig_p),
        }

    def copy(self) -> VasicekParams:
        return VasicekParams(
            kappa_Q=self.kappa_Q.copy(),
            delta0=self.delta0,
            Sigma=self.Sigma.copy(),
            K_P=self.K_P.copy(),
            theta_P=self.theta_P.copy(),
            sigma_e=self.sigma_e.copy(),
        )

    def to_dict(self) -> dict[str, object]:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        mpr = self.market_price_of_risk()
        d["lambda0"] = mpr["lambda0"].tolist()
        d["lambda1"] = mpr["lambda1"].tolist()
        d["short_rate_long_run"] = self.short_rate_long_run()
        hl = self.factor_half_lives()
        d["half_life_Q_years"] = hl["Q"].tolist()
        d["half_life_P_years"] = hl["P"].tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> VasicekParams:
        keys = ("kappa_Q", "delta0", "Sigma", "K_P", "theta_P", "sigma_e")
        return cls(**{k: d[k] for k in keys})


# ---------------------------------------------------------------------------
# Unconstrained <-> natural transforms
# ---------------------------------------------------------------------------
@dataclass
class ParamTransform:
    """
    Maps an unconstrained real vector u to VasicekParams and back.

    Layout of u:
        [0:3]   log-increments of ordered kappa_Q:
                kappa_1 = kappa_min + exp(u0), kappa_2 = kappa_1 + exp(u1), kappa_3 = kappa_2 + exp(u2)
                (kappa_min >= 0 is an optional floor on Q mean reversion, default 0)
        [3]     100 * delta0
        [4:10]  lower triangle of Sigma (row-major tril order), diagonal entries in logs,
                everything scaled by 100
        [10:19] K_P (row-major, unconstrained; stationarity enforced in the likelihood)
        [19:22] 100 * theta_P
        [22:]   log(10000 * sigma_e)   (sigma_e in basis points)
    """

    n_maturities: int
    common_measurement_error: bool = False
    kappa_min: float = 0.0
    names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        names = ["log_dk1", "log_dk2", "log_dk3", "delta0_pct"]
        for i, j in zip(*_TRIL):
            names.append(f"Sigma_{i + 1}{j + 1}" + ("_log" if i == j else ""))
        for i in range(N_FACTORS):
            for j in range(N_FACTORS):
                names.append(f"KP_{i + 1}{j + 1}")
        names += [f"thetaP_{i + 1}_pct" for i in range(N_FACTORS)]
        n_err = 1 if self.common_measurement_error else self.n_maturities
        names += [f"log_sigma_e_bp_{k + 1}" for k in range(n_err)]
        self.names = names

    @property
    def size(self) -> int:
        return len(self.names)

    def to_params(self, u: np.ndarray) -> VasicekParams:
        u = np.asarray(u, dtype=float)
        inc = np.exp(np.clip(u[0:3], -12.0, 4.0))
        kappa_Q = self.kappa_min + np.cumsum(inc)
        delta0 = u[3] / _LEVEL_SCALE

        tril_vals = u[4:10].copy() / _LEVEL_SCALE
        for k in _DIAG_IN_TRIL:
            tril_vals[k] = np.exp(np.clip(u[4 + k], -15.0, 5.0)) / _LEVEL_SCALE
        Sigma = np.zeros((N_FACTORS, N_FACTORS))
        Sigma[_TRIL] = tril_vals

        K_P = u[10:19].reshape(N_FACTORS, N_FACTORS)
        theta_P = u[19:22] / _LEVEL_SCALE
        se = np.exp(np.clip(u[22:], -10.0, 8.0)) / 1e4
        if self.common_measurement_error:
            se = np.repeat(se, self.n_maturities)
        return VasicekParams(kappa_Q, delta0, Sigma, K_P, theta_P, se)

    def to_unconstrained(self, p: VasicekParams) -> np.ndarray:
        k = np.asarray(p.kappa_Q, dtype=float)
        if not np.all(np.diff(k) > 0) or k[0] <= 0:
            raise ValueError("kappa_Q must be positive and strictly increasing for identification")
        u = np.empty(self.size)
        u[0] = np.log(max(k[0] - self.kappa_min, 1e-6))
        u[1] = np.log(k[1] - k[0])
        u[2] = np.log(k[2] - k[1])
        u[3] = p.delta0 * _LEVEL_SCALE
        tril_vals = p.Sigma[_TRIL] * _LEVEL_SCALE
        for kk in _DIAG_IN_TRIL:
            tril_vals[kk] = np.log(max(p.Sigma[_TRIL][kk], 1e-12) * _LEVEL_SCALE)
        u[4:10] = tril_vals
        u[10:19] = p.K_P.reshape(-1)
        u[19:22] = p.theta_P * _LEVEL_SCALE
        se_bp = np.maximum(p.sigma_e, 1e-8) * 1e4
        if self.common_measurement_error:
            u[22] = np.log(np.mean(se_bp))
        else:
            u[22:] = np.log(se_bp)
        return u

    def natural_vector(self, p: VasicekParams) -> np.ndarray:
        """Flat vector of natural parameters (same ordering as `natural_names`)."""
        se = p.sigma_e[:1] if self.common_measurement_error else p.sigma_e
        return np.concatenate(
            [
                p.kappa_Q,
                [p.delta0],
                p.Sigma[_TRIL],
                p.K_P.reshape(-1),
                p.theta_P,
                se,
            ]
        )

    @property
    def natural_names(self) -> list[str]:
        names = [f"kappaQ_{i + 1}" for i in range(N_FACTORS)] + ["delta0"]
        names += [f"Sigma_{i + 1}{j + 1}" for i, j in zip(*_TRIL)]
        names += [f"KP_{i + 1}{j + 1}" for i in range(N_FACTORS) for j in range(N_FACTORS)]
        names += [f"thetaP_{i + 1}" for i in range(N_FACTORS)]
        n_err = 1 if self.common_measurement_error else self.n_maturities
        names += [f"sigma_e_{k + 1}" for k in range(n_err)]
        return names
