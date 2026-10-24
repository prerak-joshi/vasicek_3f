# Methodology

Notation: annualised time, continuously compounded yields in decimals, $x_t\in\mathbb R^3$, $\iota=(1,1,1)'$.

## 1. Model

**Physical measure P**

$$dx_t = K^P(\theta^P - x_t)\,dt + \Sigma\,dW^P_t$$

**Pricing measure Q**

$$dx_t = K^Q(\theta^Q - x_t)\,dt + \Sigma\,dW^Q_t$$

**Short rate** $r_t = \delta_0 + \delta_1' x_t$.

Volatility $\Sigma$ is the same under both measures (Girsanov changes drifts, not diffusions). The drifts are linked by the market price of risk $\lambda_t$:

$$K^P(\theta^P - x_t) - K^Q(\theta^Q - x_t) = \Sigma\lambda_t .$$

With affine drifts this is the *essentially affine* specification (Duffee 2002): $\lambda_t=\lambda_0+\lambda_1x_t$.

## 2. Identification (canonical form)

Latent factors can be rotated, shifted and rescaled without changing observable yields, so restrictions are needed. Following Dai-Singleton and Joslin-Singleton-Zhu:

| Restriction | Removes |
|---|---|
| $K^Q=\mathrm{diag}(\kappa_1,\kappa_2,\kappa_3)$, $0<\kappa_1<\kappa_2<\kappa_3$ | rotations and permutations |
| $\theta^Q = 0$ | shifts |
| $\delta_1=\iota$ | scaling |
| $\Sigma$ lower triangular, positive diagonal | remaining rotation of shocks |

$\delta_0$ is then the long-run Q mean of the short rate. $K^P$ (9 parameters) and $\theta^P$ (3) are unrestricted. With $\Sigma$ (6), $\kappa^Q$ (3), $\delta_0$ (1) and one measurement-error s.d. per maturity, the model has $22+N$ parameters (32 for ten maturities).

Implied risk prices: $\Sigma\lambda_0 = K^P\theta^P$ and $\Sigma\lambda_1 = K^Q-K^P$. A zero term premium at every date requires $K^P=K^Q$ and $\theta^P=0$.

## 3. Affine bond pricing

$P_t(\tau)=E^Q_t[e^{-\int_t^{t+\tau}r_s ds}]=\exp(A(\tau)+B(\tau)'x_t)$. For drift $k_0-Kx$, Feynman-Kac gives

$$B'(\tau) = -\delta_1 - K'B(\tau),\qquad A'(\tau) = -\delta_0 + k_0'B(\tau) + \tfrac12 B(\tau)'\Sigma\Sigma'B(\tau),$$

with $A(0)=0$, $B(0)=0$. Yields are $y_t(\tau) = a(\tau)+b(\tau)'x_t$ with $a=-A/\tau$, $b=-B/\tau$.

**Q closed form** ($K=K^Q$ diagonal, $k_0=0$, $\delta_1=\iota$). Let $\phi_i(s)=(1-e^{-\kappa_i s})/\kappa_i$. Then $B_i(\tau)=-\phi_i(\tau)$ and

$$b_i(\tau)=\frac{1-e^{-\kappa_i\tau}}{\kappa_i\tau},\qquad a(\tau) = \delta_0 - \frac{1}{2\tau}\sum_{i,j}(\Sigma\Sigma')_{ij} I_{ij}(\tau),$$

$$I_{ij}(\tau)=\int_0^\tau\phi_i\phi_j\,ds=\frac{\tau-\phi_i(\tau)-\phi_j(\tau)+\phi_{ij}(\tau)}{\kappa_i\kappa_j},\quad \phi_{ij}(\tau)=\frac{1-e^{-(\kappa_i+\kappa_j)\tau}}{\kappa_i+\kappa_j}.$$

The second term of $a(\tau)$ is the **convexity adjustment** (Jensen's inequality), which lowers long yields and grows with maturity and volatility. Implemented in `affine.q_loadings`; tested against the ODE.

**P loadings** use the same ODEs with $K=K^P$ (full) and $k_0=K^P\theta^P$, solved numerically (`affine.p_loadings`).

## 4. Term premium decomposition

$$TP_t(\tau) = \underbrace{a^Q(\tau)+b^Q(\tau)'x_t}_{y^Q_t(\tau)} - \underbrace{a^P(\tau)+b^P(\tau)'x_t}_{y^P_t(\tau)} = (a^Q-a^P) + (b^Q-b^P)'x_t .$$

The strict Expectations-Hypothesis component is the average expected short rate:

$$\frac1\tau\int_0^\tau E^P_t[r_{t+s}]ds = \delta_0+\iota'\theta^P+\iota'M(\tau)(x_t-\theta^P),\qquad M(\tau)=(K^P\tau)^{-1}(I-e^{-K^P\tau}).$$

Pricing under P adds a convexity term, so

$$y^P_t(\tau) = \text{EH}_t(\tau) - \text{convexity}^P(\tau).$$

The project uses $y^P$ as specified; the gap (`convexity_P` in the outputs) is typically a basis point or two at 10 years for these parameter values.

## 5. Exact discretisation

For monthly data ($\Delta=1/12$) the OU process is exactly

$$x_{t+\Delta} = c + Fx_t + \eta_t,\quad F=e^{-K^P\Delta},\quad c=(I-F)\theta^P,\quad \eta_t\sim N\!\Big(0,\int_0^\Delta e^{-K^Ps}\Sigma\Sigma'e^{-K^{P\prime}s}ds\Big).$$

The covariance integral is computed with **Van Loan's method**: $\exp\!\big(\begin{smallmatrix}K^P & \Sigma\Sigma'\\ 0 & -K^{P\prime}\end{smallmatrix}\big)\Delta$ yields both $F$ and $Q$ exactly for any $K^P$. Since $Q$ is linear in $\Sigma\Sigma'$, the map is inverted exactly for starting values (`discretize.diffusion_from_discrete_cov`).

## 6. State-space form and Kalman filter

$$y_t = a + Hx_t + e_t,\quad e_t\sim N(0,R),\ R=\mathrm{diag}(\sigma_e^2)\qquad\text{(measurement, from Q)}$$

$$x_t = c + Fx_{t-1} + \eta_t,\quad \eta_t\sim N(0,Q)\qquad\text{(transition, from P)}$$

Filter recursions: predict $x_{t|t-1}=c+Fx_{t-1|t-1}$, $P_{t|t-1}=FP_{t-1|t-1}F'+Q$; innovation $v_t=y_t-a-Hx_{t|t-1}$, $S_t=HP_{t|t-1}H'+R$; gain $K_t=P_{t|t-1}H'S_t^{-1}$; update $x_{t|t}=x_{t|t-1}+K_tv_t$ with the Joseph-form covariance update. Initialised at the stationary distribution (discrete Lyapunov equation). Rows with missing yields are dropped from the measurement equation for that date.

**Likelihood** (prediction-error decomposition): $\log L=-\frac12\sum_t\big[n_t\log2\pi+\log|S_t|+v_t'S_t^{-1}v_t\big]$.

**Speed-ups, both exact:**
- *Collapsed filter* (Jungbacker-Koopman 2015): with $C=H'R^{-1}H$, filter the 3-vector $y^*_t=C^{-1}H'R^{-1}(y_t-a)=x_t+e^*_t$, $\mathrm{Var}(e^*)=C^{-1}$, and add the closed-form likelihood of the orthogonal residual $u_t=y_t-a-Hy^*_t$: $-\frac T2[(N-3)\log2\pi+\log|R|+\log|C|]-\frac12\sum u_t'R^{-1}u_t$.
- *Steady state:* once $P_{t|t-1}$ converges (relative change < 1e-11), the gain is constant and only the mean recursion remains.

**RTS smoother:** $J_t=P_{t|t}F'P_{t+1|t}^{-1}$, $x_{t|T}=x_{t|t}+J_t(x_{t+1|T}-x_{t+1|t})$, $P_{t|T}=P_{t|t}+J_t(P_{t+1|T}-P_{t+1|t})J_t'$.

## 7. Estimation

**Unconstrained parameterisation** (`params.ParamTransform`): ordered $\kappa$ via cumulative sums of exponentials; log diagonal of $\Sigma$; log measurement errors in bp; levels in percent for scaling. Stationarity of $K^P$ is enforced by returning a penalty when an eigenvalue has a non-positive real part.

**Starting values.** (1) For given $\kappa^Q$, each date's state is an OLS regression of yields on $b(\tau)$; $\delta_0$ has the pooled closed form $\delta_0=\overline{\iota'M_b z_t}/\iota'M_b\iota$ with $M_b$ the residual maker and $z_t=y_t+\text{convexity}$; Nelder-Mead over $\kappa^Q$ from several starts. (2) OLS VAR(1) on those states; $K^P=-\log m(F)/\Delta$, $\theta^P=(I-F)^{-1}c$, $\Sigma$ from the exact Van Loan inversion. Iterated twice.

**MLE:** L-BFGS-B on the mean negative log-likelihood, then a BFGS polish (skipped in fast mode). Optional random restarts.

**Standard errors:** inverse of the numerical Hessian in the unconstrained space, mapped to natural parameters with the delta method. They are approximate; inspect the Hessian's positive-definiteness flag in `params.json`.

## 7a. Restrictions used on real data

Three degenerate solutions appear on US Treasury data that do not appear in simulation, and the pipeline applies the standard remedies automatically for real data (`--source gsw`); simulations are estimated unrestricted.

| Problem on real data | Remedy (default for real data) | Flag |
|---|---|---|
| Some maturities are priced almost exactly and their measurement-error variance collapses to zero, inflating the likelihood | one common measurement-error s.d. for all maturities (as in Duffee 2002, JSZ 2011) | `--common-error / --no-common-error` |
| The slowest Q factor drifts to $\kappa^Q_1\to 0$, making $\delta_0 = r^Q_\infty$ unidentified (estimates of 40%+) | floor $\kappa^Q_1 \ge 0.02$ (Q half-life ≤ ~35 years), built into the parameter transform | `--kappa-min` |
| The P dynamics find a unit root in the level factor (half-life of centuries), making $\theta^P$ and the term-premium level meaningless | smooth penalty when the slowest real eigenvalue of $K^P$ falls below 0.02 | `--kp-min` |

When a floor binds (the estimate sits on it), report it as such: the data alone cannot pin that parameter down, and the term-premium level is conditional on the restriction. Changes in the term premium are much less sensitive to these choices than its level.

## 8. Identification lessons

- The cross-section identifies $\kappa^Q$, $\delta_0$ and (through convexity and the factor dynamics) $\Sigma$ precisely.
- $K^P$ and $\theta^P$ come from the time series of highly persistent factors. With a few decades of data, persistence is estimated with large error and downward bias (too-fast mean reversion), the classic small-sample problem discussed by Bauer, Rudebusch & Wu (2012) and Kim & Orphanides (2012).
- Because $TP=y^Q-y^P$ and $y^P$ depends on $(K^P,\theta^P)$, the **level** of the term premium inherits that uncertainty, while its **time variation** is driven by the well-measured states. Survey forecasts (Kim-Wright) or bias correction are the standard remedies; neither is implemented here.

## 9. Small-sample bias correction of the P dynamics

Persistence estimated from a few decades of monthly data is biased towards faster mean reversion. Following the logic of Bauer, Rudebusch & Wu (2012), the project reports a second term premium computed with bias-corrected P dynamics (`bias_correction.py`):

1. Take the filtered states $x_{t|t}$ at the MLE and demean them with the sample mean $m$.
2. Fit $z_t = F z_{t-1} + u_t$ by OLS.
3. Residual bootstrap (Kilian 1998): simulate $B$ samples from $(\hat F, \hat u)$, re-estimate $F^*_b$, and set $\text{bias}=\bar F^*-\hat F$, $F_{bc}=\hat F-\text{bias}$. If the corrected matrix has a root on or outside the unit circle, the correction is shrunk in steps of 1% until it is stationary.
4. $K^P_{bc}=-\log m(F_{bc})/\Delta$ and $\theta^P_{bc}=m$; all Q parameters and $\Sigma$ stay at the MLE.

This is a transparent two-step adjustment of the P block only. It is not a joint re-estimation and it does not use the inverse-bootstrap (indirect inference) refinement of BRW. The dashboard lets you switch between the two P estimates; the gap between them is itself a useful measure of model uncertainty.

## 10. Parameter-uncertainty bands

The MLE's asymptotic covariance in the unconstrained space, $\widehat{\mathrm{Cov}}(\hat u)=H^{-1}$, is used to draw $u^{(s)}\sim N(\hat u, H^{-1})$ (Krinsky-Robb simulation). For each stationary draw the smoother is re-run and the decomposition recomputed; pointwise 5% and 95% quantiles of $TP_t(\tau)$ form the band (`uncertainty.py`). The bands capture parameter uncertainty, which dominates for term premia; state-filtering uncertainty given parameters is small and is not added. They are pointwise, not simultaneous, and inherit the limitations of the normal approximation.

## 11. Out-of-sample forecast evaluation

The P dynamics drive both expected short rates and the term premium, so their forecasting performance is a direct check on them (`forecast.py`):

- Expanding window starting with the first half of the sample; parameters re-estimated every 12 months, warm-started from the previous estimate.
- At each origin $t$: $E_t[y_{t+h}(\tau)] = a^Q(\tau)+b^Q(\tau)'\big(\theta^P+F^h(x_{t|t}-\theta^P)\big)$ using the filtered state, so no data after $t$ is used.
- Benchmark: random walk $y_t(\tau)$. Loss: squared error. Test: Diebold-Mariano with Newey-West variance ($h-1$ lags); negative statistics favour the model.

The exercise is pseudo out-of-sample: GSW yields are final data and the specification was chosen with the full sample available. Beating the random walk at 6–12 month horizons is hard for affine models on post-1990 data (Duffee 2002, 2011), so a ratio near or above one on real data is a common, honest result.
