# Interpreting the term premium

## What the number is

At each date and maturity the model splits the fitted yield into two affine pieces of the same latent state:

$$y^Q_t(\tau) = y^P_t(\tau) + TP_t(\tau).$$

- $y^Q$ prices bonds with the risk-neutral (Q) dynamics estimated from the cross-section.
- $y^P$ prices bonds as if investors required no compensation for risk, using the physical (P) dynamics estimated from the time series. It is close to the average expected short rate over the bond's life; the two differ only by a small P-convexity term.
- $TP$ is the compensation for bearing duration risk implied by the gap between the two sets of dynamics: $\Sigma\lambda_t = K^P\theta^P + (K^Q-K^P)x_t$.

## What to trust most

| Output | Reliability | Why |
|---|---|---|
| Fitted yields, Q parameters | high | pinned down by the cross-section; pricing errors are a few bp |
| **Changes** in the term premium | moderate–high | driven by well-measured states; robust across P estimates |
| **Level** of the term premium | low–moderate | depends on $K^P,\theta^P$, which are weakly identified |
| Long-horizon expected rates | low | extrapolates the weakly identified P dynamics |

Three tools in the project quantify the uncertainty in the level:

1. **MLE vs bias-corrected P.** If the two estimates differ by 50bp, the level is uncertain by at least that much.
2. **Parameter band.** The 90% band from the MLE's sampling distribution.
3. **Out-of-sample forecasts.** If the P dynamics cannot beat a random walk, their implied expected rates (and hence the term premium level) deserve caution.

## Reading the attribution chart

The *What moved the curve?* chart splits the change in each observed yield over a window into:

- **Expectations (P affine):** the change in the no-risk-premium component, i.e. the market's revised path of expected short rates *as filtered through the model's P dynamics*;
- **Term premium:** the change in compensation for duration risk;
- **Pricing error:** the part the three-factor model does not fit.

A rise in the 10-year yield that shows up mostly as term premium is consistent with higher uncertainty or supply-driven repricing; one that shows up mostly as expectations is consistent with a repricing of the policy path. The split inherits the model's identification limits.

## Comparing with ACM and Kim-Wright

- **ACM** (Adrian, Crump & Moench) estimates an affine model by a sequence of regressions on many yields; its P dynamics come from a VAR on yield principal components.
- **Kim-Wright** is a three-factor Gaussian model like this one, but adds survey forecasts of short rates, which anchors the P dynamics and typically produces a smoother, more stable expectations component.

Use the 12-month-change correlation as the main comparison statistic. Level differences of tens of basis points between published models are normal.

## Caveats

- **Weak P identification** (see above). Small-sample bias pushes estimated mean reversion up and the term premium's variability up; the bias correction addresses this only partially.
- **Gaussian dynamics:** constant volatility, possible negative yields, no lower bound on rates. The 2009–2015 and 2020–2021 periods near the lower bound are a known weakness of Gaussian models.
- **No survey or macroeconomic information.**
- **GSW inputs are themselves fitted curves** (Svensson), which smooths away some pricing errors.
- **Pseudo out-of-sample:** final data vintages, specification chosen with hindsight.
- **Not investment advice**, not an official estimate.
