# FAQ

**Why three factors?**
Principal components of yield levels show that three factors (level, slope, curvature) explain over 99% of cross-sectional variation. The dashboard's PCA chart shows this for the loaded data.

**Why are the factors not exactly level, slope and curvature?**
Latent factors are identified only up to a normalisation. Here $K^Q$ is diagonal and ordered, $\theta^Q=0$, $\delta_1=\iota$ and $\Sigma$ is lower triangular, so the factors are a particular rotation of level/slope/curvature. The *Factors* tab reports how they correlate with the empirical versions.

**Why doesn't my term premium match ACM or Kim-Wright?**
Different models, different information. ACM is regression-based; Kim-Wright uses survey forecasts to anchor expected rates. All three agree more on *changes* than on *levels*. See [Interpreting the term premium](term-premium.md).

**Which term premium should I use, MLE or bias-corrected?**
Neither is "the" answer. The MLE is internally consistent; the bias-corrected version addresses the known downward bias in estimated persistence. The difference between them is a useful gauge of how uncertain the level is.

**Is the physical-measure affine yield the market's expected short rate?**
Approximately, under the model's P dynamics. It equals the model-implied average expected short rate minus a small convexity term (reported as `convexity_P`). It is not an observed expectation.

**Why does estimation take minutes?**
The likelihood is cheap (milliseconds), but there are 32 parameters and gradients are computed by finite differences. Standard errors need a 32×32 numerical Hessian (~2,000 likelihood evaluations), and bands and the out-of-sample test re-run the filter or the estimator many times. Use `--fast` for a quick look.

**Can I use other countries' yields?**
Yes, if you load a panel of continuously compounded zero-coupon yields (decimals, month-end, maturities in years) into a DataFrame and pass `y.values` to `estimate`. A loader for the Bank of England's nominal curve would be a welcome contribution.

**Why do the dashboard sliders not re-estimate the model?**
They are sensitivity analysis: they shock the estimated parameters and recompute outputs in milliseconds, reporting the likelihood cost. Re-estimation would let all other parameters adjust and takes minutes.

**The saved results don't load on my deployment.**
Results are stored as version-independent gzipped JSON (`bundle.json.gz`). If an old `bundle.pkl` fails to load, the dashboard runs a fresh demo estimation instead; re-run the pipeline to regenerate the bundle.
