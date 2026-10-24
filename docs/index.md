# vasicek3f

**A three-factor Gaussian Vasicek term structure model of government bond yields, estimated by Kalman-filter maximum likelihood, with a model-based term premium decomposition.**

[Live dashboard](https://vasicek3f-5ygmrt2fr8rvg97zfofzlq.streamlit.app){ .md-button .md-button--primary }
[Source on GitHub](https://github.com/prerak-joshi/vasicek_3f){ .md-button }

---

## What it does

- Models the full yield curve with three latent factors in the canonical Gaussian A0(3) form.
- Separates **pricing-measure (Q)** dynamics, identified by the cross-section of yields, from **physical-measure (P)** dynamics, identified by their time series.
- Estimates everything jointly by **exact maximum likelihood** using the Kalman filter, with standard errors.
- Decomposes yields into a pricing-measure fitted yield and a physical-measure affine yield, giving a **term premium** at every maturity and date.
- Reports a **bias-corrected** term premium, **parameter-uncertainty bands**, and an **out-of-sample** forecast test against a random walk.
- Compares results with the **ACM** (NY Fed) and **Kim-Wright** (Fed) term premium series.

$$
\underbrace{TP_t(\tau)}_{\text{term premium}} =
\underbrace{a^Q(\tau)+b^Q(\tau)'x_t}_{\text{pricing-measure fitted yield}} -
\underbrace{a^P(\tau)+b^P(\tau)'x_t}_{\text{physical-measure affine yield}}
$$

The physical-measure affine yield is a tractable no-risk-premium benchmark inside the model. It is **not** a strict Expectations-Hypothesis estimate; the two differ by a small P-measure convexity term, which is reported.

## Where to start

| If you want to… | Read |
|---|---|
| install and run it | [Getting started](getting-started.md) |
| understand the mathematics | [Methodology](methodology.md) |
| interpret the term premium output | [Interpreting the term premium](term-premium.md) |
| use the dashboard | [Dashboard guide](dashboard.md) |
| see how the code is organised | [Architecture](architecture.md) |
| check that it works | [Validation](validation.md) |
| call the functions from Python | [API reference](api.md) |

## Scope

This is an educational and research project for practitioners and students of fixed income modelling. It is not an official or production term premium measure and not investment advice. See [Interpreting the term premium](term-premium.md#caveats) for the model's limitations.
