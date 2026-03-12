# Mathematical Foundations And Why They Matter

This document explains the main mathematical ideas used in the project, what each model is trying to capture, and why those choices affect pricing, scanning, and backtesting outcomes.

## 1. Core Pricing Problem

An option price is the discounted expected payoff under a risk-neutral measure:

$$
V_0 = e^{-rT} \mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
$$

For a European call:

$$
\text{Payoff}(S_T) = \max(S_T - K, 0)
$$

For a European put:

$$
\text{Payoff}(S_T) = \max(K - S_T, 0)
$$

The entire project is about changing the model for how $S_T$ is generated or how its distribution is summarized.

## 2. Black-Scholes

### Model assumption

Black-Scholes assumes the asset follows geometric Brownian motion with constant volatility:

$$
dS_t = (r - q)S_t dt + \sigma S_t dW_t
$$

### Closed-form price

For a European call:

$$
C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

with

$$
d_1 = \frac{\ln(S_0/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma \sqrt{T}},
\quad
d_2 = d_1 - \sigma \sqrt{T}
$$

### Why it matters here

- It is the analytical benchmark used for sanity checks.
- It gives a stable proxy market price in the backtester.
- It provides a baseline against which richer models can be compared.

### Limitation

Constant volatility is too simple for real equity option markets. It cannot explain smiles, skews, or volatility clustering.

## 3. GBM Monte Carlo

GBM Monte Carlo simulates paths directly rather than using the Black-Scholes closed form.

Terminal price under GBM is:

$$
S_T = S_0 \exp\left((r - q - \tfrac{1}{2}\sigma^2)T + \sigma \sqrt{T} Z\right)
$$

where $Z \sim N(0,1)$.

### Why it matters here

- It demonstrates convergence to Black-Scholes.
- It provides price-path intuition for the UI.
- It is a stepping stone toward more complex Monte Carlo engines.

### Impact on project credibility

GBM alone is not impressive to a judge, but showing that your Monte Carlo converges to the analytical result proves numerical competence.

## 4. Heston Stochastic Volatility

### Model assumption

Heston allows variance to evolve randomly:

$$
dS_t = (r - q)S_t dt + \sqrt{V_t} S_t dW_t^S
$$

$$
dV_t = \kappa(\theta - V_t)dt + \xi \sqrt{V_t} dW_t^V
$$

$$
\text{Corr}(dW_t^S, dW_t^V) = \rho
$$

### Parameter meaning

- $V_0$: initial variance
- $\kappa$: mean-reversion speed
- $\theta$: long-run variance
- $\xi$: volatility of volatility
- $\rho$: spot-vol correlation

### Why it matters

Heston is important because real options markets are not flat in volatility. Negative spot-vol correlation produces skew, and random variance dynamics better reflect market behavior than constant $\sigma$.

### Feller condition

The variance process is better behaved when:

$$
2\kappa\theta > \xi^2
$$

This project explicitly checks that condition and surfaces the result in the UI. That matters because it shows awareness of model boundary conditions rather than blind simulation.

### Fourier pricing impact

Heston also supports semi-closed-form pricing through a characteristic function and numerical integration. That matters because:

- it is much faster than brute-force Monte Carlo for scanning many contracts
- it supports a more serious architecture story in the scanner
- it allows calibration workflows that would be too slow under pure Monte Carlo

## 5. Jump Diffusion

### Model assumption

Merton Jump Diffusion extends diffusion with Poisson jump arrivals:

$$
dS_t = (r - q - \lambda \kappa_J)S_t dt + \sigma S_t dW_t + (e^Y - 1)S_t dN_t
$$

where:

- $N_t$ is a Poisson process with intensity $\lambda$
- $Y \sim N(\mu_J, \sigma_J^2)$ is jump size in log space
- $\kappa_J = \mathbb{E}[e^Y - 1]$

### Why it matters

Diffusion-only models struggle with crash risk. Jump Diffusion adds discontinuous moves and creates heavier left tails.

### Practical impact

- OTM put prices can become more realistic than under GBM.
- Tail scenarios in the UI become more instructive.
- The project can discuss crash sensitivity and fat tails with more credibility.

### Risk-neutral compensator

The jump compensator term matters because without it the process can become risk-neutral inconsistent. That would distort expected discounted prices and undermine scan signals.

## 6. Local Stochastic Volatility

### Idea

LSV attempts to combine the realistic dynamics of stochastic volatility with a leverage function that helps fit the observed volatility surface.

In the project, the asset dynamics are adjusted by a leverage factor $L(S_t, t)$:

$$
dS_t = (r - q)S_t dt + L(S_t,t)\sqrt{V_t}S_t dW_t^S
$$

### Why it matters

Pure Heston may not fit every strike and maturity well enough. LSV is intended to bridge the gap between a stochastic-vol dynamics story and an observed surface-fitting story.

### Practical impact

- stronger research narrative about fitting the implied volatility surface
- more advanced scanner/calibration discussion
- better explanation for why model sophistication is not just cosmetic

### Limitation

This remains an advanced research component. It should be presented as exploratory and technically ambitious, not overclaimed as institutional-grade calibration infrastructure.

## 7. Calibration

### Objective

Calibration means choosing model parameters that reduce pricing error relative to observed market volatility or price data.

The Heston calibration objective is a weighted sum of squared errors:

$$
\text{SSE} = \sum_i w_i \left(IV_{market,i} - IV_{model,i}(\kappa,\theta,\xi,\rho,V_0)\right)^2
$$

### Why it matters

Without calibration, scan signals can simply be artifacts of bad parameter guesses. Calibration converts the project from a parameter demo into a market-fitting research tool.

### Impact on presentation

Calibration is one of the strongest pieces of evidence you can show a judge because it answers the question: "Did you fit the model to the market, or just choose numbers you liked?"

## 8. Greeks And Risk Sensitivities

Greeks describe how option value changes as inputs move.

### Main Greeks used here

- Delta: sensitivity to spot
- Gamma: sensitivity of Delta to spot
- Vega: sensitivity to volatility

In practical trading language:

- Delta shows directional exposure.
- Gamma shows convexity and hedge instability.
- Vega shows how strongly a contract responds to volatility repricing.

### Why it matters

Without Greeks, a price is just a number. Greeks turn the project into a risk-analysis tool.

### Impact on the backtester

The backtester’s delta-hedging logic matters because it attempts to separate volatility-driven PnL from pure directional drift. That does not make it a real execution-grade system, but it makes the experiment intellectually stronger.

## 9. Scanner Logic And Execution Realism

The scanner does not compare model value to the midpoint and call it an opportunity. It uses bid/ask-aware logic:

- buy case: model value must exceed ask-side execution cost
- sell case: model value must be below bid-side execution proceeds

### Why it matters

This improves realism because live trading must cross the spread. Midpoint-only arbitrage detection creates fake edges.

### Additional diagnostic impact

The scanner also reports:

- reason counts for filtered contracts
- signal counts
- sigma source counts

This matters because it makes the engine auditable. Auditability is a major trust signal in any quant presentation.

## 10. Backtester Methodology

The backtester is intentionally framed as a controlled research simulator.

### Strong points

- no-look-ahead volatility estimation
- explicit delta hedging
- transaction cost accounting
- cost sensitivity outputs
- methodology disclosure returned to the UI

### Critical limitation

It does not replay historical options quote data. Entry prices are proxied from Black-Scholes rather than historical option NBBO.

### Why the disclosure matters

This is one of the most important honesty upgrades in the project. Without that disclosure, the system could be mistaken for evidence of historical tradable alpha. With the disclosure, it becomes a more credible research instrument.

## 11. What The Math Means For Judges

The mathematical sophistication matters less than whether the assumptions are explicit and defensible.

The strongest message is:

- you understand baseline pricing
- you understand where baseline assumptions fail
- you implemented more advanced models to address those failures
- you added diagnostics and disclosures where realism still breaks down

That combination is much stronger than claiming the project is already a deployable trading engine.