# 15-Minute Detailed Presentation Script

Use this as your speaking script for a 15-minute delivery. The timing below is approximate and includes natural pauses.

## 0:00 - 0:45 | Opening

"Good [morning/afternoon]. This project is a quantitative research terminal for options pricing and model evaluation.

The core question is: if we price the same option using different models, how do both the fair value and the full risk distribution change?

Instead of stopping at one number, we compare full simulated distributions, quote-fit diagnostics, and historical quote backtesting with hedging and transaction costs."

Transition:

"I’ll start from Black-Scholes as the baseline, then move up model complexity to GBM Monte Carlo, Jump Diffusion, Heston, and LSV."

## 0:45 - 2:15 | What the Project Does

"The platform has three major analytical layers:

1. Option pricing and path visualization.
2. Quote-based validation and scanner diagnostics.
3. Historical quote backtesting with delta hedging.

The design principle is transparency:

- assumptions are explicit,
- execution realism is considered with bid/ask-aware logic,
- and limitations are disclosed instead of hidden.

This is important because in quantitative work, credibility depends less on fancy equations and more on whether the methodology is auditable and honest."

Transition:

"Now I’ll define the model ladder from simplest to most flexible."

## 2:15 - 4:45 | Model Ladder (Conceptual)

### Black-Scholes

"Black-Scholes is the analytical baseline.
It assumes geometric Brownian motion with constant volatility and gives a closed-form European option price.

It is deterministic once inputs are fixed: spot, strike, maturity, rate, dividend, and volatility."

### GBM Monte Carlo

"GBM Monte Carlo keeps the same assumptions, but computes price numerically by simulation.

So this is not a new model assumption yet; it is a new numerical method.

This gives us two benefits:

- distribution and path intuition,
- and convergence checks against Black-Scholes."

### Jump Diffusion

"Jump Diffusion adds a Poisson jump term on top of continuous diffusion.

This captures discontinuous shocks, heavier tails, and crash-like events that constant-vol diffusion misses."

### Heston

"Heston makes variance itself stochastic and mean-reverting.

This is crucial for matching volatility skew and clustering behavior seen in real markets."

### LSV

"LSV combines stochastic volatility with a local leverage surface.

Intuition: Heston gives realistic stochastic dynamics; local leverage improves fit to strike/maturity smile structure.

So LSV is usually the most flexible, but also the hardest to calibrate robustly."

Transition:

"Now I’ll show the equations and how they map to implementation."

## 4:45 - 7:30 | Core Math + Implementation Mapping

### Risk-neutral pricing identity

"All pricing is rooted in:

\[
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
\]

Black-Scholes computes this expectation analytically under GBM assumptions.
Monte Carlo estimates it by sampling many terminal outcomes and averaging discounted payoffs."

### GBM simulation intuition

"In simulation form:

\[
S_T = S_0\exp\left((r-q-\tfrac12\sigma^2)T + \sigma\sqrt{T}Z\right)
\]

with \(Z \sim N(0,1)\).

Then estimate:

\[
\hat V_0 = e^{-rT}\frac{1}{N}\sum_{i=1}^{N}\text{Payoff}(S_T^{(i)})
\]

and standard error:

\[
\widehat{SE}=e^{-rT}\frac{s_{payoff}}{\sqrt{N}}
\]

So more paths reduce statistical noise, approximately at rate \(1/\sqrt{N}\)."

### Jump Diffusion structure

"Under risk-neutral form:

\[
\frac{dS_t}{S_t}=(r-q-\lambda\kappa)dt+\sigma dW_t+(e^Y-1)dN_t
\]

where the compensator \(-\lambda\kappa\) keeps drift risk-neutral consistent."

### Heston structure

"Heston uses:

\[
dS_t=(r-q)S_tdt+\sqrt{V_t}S_tdW_t^S
\]
\[
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}dW_t^V
\]

with correlated shocks.

This project also checks Feller condition for variance-process stability guidance."

### LSV structure

"LSV adds leverage surface \(L(S,t)\):

\[
dS_t=(r-q)S_tdt+L(S_t,t)\sqrt{V_t}S_tdW_t^S
\]

In practical terms, LSV calibration tries to improve fit to the observed implied-vol surface." 

Transition:

"Now I’ll move from theory to what we observed empirically."

## 7:30 - 10:00 | Results Interpretation

"For the sample short-dated SPX contract in the deck:

- GBM and Jump Diffusion were close,
- Heston showed wider dispersion,
- LSV was closest to market mid.

The key insight is not ‘one permanent winner.’
The key insight is model sensitivity:

- different assumptions move tail mass,
- tail mass changes expected nonlinear payoff,
- and that changes fair value."

Say explicitly:

"Similar mean terminal price does not imply similar option value, because options are nonlinear in \(S_T\)."

Distribution takeaway:

"We use histogram/CDF comparisons because a single price hides risk shape. A model can match mid price yet imply very different tail exposure."

Transition:

"Next, how we validate realism and avoid common quant pitfalls."

## 10:00 - 12:30 | Validation, Execution Realism, Backtesting

### Quote-fit and scanner realism

"Validation is quote-based: model prices are compared against observed quotes using metrics like MAE/RMSE and spread-aware diagnostics.

Scanner logic is execution-aware:

- buy edge compares model value versus ask-side crossing cost,
- sell edge compares bid-side proceeds versus model value.

This avoids midpoint-only overstatement of opportunities."

### Backtesting layer

"Backtesting in current app flow is historical quote mode.

It includes:

- no-look-ahead volatility estimation,
- daily delta hedging,
- transaction costs,
- and solvency/risk guardrails.

This is presented as controlled research evidence, not a production execution claim."

### LSV in backtest

"LSV is available in historical backtesting.
If calibrated leverage surface is present, it is used.
If not, fallback uses \(L=1\), which is a safe, transparent degradation."

Transition:

"I’ll close with limitations and what this means for model selection decisions."

## 12:30 - 14:00 | Limitations and Decision Framework

"Main limitations:

- data quality and quote coverage constraints,
- calibration stability sensitivity,
- Monte Carlo error for finite path counts,
- and no claim of execution-grade deployment.

Model selection should be regime-dependent:

- GBM for baseline speed and intuition,
- Jump for discontinuity sensitivity,
- Heston for stochastic-vol/skew structure,
- LSV for best surface fit when calibration quality is strong."

## 14:00 - 15:00 | Closing

"Conclusion:

This project shows that model choice affects not only point estimates but full distributional risk.

The contribution is not just implementing multiple models; it is connecting them to transparent diagnostics, execution-aware comparisons, and historically grounded hedged evaluation.

So the most defensible output is not ‘this model always wins,’ but: under this dataset and assumptions, here is the fit-risk tradeoff, here are the limits, and here is why the conclusion is credible." 

---

# Rapid Q&A (Use After 15-Min Script)

## Q1: Is Monte Carlo the stochastic part?

"No. The stochastic part is in the model terms like \(dW_t\), \(dN_t\), and stochastic variance \(V_t\). Monte Carlo is the numerical sampling method used to estimate expectations from those stochastic models."

## Q2: Why does BS give one number while GBM gives many paths?

"BS integrates randomness analytically into a closed-form expectation. GBM Monte Carlo estimates the same expectation by random sampling, so it naturally produces path distributions."

## Q3: Why can Heston and LSV prices differ a lot?

"LSV modifies the diffusion loading with a leverage surface to better fit smile/skew structure. If that surface differs meaningfully from 1, tails and payoff expectation can shift materially."

## Q4: Does LSV always beat Heston?

"No. LSV can improve fit, but only if calibration quality is good. With noisy or sparse data it can be unstable or overfit."

## Q5: How do you address overclaiming risk?

"By using execution-aware diagnostics, explicit data limitations, and controlled-research wording instead of claiming live tradable alpha."

## Q6: What would you improve next?

"Cross-regime robustness study, calibration stability stress tests, and deeper historical quote coverage across assets/maturities."

---

# Delivery Notes

1. Keep pace around 130-150 words/minute.
2. Use pauses after each transition sentence.
3. If interrupted, jump directly to the nearest heading and continue.
4. Re-anchor to the core message: "distribution + transparency + execution realism."

