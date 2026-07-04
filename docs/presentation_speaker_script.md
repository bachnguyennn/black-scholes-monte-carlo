# IMCS2 Presentation Speaker Script

This document is a full script companion for your slide deck. It is designed so you can:

1. Present clearly slide by slide.
2. Defend technical decisions under questioning.
3. Explain model math in plain language without losing rigor.

Use this as speaking notes, not a word-for-word reading script.

## Quick Positioning (before Slide 1)

Use this opening if judges ask for one-sentence context:

"This project is a quantitative research terminal for options pricing and validation. It compares Black-Scholes, GBM Monte Carlo, Jump Diffusion, Heston, and LSV under a transparent, execution-aware workflow with historical quote backtesting and explicit model-risk disclosure."

## Slide 1: Title

Suggested narration:

"This presentation is about distribution-based analysis of option pricing models. Instead of only comparing one price per model, we also compare the full shape of simulated outcomes. That gives a better view of risk, tail behavior, and model uncertainty."

Key message:

- You are evaluating both price fit and distribution behavior.

## Slide 2: Project Synopsis

Suggested narration:

"Black-Scholes is our baseline. It maps a few inputs to a deterministic option price. In contrast, our Monte Carlo engines simulate many random outcomes and estimate expected payoff numerically. We use Black-Scholes as a benchmark and then step up model complexity to capture real market effects like jumps, stochastic volatility, and smile/skew structure."

If asked "is BS stochastic or deterministic?":

- The underlying model assumption is stochastic.
- The BS pricing formula result is deterministic once inputs are fixed.

## Slide 3: Key Terms

Suggested narration:

"These terms define the evaluation language for the project: volatility and stochastic dynamics define the process, Greeks define sensitivity risk, and bid/ask/mid define execution reality. We avoid midpoint-only fantasy fills and include spread-aware diagnostics."

If asked about Greeks in your app:

- Pricing tab reports Delta, Gamma, Vega, Theta.
- Rho is a standard Greek concept but not the primary reported metric in this app UI.

## Slide 4: Objectives

Suggested narration:

"We evaluate models in two ways: static market-fit diagnostics and distribution analysis. Static fit tells us how close model prices are to observed quotes. Distribution analysis tells us what each model implies about risk and payoff shape."

Important update for accuracy:

- Your current validation workflow is live quote-based in app tabs, while your backtester uses historical quote CSV.

## Slide 5: Models Considered

Suggested narration:

"We organize models as a ladder of complexity:
Black-Scholes baseline, GBM Monte Carlo under the same assumptions, Jump Diffusion for discontinuities, Heston for stochastic variance, and LSV for local plus stochastic volatility surface fit."

Say this clearly:

- Monte Carlo is a numerical method.
- GBM/Heston/Jump/LSV are model assumptions.

## Slide 6: Why BS Looks Different

Suggested narration:

"Black-Scholes already integrates over uncertainty analytically, so it gives one closed-form price. GBM Monte Carlo estimates the same expectation by random sampling. That is why GBM gives path charts while BS does not."

Core formula you can cite:

\[
V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
\]

BS computes this analytically; MC estimates it statistically.

## Slide 7: BS-GBM Model

Suggested narration:

"GBM uses the same constant-vol assumptions as Black-Scholes but estimates price by simulation. This is useful as a numerical sanity check: with enough paths, GBM MC should approach the BS benchmark."

Implementation detail you can mention:

- Your GBM engine uses exact terminal lognormal draw, then discounted payoff mean.

## Slide 8: Jump Diffusion

Suggested narration:

"Jump Diffusion extends continuous GBM with rare Poisson jumps. This adds heavier tails and crash scenarios. It is especially useful when pure diffusion misses tail risk."

Risk-neutral form to quote:

\[
\frac{dS_t}{S_t}=(r-q-\lambda\kappa)dt+\sigma dW_t+(e^Y-1)dN_t
\]

where \(\kappa=\mathbb{E}[e^Y-1]\).

Why compensator matters:

- Without compensator, drift is not risk-neutral consistent.

## Slide 9: Heston

Suggested narration:

"Heston addresses the constant-vol limitation by making variance itself stochastic and mean-reverting. This captures volatility clustering and skew behavior better than constant-\(\sigma\) models."

Model equations:

\[
dS_t=(r-q)S_tdt+\sqrt{V_t}S_tdW_t^S
\]
\[
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}dW_t^V
\]

Technical point to show maturity:

- Mention Feller condition check: \(2\kappa\theta>\xi^2\).

## Slide 10: LSV

Suggested narration:

"LSV combines stochastic volatility dynamics with a local leverage surface. It is designed to fit observed implied-vol surfaces better while retaining stochastic-vol behavior."

Equation:

\[
dS_t=(r-q)S_tdt+L(S_t,t)\sqrt{V_t}S_tdW_t^S
\]

Accuracy note for your current code:

- LSV is now explicit in model selection and historical backtesting.
- If leverage surface is uncalibrated, fallback is \(L=1\), effectively reducing to Heston-like behavior.

## Slide 11: Key Findings

Suggested narration:

"Main empirical finding for this sample contract: GBM and Jump were close, Heston had wider dispersion, and LSV matched market mid most closely. The broader point is not one winner forever, but model sensitivity to assumptions and calibration state."

Say this to avoid overclaim:

- "These are contract- and regime-specific results, not universal rankings."

## Slide 12: Terminal Distribution Comparison

Suggested narration:

"Here we compare terminal price distribution shapes, not just means. Wider spreads imply higher uncertainty and different tail behavior. This is where model structure differences become visible."

Good judge-facing line:

- "Distribution diagnostics are the risk story behind a single option price."

## Slide 13: Payoff Distribution Comparison

Suggested narration:

"Payoff distributions show many near-zero outcomes with long right tails, which is typical for options. Different models change tail mass and therefore expected value and risk profile."

If asked "what drives payoff differences?":

- Tail probability and tail magnitude.
- Not only probability of finishing ITM.

## Slide 14: Monte Carlo Comparison Table

Suggested narration:

"This table is a contract-specific snapshot: option price, terminal mean/std, ITM probability, and mean payoff. It demonstrates why we evaluate both moments and pricing outputs together."

If asked "why similar mean S_T but different option price?":

- Option payoff is nonlinear.
- Variance and tail asymmetry can change expected payoff materially.

## Slide 15: Static Evaluation vs Backtesting

Suggested narration (updated to your current app):

"We separate quote-fit diagnostics from strategy PnL testing. Static validation asks how close fair value is to observed quotes. Historical backtesting asks what happens through time with hedging, costs, and risk constraints."

Important accuracy update:

- Backtester now runs historical quote mode only in UI.
- Synthetic proxy mode is removed from the main backtester tab workflow.

## Slide 16: Conclusion and Takeaways

Suggested narration:

"No single model dominates in every regime. Simpler models are faster and easier to interpret; richer models capture more market structure but require stronger calibration and diagnostics. Our project emphasizes transparency: explicit assumptions, execution-aware comparisons, and clear limits on what conclusions are valid."

Update your "Future Directions" line to avoid outdated claims:

Replace:

- "Full historical backtesting with hedging and PnL"

With:

- "Expand historical backtesting across more assets/regimes and compare stability of model rankings."

## Technical Q&A Bank (likely judge questions)

### Q1) Is Monte Carlo a model?

Answer:

"No. Monte Carlo is the numerical method used to estimate expectations. GBM, Jump Diffusion, Heston, and LSV are the stochastic models being simulated."

### Q2) If Black-Scholes is deterministic, why is it based on randomness?

Answer:

"The underlying process assumption is stochastic. The pricing solution is deterministic because the expectation has a closed form under those assumptions."

### Q3) Why use Heston if it is harder?

Answer:

"Because constant volatility is often unrealistic. Heston captures stochastic variance and skew dynamics, which improves realism for many surfaces."

### Q4) What does LSV add beyond Heston?

Answer:

"LSV adds a local leverage surface on top of stochastic variance, improving fit to strike-maturity structure of implied volatility."

### Q5) Is LSV always better?

Answer:

"No. It depends on calibration quality, data quality, and regime. More flexibility can overfit or become unstable with weak inputs."

### Q6) Why compare to bid/ask and not only mid?

Answer:

"Because executable trading crosses spreads. Midpoint-only comparisons can overstate opportunities."

### Q7) What is your biggest realism limit?

Answer:

"Data and execution realism. Even with historical quotes and hedging logic, this remains a controlled research simulator, not a production execution stack."

### Q8) Why can two models with similar mean \(S_T\) have different option prices?

Answer:

"Options depend on the whole distribution and nonlinear payoff. Tails and dispersion matter, not only mean."

### Q9) How do you hedge in backtests?

Answer:

"Daily delta rebalancing with transaction costs and reserve constraints. Hedge thresholds reduce overtrading noise."

### Q10) How do you avoid look-ahead bias?

Answer:

"Rolling volatility uses only information available before each entry date."

## Formula Cheat Sheet (for fast recall)

### Risk-neutral pricing identity

\[
V_0=e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
\]

### Black-Scholes call

\[
C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)
\]
\[
d_1=\frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\quad d_2=d_1-\sigma\sqrt{T}
\]

### GBM terminal draw

\[
S_T=S_0\exp\left((r-q-\tfrac12\sigma^2)T+\sigma\sqrt{T}Z\right)
\]

### Jump Diffusion

\[
\frac{dS_t}{S_t}=(r-q-\lambda\kappa)dt+\sigma dW_t+(e^Y-1)dN_t
\]

### Heston

\[
dS_t=(r-q)S_tdt+\sqrt{V_t}S_tdW_t^S,
\quad
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}dW_t^V
\]

### LSV

\[
dS_t=(r-q)S_tdt+L(S_t,t)\sqrt{V_t}S_tdW_t^S
\]

## 60-Second Closing Version (if time is cut)

"We benchmarked Black-Scholes, then moved to Monte Carlo GBM, Jump Diffusion, Heston, and LSV to compare both pricing and full distribution behavior. The main takeaway is model choice changes tail risk, dispersion, and price fit. We use execution-aware diagnostics and historical quote backtesting with hedging to avoid overclaiming. Results are regime-specific, so transparency around assumptions, calibration quality, and data limits is essential."

## Final Delivery Tips

1. Lead with assumptions before results.
2. Say "contract-specific" whenever showing a winner.
3. Emphasize "execution-aware" and "no-look-ahead" as credibility points.
4. If challenged on limitations, agree quickly and explain mitigations already in code.
5. Keep confidence high but avoid claiming production readiness.

