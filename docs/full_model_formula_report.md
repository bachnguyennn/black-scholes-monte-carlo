# Full Quant Model Report

## 1. Purpose

This report explains, in a single flow, how the project moves from the simplest model (Black-Scholes) to the most advanced one (LSV), including:

- exact stochastic equations,
- simulation/pricing equations,
- calibration objectives,
- risk and validation metrics,
- and how each formula is used in this codebase.

The goal is to make every step mathematically and implementation-wise defensible.

## 2. Notation

- \(S_t\): underlying spot at time \(t\)
- \(K\): strike
- \(T\): maturity in years
- \(r\): continuously compounded risk-free rate
- \(q\): continuous dividend yield
- \(\sigma\): constant volatility (when assumed constant)
- \(V_t\): instantaneous variance process (stochastic-vol models)
- \(W_t\): Brownian motion
- \(N_t\): Poisson process with intensity \(\lambda\)
- \(\mathbb{Q}\): risk-neutral measure
- \(\Phi(\cdot)\): standard normal CDF

Core pricing identity used throughout:

$$
V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]
$$

For European options:

$$
\text{Call payoff}=(S_T-K)^+,\qquad \text{Put payoff}=(K-S_T)^+
$$

## 3. Model Ladder (from BS to latest)

### Step 0: Black-Scholes (baseline closed form)

Assume geometric Brownian motion with constant vol:

$$
dS_t=(r-q)S_t\,dt+\sigma S_t\,dW_t
$$

Closed-form call/put:

$$
C=S_0e^{-qT}\Phi(d_1)-Ke^{-rT}\Phi(d_2)
$$

$$
P=Ke^{-rT}\Phi(-d_2)-S_0e^{-qT}\Phi(-d_1)
$$

$$
d_1=\frac{\ln(S_0/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},\qquad d_2=d_1-\sigma\sqrt{T}
$$

What this gives:

- fast benchmark,
- sanity check for MC outputs,
- baseline "market proxy" in synthetic backtesting.

In code: `src/core/black_scholes.py`.

### Step 1: GBM Monte Carlo (same process, numerical expectation)

No model change yet. Only pricing method changes from closed form to simulation.

Exact terminal draw under GBM:

$$
S_T=S_0\exp\left((r-q-\tfrac12\sigma^2)T+\sigma\sqrt{T}Z\right),\quad Z\sim\mathcal N(0,1)
$$

MC estimator:

$$
\hat V_0=e^{-rT}\frac1N\sum_{i=1}^N \text{Payoff}(S_T^{(i)})
$$

Standard error estimator:

$$
\text{SE}(\hat V_0)=e^{-rT}\frac{\text{Std}[\text{Payoff}]}{\sqrt{N}}
$$

What changed vs BS: only numerical integration method.

In code: `src/core/gbm_engine.py`, `src/web/tabs/tab_option_analysis.py`.

### Step 2: Jump Diffusion (add discontinuous jump term)

Now we add jumps on top of diffusion.

Risk-neutral Merton jump-diffusion SDE:

$$
\frac{dS_t}{S_t}=(r-q-\lambda\kappa_J)dt+\sigma dW_t+(e^Y-1)dN_t
$$

with

$$
Y\sim\mathcal N(\mu_J,\sigma_J^2),\qquad \kappa_J=\mathbb E[e^Y-1]=e^{\mu_J+\frac12\sigma_J^2}-1
$$

The \(-\lambda\kappa_J\) term is the jump compensator that enforces risk-neutral drift consistency.

Discrete log update (per step \(\Delta t\)):

$$
\Delta\ln S=(r-q-\tfrac12\sigma^2-\lambda\kappa_J)\Delta t+\sigma\sqrt{\Delta t}Z+\sum_{m=1}^{N_{\Delta t}}Y_m
$$

where \(N_{\Delta t}\sim\text{Poisson}(\lambda\Delta t)\).

What changed vs GBM MC: rare, discontinuous jumps create skew/fat tails/crash behavior.

In code: `src/core/jump_diffusion.py`.

### Step 3: Heston (replace constant vol with stochastic variance)

Now we move from constant-vol diffusion to a two-factor diffusion.

$$
dS_t=(r-q)S_tdt+\sqrt{V_t}S_tdW_t^S
$$

$$
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}dW_t^V
$$

$$
dW_t^S dW_t^V=\rho\,dt
$$

Parameter roles:

- \(V_0\): initial variance,
- \(\kappa\): mean reversion speed,
- \(\theta\): long-run variance,
- \(\xi\): vol-of-vol,
- \(\rho\): spot-vol correlation (negative gives equity skew).

Feller condition (process regularity guideline):

$$
2\kappa\theta>\xi^2
$$

In this project, Heston is priced two ways:

- Monte Carlo path simulation,
- Fourier characteristic-function pricing (fast path for scanner/calibration).

What changed vs Jump Diffusion: jumps are not required; volatility itself becomes random and mean-reverting.

In code: `src/core/heston_model.py`.

### Step 4: LSV (latest model: local leverage on top of stochastic vol)

LSV extends Heston by multiplying stochastic variance by a local leverage surface \(L(S,t)\):

$$
dS_t=(r-q)S_tdt+L(S_t,t)\sqrt{V_t}S_tdW_t^S
$$

$$
dV_t=\kappa(\theta-V_t)dt+\xi\sqrt{V_t}dW_t^V,\qquad dW_t^S dW_t^V=\rho dt
$$

Interpretation:

- Heston gives dynamic stochastic-vol structure,
- \(L(S,t)\) adjusts local shape so the model can match observed smile/skew surface more closely.

What changed vs Heston: deterministic local correction field \(L\) added to diffusion loading.

In code: `src/core/lsv_model.py`, `src/core/calibration_engine.py`.

## 4. Simulation Schemes Used Here

### GBM

- Terminal simulation uses exact lognormal formula (not Euler).

### Jump Diffusion

- Simulates log-price increments with Poisson jump counts and Gaussian jump magnitudes.
- Includes risk-neutral jump compensator in drift.

### Heston

- Uses Euler-Maruyama with full truncation on variance to avoid \(\sqrt{V_t}\) issues when variance drifts negative numerically.
- Correlated shocks built by

$$
Z_V=\rho Z_S+\sqrt{1-\rho^2}\,Z_{\perp}
$$

### LSV

- Same Heston variance evolution,
- Asset diffusion scaled by \(L(S,t)\),
- In JIT simulation path, leverage lookup is nearest-neighbor on strike-maturity grid.

## 5. Pricing and Error Metrics

For all simulation-based valuations:

$$
\hat V_0=e^{-rT}\bar{X},\qquad \bar{X}=\frac1N\sum_{i=1}^N X_i,\quad X_i=\text{Payoff}(S_T^{(i)})
$$

Standard error:

$$
\widehat{\text{SE}}=e^{-rT}\frac{s_X}{\sqrt{N}}
$$

Approximate 95% confidence interval:

$$
\hat V_0\pm 1.96\,\widehat{\text{SE}}
$$

UI currently shows price and standard error; confidence bands can be derived immediately from these.

## 6. Calibration Math

### Heston calibration objective

Project objective (weighted SSE in IV-space):

$$
\text{SSE}(\Theta)=\sum_{i=1}^n w_i\big(\sigma^{mkt}_{IV,i}-\sigma^{hes}_{IV,i}(\Theta)\big)^2
$$

with \(\Theta=(\kappa,\theta,\xi,\rho,V_0)\).

Weights are ATM-centered (Gaussian in log-moneyness):

$$
w_i\propto \exp\left(-2\,[\ln(K_i/S_0)]^2\right)
$$

Normalized so \(\sum_i w_i=1\).

Calibration uses bounded SLSQP optimization.

### IV back-solve from model price

Given model price \(C\), implied vol is found by solving:

$$
C_{BS}(\sigma)=C
$$

via bisection over a bounded \(\sigma\)-interval.

### LSV leverage calibration (Dupire-to-LSV bridge)

Define total variance surface:

$$
w(k,T)=\sigma_{IV}(k,T)^2T
$$

with log-moneyness \(k=\ln(K/F)\).

Dupire local variance (Gatheral form):

$$
\sigma_{loc}^2=\frac{\partial_T w}{1-(k/w)\partial_k w + \tfrac14(-\tfrac14-1/w+k^2/w^2)(\partial_k w)^2 + \tfrac12\partial_{kk}w}
$$

Heston expected variance term:

$$
\mathbb E[V_t]=\theta+(V_0-\theta)e^{-\kappa t}
$$

Leverage surface:

$$
L(K,T)=\sqrt{\frac{\sigma_{loc}^2(K,T)}{\mathbb E[V_T]}}
$$

So that \(L^2\mathbb E[V]\) reproduces local variance by construction (subject to discretization and filtering).

## 7. Scanner Signal Formulas (Execution-aware)

For each contract with bid \(b\), ask \(a\), model price \(P_m\):

- effective ask: \(a_{eff}=a(1+\text{penalty})\)
- effective bid: \(b_{eff}=b(1-\text{penalty})\)

BUY edge:

$$
\text{edge}_{buy}=P_m-a_{eff}
$$

SELL edge:

$$
\text{edge}_{sell}=b_{eff}-P_m
$$

Decision:

- if \(\text{edge}_{buy}>0\): BUY
- else if \(\text{edge}_{sell}>0\): SELL
- else HOLD

Percent edge:

$$
\text{edge\%}_{buy}=100\cdot\frac{\text{edge}_{buy}}{a},\qquad
\text{edge\%}_{sell}=100\cdot\frac{\text{edge}_{sell}}{b}
$$

If HOLD, project reports midpoint-relative gap for ranking/diagnostics.

This is better than midpoint-only logic because it checks spread crossing plus liquidity penalties.

## 8. Backtester Core Formulas

### Entry edge gate

For synthetic mode:

$$
\text{edge}=\frac{P_{fair}-P_{mkt\,proxy}}{P_{mkt\,proxy}}
$$

Trade enters only if edge exceeds threshold.

### Delta hedge target

For long option position with contracts \(n\) and multiplier \(M\):

$$
\text{target shares}=-\Delta\cdot n\cdot M
$$

Rebalance only if delta/share gap exceeds threshold.

### Transaction costs

Option entry cost (bps on premium) and hedge turnover cost (bps on stock notional) are subtracted from net equity.

### Daily returns and Sharpe

Given net equity series \(E_t\):

$$
R_t=\frac{E_t}{E_{t-1}}-1
$$

$$
\text{Sharpe}=\frac{\bar{R}}{\text{Std}(R)}\sqrt{252}
$$

### Drawdown

$$
\text{DD}_t=100\cdot\frac{E_t-\max_{u\le t}E_u}{\max_{u\le t}E_u}
$$

$$
\text{MaxDD}=\min_t \text{DD}_t
$$

### Win rate

$$
\text{WinRate}=100\cdot\frac{\#\text{winning trades}}{\#\text{total trades}}
$$

## 9. Greeks Used in UI

Greeks are produced through a mixed approach:

- Pricing for MC models (`gbm`, `jump_diffusion`) is simulation-based.
- Delta/Vega/Gamma are from Black-Scholes analytical/JAX-autodiff path in current implementation.

Black-Scholes identities used:

$$
\Delta_{call}=\Phi(d_1),\quad \Delta_{put}=\Phi(d_1)-1
$$

$$
\Gamma=\frac{\phi(d_1)}{S_0\sigma\sqrt{T}}
$$

$$
\text{Vega}_{raw}=S_0\phi(d_1)\sqrt{T}
$$

Project reports vega per 1 vol point (dividing raw vega by 100).

## 10. How Models Are Used in This App

### Option Pricing tab

- Black-Scholes shown as benchmark.
- Second box now shows active model label (`GBM (MC)`, `Jump Diffusion (MC)`, `Heston (MC)`).
- Path visualization uses model-specific simulators.

### Scanner tab

- Heston route uses Fourier fast pricing.
- Jump Diffusion and LSV routes use simulation.
- Filtering, slippage penalties, and diagnostics improve signal auditability.

### Model Validation tab

- Compares model prices vs live quotes and reports residual patterns.

### Backtester tab

- Controlled research setup with no-look-ahead volatility,
- delta hedging and explicit cost accounting,
- transparent methodology disclosure.

## 11. Practical Progression Summary (one-line per step)

1. BS: constant-vol closed form.
2. GBM MC: same dynamics, numerical expectation instead of closed form.
3. Jump Diffusion: add Poisson jump term and compensator.
4. Heston: replace constant vol with stochastic mean-reverting variance.
5. LSV: multiply Heston diffusion by local leverage surface calibrated to smile.

## 12. Known Assumption Boundaries

- European payoff framework; early exercise is not modeled.
- Monte Carlo error remains finite and path-count dependent.
- Heston/LSV calibration quality depends on quote quality and filtering.
- LSV leverage grid/interpolation/discretization choices can materially affect results.
- Backtests are controlled research evidence, not execution-grade alpha proof.

## 13. File Map (where formulas live)

- Black-Scholes: `src/core/black_scholes.py`
- GBM MC: `src/core/gbm_engine.py`
- Jump Diffusion: `src/core/jump_diffusion.py`
- Heston MC + Fourier: `src/core/heston_model.py`
- LSV simulation and leverage calibration: `src/core/lsv_model.py`
- Heston/LSV calibration pipelines: `src/core/calibration_engine.py`
- Scanner edge logic: `src/core/scanner_engine.py`
- Backtester metrics and bookkeeping: `src/core/backtester.py`
- UI integration: `src/web/tabs/*.py`, `src/web/app.py`

## 14. Suggested Short Oral Script (for judges)

"We start from Black-Scholes as an analytical baseline, then switch to Monte Carlo under GBM to validate numerical convergence. Next we add jump risk with Merton jump diffusion to model discontinuities. Then we move to Heston so variance itself is stochastic and correlated with spot, which captures skew dynamics better. Finally we apply LSV, which multiplies stochastic variance by a calibrated local leverage surface to better match the full implied-vol surface. We evaluate outputs using execution-aware scanner edges, calibration SSE diagnostics, and backtest risk metrics like Sharpe and max drawdown with explicit methodological disclosure."

