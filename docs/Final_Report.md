# Monte Carlo European Option Pricing Engine: A Multi-Model Quantitative Finance Platform

**Course:** IMCS 3020U — Integrated Application Project II

**Institution:** Ontario Tech University, Faculty of Science

**Date:** April 25, 2026

---

## Abstract

This report presents the design, implementation, and evaluation of a European option pricing platform built around five related valuation models: the Black-Scholes-Merton (BSM) analytical baseline, Geometric Brownian Motion (GBM) Monte Carlo, Heston Stochastic Volatility, Merton Jump Diffusion, and Local Stochastic Volatility (LSV). The system focuses exclusively on the pricing of European-style options—contracts that may only be exercised at expiration—which admit the closed-form and Monte Carlo solutions central to this work. The system demonstrates the synthesis of mathematical modelling and computer science by implementing each model's stochastic differential equations or analytical pricing formulas using vectorized numerical methods, wrapping them in an interactive Streamlit dashboard with a FastAPI backend, and validating them through convergence testing, calibration to live market data, and a delta-hedged historical backtesting framework. Live market data is sourced from Yahoo Finance via the yfinance library and the Polygon.io REST API, while historical backtesting uses end-of-day S&P 500 Index (SPX) option quotes from the Chicago Board Options Exchange (CBOE). The project illustrates how each successive model addresses specific limitations of its predecessor—constant volatility, absence of crash risk, and inability to fit observed implied volatility surfaces—while maintaining computational tractability through techniques such as Fourier-based analytical pricing and optional Numba JIT compilation.

---

## Table of Contents

1. Introduction
2. Background and Literature Review
3. Problem Definition
4. Methodology
5. System Architecture and Implementation
6. Mathematical Models
7. Computational Methods
8. Calibration and Validation
9. Results
10. Limitations and Future Work
11. Conclusion
12. References

---

## 1. Introduction

Options are financial derivatives whose value depends on an underlying asset's future price behaviour. This project focuses exclusively on **European-style options**—contracts that can only be exercised at the expiration date. This restriction is deliberate: European options admit closed-form analytical solutions (Black & Scholes, 1973; Merton, 1973) and straightforward Monte Carlo estimation via terminal price distributions, making them the natural setting for comparing model accuracy. The foundational Black-Scholes-Merton (BSM) model provides an elegant closed-form solution under the assumption of constant volatility. However, empirical evidence consistently demonstrates that volatility is neither constant nor deterministic—markets exhibit volatility smiles, skews, and clustering that the BSM framework cannot reproduce (Gatheral, 2006).

This project addresses that gap by implementing a progression of related European option pricing models, each relaxing a key assumption of its predecessor. The system is built as a usable research terminal—a web-based application that allows users to price European options, scan live markets for valuation discrepancies, validate model accuracy against observed quotes, and backtest model-driven trading strategies with realistic execution assumptions. The primary benchmark asset is the S&P 500 Index (^SPX), whose listed options are European-style, making it the ideal candidate for this framework.

The project integrates core mathematical principles—stochastic calculus, partial differential equations, numerical integration, and statistical estimation—with modern software engineering practices including vectorized computation, asynchronous API design, and modular architecture. This integration of mathematics and computer science is the central objective of IMCS 3020U.

---

## 2. Background and Literature Review

### 2.1 The Black-Scholes-Merton Framework

Black and Scholes (1973) derived a closed-form pricing formula for European options—options exercisable only at maturity—by assuming the underlying asset follows geometric Brownian motion (GBM) with constant drift and volatility. Merton (1973) independently extended this framework with a rigorous treatment of continuous-time finance. The BSM formula expresses the price of a European call option as:

$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where $d_1 = \frac{\ln(S_0/K) + (r - q + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$ and $d_2 = d_1 - \sigma\sqrt{T}$.

Despite its theoretical elegance, the BSM model's constant-volatility assumption is violated in practice, as evidenced by the volatility smile phenomenon observed across all major options markets (Hull, 2018).

### 2.2 Monte Carlo Simulation in Finance

Glasserman (2003) provides a comprehensive treatment of Monte Carlo methods in financial engineering. The core idea is to approximate the risk-neutral expectation $V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[\text{Payoff}(S_T)]$ by simulating a large number of terminal price scenarios and averaging the discounted payoffs. Monte Carlo methods are particularly valuable for path-dependent derivatives and models without closed-form solutions.

### 2.3 Stochastic Volatility Models

Heston (1993) proposed a stochastic volatility model in which the variance follows a Cox-Ingersoll-Ross (CIR) mean-reverting process. The model's key advantage is a semi-closed-form solution via Fourier inversion of the characteristic function, enabling efficient pricing without Monte Carlo simulation. The Heston model naturally generates volatility skew through the correlation parameter $\rho$ between the asset and variance Brownian motions.

### 2.4 Jump Diffusion Models

Merton (1976) extended the GBM framework by adding Poisson-distributed jump events to capture sudden, discontinuous price movements such as market crashes. The jump diffusion model produces heavier tails than GBM and can better price out-of-the-money put options, which serve as crash insurance.

### 2.5 Local Stochastic Volatility

Dupire (1994) introduced the concept of local volatility—a deterministic function $\sigma(S, t)$ that reproduces all observed European option prices exactly. The Local Stochastic Volatility (LSV) framework combines Dupire's local volatility with Heston's stochastic variance process through a leverage function $L(S, t)$, achieving both the realistic dynamics of stochastic volatility and exact calibration to the market surface (Gatheral, 2006).

### 2.6 Discretization and Simulation Schemes

Andersen (2008) developed efficient discretization schemes for the Heston model that preserve positivity of the variance process. The Feller condition $2\kappa\theta > \xi^2$ governs whether the variance process can reach zero; when violated, special numerical treatment is required.

---

## 3. Problem Definition

The core problem addressed by this project is: **How can progressively sophisticated mathematical models be implemented, validated, and deployed in an integrated system that demonstrates both theoretical understanding and practical computational competence?**

Specifically, the project tackles three sub-problems:

1. **Model Implementation**: Translating the stochastic differential equations and analytical pricing formulas of the model stack into efficient, numerically stable Python code.
2. **Validation and Calibration**: Verifying that Monte Carlo simulations converge to known analytical solutions, and calibrating model parameters to live market data.
3. **Practical Application**: Building a research terminal that applies these models to real-time market scanning, historical backtesting, and risk analysis—demonstrating the integration of mathematical theory with software engineering.

---

## 4. Methodology

The project follows a layered development methodology, progressing from simple to complex:

### 4.1 Model Evolution Strategy

Each model is implemented as a self-contained module, with explicit documentation of its mathematical foundation, assumptions, and limitations. All models price **European-style options exclusively**; early exercise is not supported. The progression is:

1. **BSM Analytical** → Baseline benchmark with closed-form European option solution
2. **GBM Monte Carlo** → Validates Monte Carlo convergence to BSM for European payoffs
3. **Heston Stochastic Volatility** → Addresses constant-volatility limitation
4. **Merton Jump Diffusion** → Addresses absence of crash risk
5. **LSV** → Addresses inability to fit observed volatility surfaces

### 4.2 Computational Approach

All simulations use vectorized NumPy operations to eliminate Python-level loops where possible. For models requiring path-level iteration (Heston, Jump Diffusion), optional Numba JIT compilation is employed to achieve near-C performance. The Heston model additionally implements Fourier-based analytical pricing for high-throughput scanning (~5 ms per option versus ~500 ms for Monte Carlo).

### 4.3 Data Sources

The system integrates multiple market data sources:

| Source | Data Type | Usage |
|--------|-----------|-------|
| Yahoo Finance (yfinance) | Live spot prices, historical closes, options chains, implied volatilities | Primary data feed for real-time scanning, pricing, and risk analysis (Yahoo Finance, n.d.) |
| Polygon.io REST API | Spot prices, historical daily closes, options reference metadata | Secondary/fallback provider for supported REST workflows (Polygon.io, n.d.) |
| S&P 500 Daily Options Data (Kaggle) | Daily EOD bid/ask quotes, implied volatility, Greeks, volume (2010–2023) | Historical backtester input via `combined_options_data.csv` (Singh, 2024; Cboe Global Markets, n.d.) |
| U.S. Treasury Bills (^IRX) | 13-week T-Bill yield | Risk-free rate proxy, fetched live via Yahoo Finance |

The yfinance library provides the primary options chain data, including strikes, expirations, bid/ask quotes, implied volatilities, volume, and open interest for all listed European-style SPX options. The Polygon.io API serves as an optional secondary provider for spot price history when an API key is configured. Historical backtesting uses the "S&P 500 Daily Options Data (2010–2023)" dataset published on Kaggle by Singh (2024), which contains CBOE end-of-day SPX option quote snapshots including bid, ask, implied volatility, delta, and volume fields across approximately 13 years of trading history.

### 4.4 Validation Framework

Model correctness is verified through:
- **Convergence tests**: Monte Carlo prices must converge to BSM analytical European option prices as path count increases
- **Statistical tests**: Terminal price distributions are tested against theoretical moments
- **Calibration**: Heston parameters are fitted to live implied volatility surfaces using constrained optimization (SLSQP)
- **Backtesting**: Model-driven trading strategies are tested against historical data with realistic execution assumptions

---

## 5. System Architecture and Implementation

### 5.1 Technology Stack

The system is implemented in Python 3.12+ using the following libraries:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Numerical Engine | NumPy ≥ 1.26, SciPy ≥ 1.11 | Vectorized simulation, optimization, integration |
| JIT Compilation | Numba (optional) | Near-C performance for path-level loops |
| Automatic Differentiation | JAX (optional) | Exact, noise-free Greeks via AD |
| Web Frontend | Streamlit ≥ 1.30 | Interactive dashboard with five analysis tabs |
| API Backend | FastAPI | Asynchronous options scanning endpoint |
| Visualization | Plotly ≥ 5.18 | Interactive 3D surfaces and charts |
| Market Data | yfinance ≥ 0.2.36, Polygon.io API | Live spot prices, European options chains (Yahoo Finance, n.d.; Polygon.io, n.d.) |

### 5.2 Module Architecture

The codebase is organized into three layers:

```
src/
├── core/                    # Mathematical engine
│   ├── black_scholes.py     # BSM analytical formula
│   ├── gbm_engine.py        # GBM Monte Carlo simulation
│   ├── heston_model.py      # Heston SV (MC + Fourier)
│   ├── jump_diffusion.py    # Merton Jump Diffusion
│   ├── lsv_model.py         # Local Stochastic Volatility
│   ├── greeks.py            # Option Greeks (AD + finite diff)
│   ├── calibration_engine.py# Heston + LSV calibration
│   ├── scanner_engine.py    # Batch valuation gap scanner
│   ├── backtester.py        # Historical + synthetic backtester
│   ├── model_evaluation.py  # Surface-fit diagnostics
│   ├── data_fetcher.py      # Market data with provider fallbacks
│   └── config.py            # Centralized configuration constants
├── api/
│   └── main.py              # FastAPI async scanning endpoint
└── web/
    ├── app.py               # Streamlit main application
    └── tabs/                # Five analysis tab modules
```

### 5.3 Frontend Design

The Streamlit dashboard provides five analysis tabs:

1. **Option Pricing**: Single-option pricing with path visualization and probability metrics
2. **Valuation Scanner**: Live market scanning for model-versus-market valuation gaps
3. **Model Validation**: Quote-based live model fit diagnostics (MAE, RMSE, NBBO coverage)
4. **Backtester**: Historical delta-hedged strategy backtesting with cost sensitivity analysis
5. **Risk Surfaces**: 3D vectorized Greek surfaces (Gamma, Vega) across spot and volatility

---

## 6. Mathematical Models

### 6.1 Black-Scholes-Merton

The BSM model prices **European options** by assuming the asset follows GBM under the risk-neutral measure:

$$dS_t = (r - q)S_t\,dt + \sigma S_t\,dW_t$$

The exact solution for terminal price is:

$$S_T = S_0 \exp\left((r - q - \tfrac{1}{2}\sigma^2)T + \sigma\sqrt{T}\,Z\right), \quad Z \sim N(0,1)$$

The implementation (`black_scholes.py`) computes $d_1$, $d_2$, and applies the cumulative normal distribution via `scipy.stats.norm.cdf`. Input validation ensures $T > 0$ and $\sigma > 0$ to prevent division by zero. The formula is valid exclusively for European options; American-style early exercise is not modelled.

### 6.2 GBM Monte Carlo Engine

The GBM engine (`gbm_engine.py`) generates $N$ terminal prices using the exact GBM solution and estimates the European option price as:

$$\hat{V}_0 = e^{-rT} \frac{1}{N}\sum_{i=1}^{N}\max(S_T^{(i)} - K, 0)$$

Because European options depend only on the terminal price $S_T$ and not on the path taken, the simulation need only generate $S_T$ values directly—full path simulation is provided separately for visualization purposes.

The implementation uses vectorized NumPy operations: random normals are generated in a single `np.random.randn(n_sims)` call, and the drift-diffusion calculation is applied element-wise. Full path simulation for visualization uses cumulative log-returns with `np.cumsum`.

### 6.3 Heston Stochastic Volatility

The Heston model (`heston_model.py`) evolves two coupled SDEs:

$$dS_t = (r - q)S_t\,dt + \sqrt{V_t}\,S_t\,dW_t^S$$
$$dV_t = \kappa(\theta - V_t)\,dt + \xi\sqrt{V_t}\,dW_t^V$$
$$\text{Corr}(dW_t^S, dW_t^V) = \rho$$

The implementation provides two pricing paths:

1. **Monte Carlo**: Euler-Maruyama discretization with full truncation scheme ($V_{\text{pos}} = \max(V, 0)$) to prevent negative variance. Correlated Brownian motions are constructed via Cholesky decomposition: $Z^V = \rho Z^S + \sqrt{1-\rho^2}Z^{\perp}$.

2. **Fourier Analytical**: The semi-closed-form solution via the characteristic function:
$$C = S_0 e^{-qT}P_1 - Ke^{-rT}P_2$$
where $P_j = \frac{1}{2} + \frac{1}{\pi}\int_0^{\infty}\text{Re}\left[\frac{e^{-iu\ln K}\phi_j(u)}{iu}\right]du$

The Fourier path is approximately 100× faster than Monte Carlo and is used by the scanner for batch pricing.

**Feller Condition**: The implementation explicitly checks $2\kappa\theta > \xi^2$ and surfaces the result in the UI, demonstrating awareness of model boundary conditions.

### 6.4 Merton Jump Diffusion

The Jump Diffusion model (`jump_diffusion.py`) extends GBM with Poisson jump arrivals:

$$dS_t = (r - q - \lambda\kappa_J)S_t\,dt + \sigma S_t\,dW_t + (e^Y - 1)S_t\,dN_t$$

where $N_t \sim \text{Poisson}(\lambda)$, $Y \sim N(\mu_J, \sigma_J^2)$, and $\kappa_J = e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1$ is the risk-neutral jump compensator.

The compensator term is critical: without it, the process is not a martingale under the risk-neutral measure, which would distort expected discounted prices. The implementation includes the compensator in the drift adjustment.

### 6.5 Local Stochastic Volatility

The LSV model (`lsv_model.py`) modulates the Heston process with a leverage function:

$$dS_t = (r - q)S_t\,dt + L(S_t, t)\sqrt{V_t}\,S_t\,dW_t^S$$

The leverage function $L(K, T)$ is calibrated from the Dupire local variance surface:

$$\sigma_{\text{loc}}^2 = \frac{\partial w/\partial T}{1 - \frac{k}{w}\frac{\partial w}{\partial k} + \frac{1}{4}(-\frac{1}{4} - \frac{1}{w} + \frac{k^2}{w^2})(\frac{\partial w}{\partial k})^2 + \frac{1}{2}\frac{\partial^2 w}{\partial k^2}}$$

where $w(k, T) = \sigma_{IV}^2 T$ is total implied variance, $k = \ln(K/F)$ is log-moneyness, and $F = S_0 e^{(r-q)T}$ is the forward price of the underlying (Gatheral, 2006). The leverage function is then:

$$L(K, T) = \sqrt{\frac{\sigma_{\text{loc}}^2}{\mathbb{E}[V_t]}}$$

where $\mathbb{E}[V_t] = \theta + (V_0 - \theta)e^{-\kappa t}$ is the analytic Heston expected variance.

---

## 7. Computational Methods

### 7.1 Vectorized Simulation

All math is vectorized using NumPy per project standards. For example, the GBM terminal price simulation generates all $N$ paths in three vectorized operations:

```python
Z = np.random.randn(n_sims)
drift = (r - 0.5 * sigma**2) * T
S_T = S0 * np.exp(drift + sigma * np.sqrt(T) * Z)
```

### 7.2 JIT Compilation

For models requiring path-level iteration (Heston, Jump Diffusion), the `@njit(fastmath=True)` decorator from Numba compiles Python functions to optimized machine code. A graceful fallback ensures the system operates without Numba installed.

### 7.3 Automatic Differentiation for Greeks

Option Greeks (Delta, Gamma, Vega) are computed using JAX automatic differentiation when available:

```python
_jax_delta = jax.jit(jax.grad(bs_price_jax, argnums=0))
_jax_gamma = jax.jit(jax.grad(jax.grad(bs_price_jax, argnums=0), argnums=0))
```

This provides exact, noise-free Greeks without the numerical error inherent in finite-difference approximations.

### 7.4 Numerical Integration

The Heston Fourier pricer uses `scipy.integrate.quad` for numerical integration of the characteristic function over the domain $[10^{-4}, 200]$ with adaptive error control (`epsabs=1e-6`, `limit=100`).

### 7.5 Constrained Optimization

Heston calibration uses `scipy.optimize.minimize` with the SLSQP method and parameter bounds ($\kappa > 0$, $\theta > 0$, $\xi > 0$, $-1 < \rho < 0$, $V_0 > 0$) to minimize the weighted sum of squared IV errors against market-observed implied volatilities.

---

## 8. Calibration and Validation

### 8.1 Convergence Testing

The test suite (`tests/test_convergence_validation.py`) verifies that GBM Monte Carlo prices converge to BSM analytical prices. As the number of simulation paths $N$ increases, the Monte Carlo standard error decreases proportionally to $1/\sqrt{N}$, confirming correct implementation.

### 8.2 Heston Calibration Pipeline

The calibration engine (`calibration_engine.py`) performs the following steps:

1. Filter the options chain to liquid, near-the-money contracts (80%–120% moneyness)
2. Extract market implied volatilities and apply Gaussian ATM-centered weights
3. For each parameter vector $(\kappa, \theta, \xi, \rho, V_0)$, price all contracts via Fourier inversion
4. Back-solve model implied volatilities via binary search on BSM
5. Minimize weighted SSE using SLSQP with 500-iteration budget

### 8.3 IV Surface Construction

The `build_iv_surface` function interpolates scattered options chain data onto a regular $(K \times T)$ grid using `scipy.interpolate.griddata` with a fallback chain (linear → nearest). Gaussian smoothing ($\sigma = 0.5$) stabilizes the surface, and coverage metrics quantify data quality.

### 8.4 Live Model Validation

The model evaluation module (`model_evaluation.py`) computes quote-based fit metrics:
- **Price MAE/RMSE** versus quoted midpoint
- **IV MAE** versus quoted market implied volatility
- **NBBO coverage**: percentage of model prices falling within the bid-ask spread
- **Spread-normalized error**: model error expressed in units of the quoted spread

### 8.5 Backtesting Framework

The backtester (`backtester.py`) is used through the current application in **historical quote mode**, which uses actual SPX option bid/ask data from CSV with real market execution prices and the 100× contract multiplier. The codebase still contains an older synthetic proxy implementation for backward compatibility and internal experimentation, but the current user-facing workflow is historical-quote based.

The historical workflow enforces:
- **No look-ahead bias**: Volatility is computed strictly from data before the current date, with an assertion guard
- **Delta hedging**: Daily rebalancing of the underlying position to isolate volatility-driven P&L
- **Transaction costs**: Entry costs (5 bps), hedge rebalancing costs (1 bps), and slippage (1%)
- **Cost sensitivity analysis**: Final returns are reported under 0.5×, 1.0×, and 1.5× cost multiplier scenarios
- **Methodology disclosure**: The backtester explicitly labels its data sources and assumptions in the output

---

## 9. Results

### 9.1 Model Convergence

GBM Monte Carlo prices converge to BSM analytical values within statistical tolerance. At $N = 100{,}000$ paths, the standard error is consistently below $0.01$ for typical parameter configurations, confirming the numerical correctness of the simulation engine.

[INSERT TABLE 1: GBM Monte Carlo convergence against BSM with columns for $N$, Monte Carlo price, BSM price, absolute error, and standard error]

### 9.2 Heston Calibration Quality

When calibrated to live SPX options chains, the Heston model achieves SSE values on the order of $10^{-4}$ to $10^{-3}$ across 50–200 liquid contracts. The Feller condition is satisfied for typical equity calibrations ($\kappa \approx 2$, $\theta \approx 0.04$, $\xi \approx 0.3$), confirming the variance process is well-posed.

[INSERT TABLE 2: Heston calibration metrics with SSE, contract count, calibrated parameters, and Feller discriminant]

### 9.3 Scanner Performance

The Fourier-based Heston pricer processes entire options chains in under 5 seconds (versus several minutes for Monte Carlo), enabling real-time scanning. The scanner's bid-ask-aware signal logic (BUY when model price > ask; SELL when model price < bid) eliminates phantom edges that midpoint-only analysis would produce.

[INSERT TABLE 3: Scanner validation metrics with price MAE, price RMSE, IV MAE, NBBO coverage, and mean error in spreads]

### 9.4 Backtester Insights

The delta-hedged backtester demonstrates that:
- Transaction costs and slippage materially affect strategy returns (cost sensitivity analysis shows 30–50% return variation across cost scenarios)
- Model choice affects edge detection: Heston tends to price OTM puts higher than BSM, generating different signal distributions
- The no-look-ahead guard prevents the most common form of backtest contamination

[INSERT TABLE 4: Historical backtest metrics with final value, total return, Sharpe ratio, max drawdown, win rate, total trades, and total hedge costs]

### 9.5 LSV Leverage Surface

The LSV calibration produces leverage matrices that deviate from unity primarily at the wings of the volatility surface, confirming that the Heston base model captures ATM dynamics well but requires correction for deep OTM/ITM strikes—consistent with the theoretical motivation for LSV.

[INSERT FIGURE 6: LSV leverage surface or calibrated surface comparison]

---

## 10. Limitations and Future Work

### 10.1 Current Limitations

1. **Heston Monte Carlo uses Euler-Maruyama discretization**, which can introduce bias when the Feller condition is near violation. More advanced schemes such as the Quadratic Exponential (QE) scheme (Andersen, 2008) would improve accuracy.

2. **The LSV leverage function uses nearest-neighbor interpolation** within JIT-compiled code due to Numba's restrictions on SciPy interpolation. Bilinear interpolation would improve smoothness.

3. **Market data dependency**: The system relies on yfinance for live options chains, which may experience rate limiting or data gaps during high-volatility periods.

4. **Historical quote coverage is concentrated on SPX** in the current user-facing backtester workflow, which limits cross-asset generalization.

### 10.2 Future Improvements

1. **Variance reduction techniques** (antithetic variates, control variates) to reduce Monte Carlo noise without increasing path count
2. **American option support** via Longstaff-Schwartz least-squares Monte Carlo
3. **Multi-asset correlation** for portfolio-level pricing and risk analysis
4. **GPU acceleration** using CuPy or JAX for massively parallel path simulation
5. **Extended backtester coverage** to additional tickers beyond SPX using historical options databases

---

## 11. Conclusion

This project demonstrates the integration of mathematical modelling and computer science in the domain of quantitative finance. By implementing a linked stack of option pricing models—BSM, GBM Monte Carlo, Heston Stochastic Volatility, Merton Jump Diffusion, and Local Stochastic Volatility—the system illustrates how each model addresses specific empirical limitations of its predecessor.

The mathematical contributions include implementations of stochastic calculus (SDEs for GBM, Heston, and Jump Diffusion), numerical integration (Fourier-based Heston pricing), constrained optimization (Heston calibration), and finite-difference PDE methods (Dupire local variance for LSV leverage calibration).

The computer science contributions include vectorized numerical computing, JIT compilation for performance-critical paths, automatic differentiation for exact Greeks, asynchronous API design, and a modular architecture that separates mathematical engines from presentation and data layers.

The system is deployed as a usable research terminal with five analysis tabs, demonstrating that the mathematical theory translates into practical analytical capability. The backtester's explicit methodology disclosure and the calibration engine's Feller condition checking exemplify the project's commitment to scientific honesty and awareness of model limitations.

---

## References

Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*, *11*(3), 1–42. https://doi.org/10.21314/JCF.2008.189

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. *Journal of Political Economy*, *81*(3), 637–654. https://doi.org/10.1086/260062

Cboe Global Markets. (n.d.). *Cboe Options Exchange: S&P 500 Index options (SPX)*. https://www.cboe.com/tradable_products/sp_500/

Dupire, B. (1994). Pricing with a smile. *Risk*, *7*(1), 18–20.

Gatheral, J. (2006). *The volatility surface: A practitioner's guide*. John Wiley & Sons.

Glasserman, P. (2003). *Monte Carlo methods in financial engineering*. Springer-Verlag.

Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, *6*(2), 327–343. https://doi.org/10.1093/rfs/6.2.327

Hull, J. C. (2018). *Options, futures, and other derivatives* (10th ed.). Pearson.

Merton, R. C. (1973). Theory of rational option pricing. *The Bell Journal of Economics and Management Science*, *4*(1), 141–183. https://doi.org/10.2307/3003143

Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, *3*(1–2), 125–144. https://doi.org/10.1016/0304-405X(76)90022-2

Polygon.io. (n.d.). *Polygon.io: Stock, options, and crypto market data APIs*. https://polygon.io/

Singh, S. (2024). *S&P 500 daily options data (2010–2023)* [Data set]. Kaggle. https://www.kaggle.com/datasets/shubhamcodez/s-and-p-500-daily-options-data-2010-2023

Yahoo Finance. (n.d.). *Yahoo Finance: Stock market live, quotes, business & finance news*. https://finance.yahoo.com/

---

## Appendix A: Application Screenshots

### Figure 1: Option Pricing Tab

![Figure 1: Option Pricing Tab — Single European option pricing interface showing model selection (Black-Scholes, GBM, Heston, Jump Diffusion, LSV), spot/strike/maturity inputs, and real-time pricing.](/Users/Bach/Documents/OTU WINTER 2026/Project/monte_carlo_project/docs/screenshots/option-pricing.png)

### Figure 2: Live Valuation Scanner

![Figure 2: Live Valuation Scanner — Scanning SPX European options for model-versus-market valuation gaps with bid-ask-aware signal logic and data-quality diagnostics.](/Users/Bach/Documents/OTU WINTER 2026/Project/monte_carlo_project/docs/screenshots/live-scanner.png)

### Figure 3: Backtester

![Figure 3: Backtester Tab — Historical delta-hedged strategy backtesting interface showing capital, historical CSV mode, edge threshold, and fair-value model selection.](/Users/Bach/Documents/OTU WINTER 2026/Project/monte_carlo_project/docs/screenshots/backtester.png)

[INSERT FIGURE 4: 3D Greek Surface — Screenshot showing vectorized Gamma or Vega surface across spot price and volatility axes]

[INSERT FIGURE 5: Heston Calibration — Screenshot or diagram showing the calibrated implied volatility surface versus market-observed IVs]

---

## Appendix B: How to Run the System

### Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Frontend

```bash
streamlit run src/web/app.py
```

Default port: 8501. Access at http://localhost:8501.

### Running the Backend API

```bash
python3 -m uvicorn src.api.main:app --reload
```

Default port: 8000. API docs at http://localhost:8000/docs.
