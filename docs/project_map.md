# Project Map

This document provides a codebase map with the role of each major file and how the pieces connect.

## Top-Level Structure

```text
monte_carlo_project/
├── README.md
├── RUN_GUIDE.md
├── requirements.txt
├── docs/
├── src/
│   ├── api/
│   ├── core/
│   └── web/
└── tests/
```

## Root Files

### [README.md](README.md)

Project overview, positioning, and entry points.

### [RUN_GUIDE.md](RUN_GUIDE.md)

Environment setup, launch sequence, testing, troubleshooting, and demo workflow.

### [requirements.txt](requirements.txt)

Declared Python dependencies for the current implementation.

## Docs Folder

### [docs/mathematical_foundations.md](docs/mathematical_foundations.md)

Detailed explanation of the quantitative models, concepts, and why they matter.

### [docs/project_map.md](docs/project_map.md)

This file.

### Existing roadmap and remediation docs

- [docs/backtester_strategy_upgrade_impact.md](docs/backtester_strategy_upgrade_impact.md)
- [docs/final_project_roadmap.md](docs/final_project_roadmap.md)
- [docs/institutional_upgrade_tasks.md](docs/institutional_upgrade_tasks.md)
- [docs/loki_multi_agent_risk_remediation.md](docs/loki_multi_agent_risk_remediation.md)
- [docs/loki_remediation_directive.md](docs/loki_remediation_directive.md)

These are internal planning artifacts and technical evolution notes.

## Source Code

## API Layer

### [src/api/main.py](src/api/main.py)

FastAPI entry point.

Responsibilities:

- receive structured scan requests from the frontend
- run `scan_for_arbitrage`
- return result rows plus diagnostics metadata

## Core Quant Layer

### [src/core/black_scholes.py](src/core/black_scholes.py)

Analytical Black-Scholes pricing with dividend yield support.

### [src/core/gbm_engine.py](src/core/gbm_engine.py)

GBM terminal and path simulation, used for baseline Monte Carlo behavior and visualization.

### [src/core/jump_diffusion.py](src/core/jump_diffusion.py)

Merton Jump Diffusion simulation with risk-neutral jump compensator.

### [src/core/heston_model.py](src/core/heston_model.py)

Heston stochastic volatility simulation, Feller condition checks, and Fourier pricing path.

### [src/core/lsv_model.py](src/core/lsv_model.py)

Local stochastic volatility path logic and leverage-surface calibration support.

### [src/core/greeks.py](src/core/greeks.py)

Greeks calculations, including JAX-based automatic differentiation for selected sensitivities.

### [src/core/calibration_engine.py](src/core/calibration_engine.py)

Calibration workflows.

Responsibilities:

- implied volatility inversion
- Heston parameter fitting
- IV surface construction
- LSV leverage-function calibration

### [src/core/scanner_engine.py](src/core/scanner_engine.py)

Live option-chain scanner.

Responsibilities:

- sanitize option-chain rows
- route to GBM, Jump Diffusion, Heston, or LSV pricing logic
- apply spread-aware edge logic
- emit structured diagnostics

### [src/core/backtester.py](src/core/backtester.py)

Controlled research backtester.

Responsibilities:

- no-look-ahead volatility estimation
- synthetic option entry proxy
- daily delta hedging
- cost accounting
- methodology disclosure in the returned payload

### [src/core/data_fetcher.py](src/core/data_fetcher.py)

Yahoo Finance data access for spot, historical volatility, expirations, and live option chains.

### [src/core/config.py](src/core/config.py)

Central constants for scanner filters, slippage tiers, and model defaults.

## Web Layer

### [src/web/app.py](src/web/app.py)

Thin Streamlit orchestrator.

Responsibilities:

- page config and CSS
- shared sidebar state
- model parameter collection
- tab dispatch

### [src/web/tabs/tab_option_analysis.py](src/web/tabs/tab_option_analysis.py)

Single-option pricing and model comparison view.

### [src/web/tabs/tab_scanner.py](src/web/tabs/tab_scanner.py)

Live scanner interface.

Responsibilities:

- expiration selection
- calibration actions
- API call or local fallback
- diagnostics rendering
- results filtering and export

### [src/web/tabs/tab_backtester.py](src/web/tabs/tab_backtester.py)

Backtester results view.

Responsibilities:

- methodology explanation
- metrics and equity curve
- trade log and exports
- cost sensitivity display

### [src/web/tabs/tab_portfolio_risk.py](src/web/tabs/tab_portfolio_risk.py)

Risk-surface and implied-volatility-surface visualization tab.

## Tests

### [tests/test_scanner_regression.py](tests/test_scanner_regression.py)

Scanner route and diagnostics-return tests.

### [tests/test_backtester_strategy_upgrades.py](tests/test_backtester_strategy_upgrades.py)

Backtester disclosure, put-path, and hedge-cost behavior tests.

### [tests/test_calibration_regression.py](tests/test_calibration_regression.py)

Heston calibration regression using a synthetic surface.

### [tests/test_loki_remediation.py](tests/test_loki_remediation.py)

Quant-correctness tests for jump diffusion and Heston relationships.

### [tests/test_lsv_calibration.py](tests/test_lsv_calibration.py)

LSV surface-building and leverage calibration tests.

### [tests/test_convergence_validation.py](tests/test_convergence_validation.py)

Monte Carlo convergence and edge-case validation against Black-Scholes.

### [tests/test_jump_diffusion.py](tests/test_jump_diffusion.py)

Performance and statistical tests for jump diffusion simulation.

### [tests/benchmark_jit.py](tests/benchmark_jit.py)

Performance benchmark script for JIT-enabled simulation paths.

## Runtime Flow

### Live scanner flow

```text
Streamlit tab_scanner
    -> data_fetcher gets spot/expirations/chain
    -> optional calibration_engine workflow
    -> FastAPI /scan or local scanner_engine fallback
    -> scanner_engine routes model and returns results + diagnostics
    -> Streamlit renders metrics, diagnostics, tables, and charts
```

### Backtester flow

```text
Streamlit tab_backtester
    -> backtester fetches historical prices
    -> no-look-ahead volatility estimate
    -> Heston or Jump Diffusion fair-value logic
    -> synthetic entry proxy + delta hedge + costs
    -> metrics, equity curve, and trade log returned to UI
```

## Best Way To Read The Code

If you are new to the codebase, read in this order:

1. [README.md](README.md)
2. [docs/mathematical_foundations.md](docs/mathematical_foundations.md)
3. [src/web/app.py](src/web/app.py)
4. [src/core/scanner_engine.py](src/core/scanner_engine.py)
5. [src/core/backtester.py](src/core/backtester.py)
6. [src/core/calibration_engine.py](src/core/calibration_engine.py)
7. the regression tests under [tests](tests)