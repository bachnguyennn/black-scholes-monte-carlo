# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Quant Research Terminal V2 — a research-oriented options analytics platform (Ontario Tech course project). It compares multiple pricing models (Black-Scholes, GBM Monte Carlo, Heston, Jump Diffusion, LSV), runs a live valuation-gap scanner, calibrates to market surfaces, and provides a controlled research backtester. It is deliberately framed as a research platform, **not** a production trading engine — the backtester does not replay historical option NBBO fills, and this disclosure is load-bearing throughout the code and docs. Preserve that framing when editing.

## Commands

Run everything from the project root with the venv active. Imports assume the root is on the path.

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Backend (start first) — default port 8000
python3 -m uvicorn src.api.main:app --reload --port 8000

# Frontend (second terminal) — default port 8501
streamlit run src/web/app.py

# If backend runs on a non-default port, tell the scanner tab where it is:
export QUANT_TERMINAL_SCAN_API_URL=http://127.0.0.1:<port>/scan
```

### Tests

```bash
# Full suite
PYTHONPATH=. pytest -q

# Focused regression suite (fast, the ones that matter for demos)
PYTHONPATH=. pytest -q \
  tests/test_scanner_regression.py \
  tests/test_calibration_regression.py \
  tests/test_backtester_strategy_upgrades.py \
  tests/test_loki_remediation.py

# Single test
PYTHONPATH=. pytest -q tests/test_scanner_regression.py::test_name
```

`PYTHONPATH=.` is required — tests and scripts import via `from src.core...`.

### Report metrics

`PYTHONPATH=. python3 collect_metrics.py` regenerates the model metrics used in the Final Report.

## Environment / market data

Provider selection is env-driven (see RUN_GUIDE.md for full detail):

- `MARKET_DATA_PROVIDER` = `auto` (default) | `polygon` | `yfinance`
- `POLYGON_API_KEY` — optional; Polygon free/basic is intentionally only wired for spot/history and expiration discovery.
- **Full options-chain fetches always go through yfinance** (scanner, IV surface, calibration inputs, `^IRX` risk-free rate), regardless of provider. This is by design, not a bug — don't "fix" it by routing chains to Polygon.

## Architecture

Three layers under `src/`, glued by a shared parameter contract:

- **`src/core/`** — pure quant/engine logic, no UI. Pricing models (`black_scholes`, `gbm_engine`, `heston_model`, `jump_diffusion`, `lsv_model`), `greeks` (uses JAX autodiff — importing it pulls in JAX), `scanner_engine`, `calibration_engine`, `backtester`, `data_fetcher` (market data + provider fallback), `databento_provider`, `model_evaluation`, and `config`.
- **`src/api/`** — `main.py`, a FastAPI app exposing `POST /scan`. It receives an already-fetched options payload (`ScanRequest`) and delegates to `scanner_engine.scan_for_valuation_gaps(...)`, returning results plus a `diagnostics` block.
- **`src/web/`** — Streamlit UI. `app.py` builds shared sidebar state (ticker, model, Monte Carlo/Heston/jump params) and passes it into five tab modules under `src/web/tabs/` (`tab_option_analysis`, `tab_scanner`, `tab_model_validation`, `tab_backtester`, `tab_portfolio_risk`), each exposing a `render(...)` function.

**Key cross-file flow — the scanner:** the frontend fetches the live options chain (via `data_fetcher`), then `tab_scanner.py` POSTs it to the FastAPI `/scan` endpoint. If the API is unreachable it degrades gracefully by calling `scan_for_valuation_gaps` locally (`_run_local_scan`), tagging the `diagnostics.engine` field (`async_api` vs `local_fallback`) so the UI shows which path ran. Both paths call the same core function, so scanner behavior must stay consistent between `src/api/main.py` and `src/web/tabs/tab_scanner.py`.

**Configuration:** `src/core/config.py` is the single source of truth for tunable constants — scanner data-quality thresholds, slippage tiers, default Heston params, Greek grid resolution, backtester defaults. Do not inline magic numbers in modules; add or change them here.

**Scripts:** `scripts/databento_*.py` build/execute Databento SPX/XSP historical quote-download plans for the backtester (run with `PYTHONPATH=.`).

## Gotchas

- The 228 MB `combined_options_data.csv` and its zip are intentionally untracked/ignored — do not commit them. `docs/archive/` is also ignored; the rest of `docs/` (including the Final Report) is tracked.
- Greeks depend on JAX; on macOS the install can be slow — let it finish before launching.
- Prefer `^SPX` for the European-style workflow; `SPY`/`IVV`/`VOO` are American-style ETF proxies (the app warns about this).
