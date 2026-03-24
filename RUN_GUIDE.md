# Run Guide

This guide covers setup, runtime commands, validation, and the safest launch order for demos and screenshots.

## 1. Environment Setup

Create and activate a virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Required Dependencies

The current codebase uses:

- numerical stack: `numpy`, `pandas`, `scipy`
- UI stack: `streamlit`, `plotly`, `requests`
- market data: `yfinance`, optional Polygon REST integration via `requests`
- backend: `fastapi`, `uvicorn`
- performance: `numba`
- auto-diff Greeks: `jax`, `jaxlib`

If you are on macOS and `jax` install is slow, let the environment finish fully before launching the app. The Greeks module imports JAX directly.

## 3. Market Data Configuration

Optional environment variables:

```bash
export MARKET_DATA_PROVIDER=auto
export POLYGON_API_KEY=your_polygon_api_key
```

Provider modes:

- `auto`: prefer Polygon when a key is configured, otherwise use yfinance
- `polygon`: prefer Polygon for supported workflows and fall back to yfinance when needed
- `yfinance`: disable Polygon and use the legacy yfinance path

Practical Polygon free/basic behavior in this repo:

- supported: spot/history lookups for supported underlyings, plus expiration discovery via reference contracts when available
- still on yfinance: full options-chain fetches for the scanner/risk tabs, calibration inputs, and the `^IRX` risk-free rate lookup

This is deliberate. The repo does not claim Polygon free/basic provides a true live options-chain replacement.

## 4. Recommended Launch Order

Launch the API first, then Streamlit.

### Backend

Default backend port is `8000`:

```bash
export MARKET_DATA_PROVIDER=auto
export POLYGON_API_KEY=your_polygon_api_key   # optional
python3 -m uvicorn src.api.main:app --reload --port 8000
```

Optional health check:

```bash
curl http://127.0.0.1:8000/
```

### Frontend

In a second terminal:

```bash
export MARKET_DATA_PROVIDER=auto
export POLYGON_API_KEY=your_polygon_api_key   # optional
streamlit run src/web/app.py
```

Default frontend port is `8501`.

## 5. Running On A Custom Backend Port

If you run the backend on a custom port such as `8001`, export the scanner URL before launching Streamlit:

```bash
export QUANT_TERMINAL_SCAN_API_URL=http://127.0.0.1:8001/scan
streamlit run src/web/app.py
```

The scanner tab reads this environment variable and otherwise defaults to `http://127.0.0.1:8000/scan`.

## 6. Demo Workflow For Screenshots

For a stable demo session:

1. Start backend.
2. Start Streamlit.
3. Open the frontend in a clean browser session.
4. Use `SPY` as the initial ticker.
5. Capture at least one screenshot from Option Pricing, Live Scanner, Backtester, and Risk Surfaces.

Recommended order:

1. Landing view with sidebar and tabs visible.
2. Scanner results with diagnostics open.
3. Backtester with methodology warning and cost metrics visible.
4. Risk surface chart.

## 7. Testing

Run the focused regression suite from the project root:

```bash
PYTHONPATH=. pytest -q \
	tests/test_scanner_regression.py \
	tests/test_calibration_regression.py \
	tests/test_backtester_strategy_upgrades.py \
	tests/test_loki_remediation.py
```

Additional tests exist for convergence, jump diffusion behavior, and LSV calibration.

## 8. Troubleshooting

### Port already in use

If port `8000` or `8501` is busy, either stop the existing process or choose another port.

### Scanner falls back to local mode

If the backend is unavailable, the Streamlit scanner tab attempts a local fallback. This is acceptable for demos, but for a cleaner architecture story use the running FastAPI backend.

### Polygon is configured but options data still comes from yfinance

This is expected in the current implementation. Polygon free/basic is only used for supported REST workflows. The scanner and market IV surface still fetch full options chains through yfinance.

### JAX import or install issues

The Greeks module uses JAX. If it is missing, reinstall dependencies inside the active virtual environment:

```bash
pip install -r requirements.txt
```

## 9. Suggested Pre-Push Checklist

1. Confirm tests pass.
2. Confirm README links open correctly.
3. Confirm backend and frontend ports are aligned.
4. Confirm screenshots are saved under `docs/screenshots/`.
5. Confirm the project is described as a research platform, not a production trading engine.
