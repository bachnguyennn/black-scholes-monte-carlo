# Synthetic Backtester - Architecture & Implementation Plan

## Objective
Build a **Synthetic Backtester** to prove the profitability of the Jump Diffusion Monte Carlo model over historical data (e.g., the past year). Because real historical options data is locked behind expensive premium APIs, we will use a "Synthetic Builder" approach:
1. Fetch 100% accurate historical SPY stock prices.
2. Mathematically "synthesize" what the option market price *should* have been using Black-Scholes.
3. Compare it against our Jump Diffusion "Fair Value" to generate trading signals.
4. "Time-travel" forward to expiration to calculate the actual Profit & Loss (PnL) from a virtual $10,000 portfolio.

---

## Delegation: Quant Team

**Primary Goal:** Build the historical data pipeline and the simulation loop.

### Task 1: Historical Data Fetcher
- **File:** `src/core/backtester.py`
- **Logic:** Build a function to download 1 to 5 years of daily historical data for SPY. 
- **Tools:** Use `yfinance` or the user's provided Alpha Vantage API key for pulling daily `Close` prices.

### Task 2: The Trading Loop (Synthetic Engine)
- **Function:** `run_synthetic_backtest(initial_capital=10000)`
- **Step 1 (The Trigger):** Loop through the historical data, stopping at the **first trading day of every month**.
- **Step 2 (The Setup):** On that day, check the SPY Spot Price ($S_0$). Assume we are looking at a 1-month (30-day) At-The-Money (ATM) Call option ($K = S_0$). Calculate rolling 30-day historical volatility.
- **Step 3 (Synthetic Market Value):** Use the existing `black_scholes_price()` function to calculate the "Synthetic Market Premium". This represents what the option *would have cost* to buy that day.
- **Step 4 (The Scanner):** Run `simulate_jump_diffusion()` for that day. 
- **Step 5 (The Decision):** 
  - If `MC Fair Value > Synthetic Market Premium + 10% Edge`: **BUY**. Deduct the Premium from the virtual $10,000 portfolio.
  - Otherwise, do nothing.
- **Step 6 (The Resolution):** Fast-forward 30 days in the historical data to the expiration date. 
  - Look at the actual SPY price on that future date ($S_T$). 
  - Calculate payoff: `max(S_T - K, 0)`.
  - Add payoff cash back to the virtual portfolio.

### Task 3: Performance Metrics
- The backtester must return:
  - **Final Portfolio Value** (e.g., $15,000)
  - **Total Return %**
  - **Win Rate** (Winning Trades / Total Trades)
  - **A Pandas DataFrame** logging every trade, date, and PnL amount.

---

## Delegation: Web Team

**Primary Goal:** Build the Backtester UI in Streamlit.

### Task 1: New UI Tab (`src/web/app.py`)
- Add a new tab: **"📈 Historical Backtester"**.

### Task 2: Backtest Controls
- **Inputs:** 
  - Starting Capital (Default: $10,000)
  - Timeframe (e.g., Past 1 Year, Past 5 Years)
  - "Run Backtest" Button

### Task 3: Results Visualization
- **Equity Curve:** Use `plotly.express` to graph the virtual portfolio value over time (X-axis: Date, Y-axis: Account Balance).
- **Key Metrics:** Display the "Final Balance", "Win Rate", and "Total Return" using Streamlit's `st.metric()`.
- **Trade Log:** Display the Pandas DataFrame returned by the Quant team as an interactive table (`st.dataframe()`) so users can inspect every historical trade.

---

## Integration Strategy (For Lead Architect)
1. **Phase 1:** Quant team builds the "Time-Travel" loop over historical stock data and ensures the Black-Scholes and Monte Carlo engines can be called inside the loop without crashing or leaking memory.
2. **Phase 2:** Quant team validates the synthetic premium math against known basic option pricing rules.
3. **Phase 3:** Web team integrates the results and ensures the Plotly equity curve updates correctly after the backtest is run.
