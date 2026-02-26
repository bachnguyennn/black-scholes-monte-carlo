# Live Arbitrage Scanner - Architecture & Implementation Plan

## Objective
Transform the existing Monte Carlo Option Pricing engine into a **Live Arbitrage Scanner**. The system will fetch real-time live options chain data, calculate the "Fair Value" using our Jump Diffusion Monte Carlo engine, and compare it against the live Market Price to identify mispriced options (Arbitrage Opportunities).

## Architecture Flow
1.  **Data Ingestion**: Fetch real-time options chain for a given ticker (e.g., SPY) using `yfinance`.
2.  **Vectorized Pricing**: Pass the entire options chain (Strikes, Maturities, Option Types) into the `jump_diffusion.py` engine to calculate the theoretical Fair Value for all options simultaneously.
3.  **Opportunity Calculation**: 
    - `Opportunity = Theoretical Fair Value - Market Ask Price` (for buying)
    - `Opportunity = Market Bid Price - Theoretical Fair Value` (for selling)
4.  **UI Presentation**: Display a sorted, color-coded leaderboard of the most mispriced options in the Streamlit dashboard.

---

## Delegation: Quant Team

**Primary Goal:** Build the data pipeline and batch processing engine.

### Task 1: Options Chain Data Fetcher (`src/core/data_fetcher.py`)
- Create a new module using `yfinance` to fetch live options data.
- **Function `get_options_chain(ticker)`**: 
  - Fetch the current spot price ($S_0$), Risk-Free Rate ($r$), and historical volatility ($\sigma$).
  - Fetch all available expirations.
  - For the nearest 3 expirations, download all Call and Put data (Strike, Bid, Ask, Implied Volatility).
  - Return a structured Pandas DataFrame.

### Task 2: Batch Processing Engine (`src/core/scanner_engine.py`)
- Create a new module that connects `data_fetcher.py` with `jump_diffusion.py`.
- **Function `scan_for_arbitrage(options_df, S0, r, sigma, jump_params)`**:
  - Vectorize the inputs from the DataFrame to feed into `simulate_jump_diffusion()`.
  - Calculate the Monte Carlo Fair Value for every row in the DataFrame.
  - Calculate the **Edge (Discrepancy)**: `Edge = MC_Price - Market_Mid_Price`.
  - Return the DataFrame sorted by the largest absolute Edge.

---

## Delegation: Web Team

**Primary Goal:** Build the Interactive Scanner UI in Streamlit.

### Task 1: Scanner Dashboard Tab (`src/web/app.py`)
- Refactor the UI to use Streamlit Tabs:
  - **Tab 1:** `Single Option Analysis` (The current dashboard).
  - **Tab 2:** `Live Arbitrage Scanner` (The new feature).

### Task 2: Scanner Controls & Display
- **Inputs:**
  - Ticker Symbol (e.g., SPY)
  - "Scan Market" Button
- **Progress:** Show a spinner/progress bar while `scan_for_arbitrage()` runs.
- **Data Table:** 
  - Display the results using `st.dataframe()`.
  - Columns needed: `Type`, `Strike`, `Expiration`, `Market Price`, `MC Fair Value`, `Edge ($)`, `Recommendation`.
- **Styling:**
  - Apply Pandas styling (`df.style.apply()`) to color-code the `Edge` column:
    - **Green** for severely undervalued options (Strong Buy).
    - **Red** for severely overvalued options (Strong Sell).

---

## Integration Strategy (For Lead Architect)
1. **Phase 1**: Quant team builds and unit tests `data_fetcher.py` to ensure reliable `yfinance` options retrieval.
2. **Phase 2**: Quant team implements the batch processing in `scanner_engine.py` ensuring the Monte Carlo engine can handle arrays of Strikes ($K$) and Maturities ($T$) without crashing.
3. **Phase 3**: Web team integrates the new tab into `app.py` and wires up the "Scan Market" button to the new Quant engine.
4. **Phase 4**: End-to-end testing hunting for real mispriced options on live market data.
