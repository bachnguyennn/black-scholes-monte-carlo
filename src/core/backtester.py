"""
backtester.py

Synthetic Backtester: Proves historical profitability of the Jump Diffusion
Monte Carlo model by "time-traveling" through past SPY data.

Strategy:
    - On the 1st trading day of each month, synthesize ATM call option prices
      using Black-Scholes (as "market") and Jump Diffusion MC (as "fair value").
    - If MC Fair Value > BS Market Price by a threshold (edge), BUY the option.
    - Fast-forward to expiration (30 days later) and settle using actual SPY price.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.core.black_scholes import black_scholes_price
from src.core.jump_diffusion import simulate_jump_diffusion


def fetch_historical_prices(ticker='SPY', period='2y'):
    """
    Downloads historical daily closing prices for backtesting.

    Inputs:
        ticker: Yahoo Finance ticker (str)
        period: yfinance period string, e.g. '1y', '2y', '5y'

    Output:
        pd.Series indexed by date with daily Close prices.
        Returns None if fetch fails.
    """
    try:
        tk = yf.Ticker(ticker)
        history = tk.history(period=period)
        if history.empty:
            return None
        return history['Close']
    except Exception:
        return None


def _calculate_rolling_vol(prices, window=30):
    """
    Calculates annualized rolling historical volatility.

    Inputs:
        prices: pd.Series of daily close prices
        window: lookback window in trading days (int)

    Output:
        pd.Series of annualized volatility values
    """
    log_returns = np.log(prices / prices.shift(1))
    rolling_std = log_returns.rolling(window=window).std()
    return rolling_std * np.sqrt(252)


def _get_first_trading_day_per_month(prices):
    """
    Finds the first trading day of each month from a price series.

    Inputs:
        prices: pd.Series indexed by datetime

    Output:
        List of datetime indices representing the first trading day of each month.
    """
    monthly_groups = prices.groupby([prices.index.year, prices.index.month])
    return [group.index[0] for _, group in monthly_groups]


def run_synthetic_backtest(
    ticker='SPY',
    period='2y',
    initial_capital=10000.0,
    option_type='call',
    edge_threshold=0.10,
    risk_free_rate=0.05,
    n_sims=10000,
    jump_intensity=0.1,
    jump_mean=-0.05,
    jump_std=0.03,
    expiry_days=30
):
    """
    Runs the full synthetic backtest over historical data.

    Strategy:
        On the 1st trading day of each month:
        1. Get SPY spot price (S0) and rolling 30-day volatility.
        2. Synthesize the "market" option price using Black-Scholes (ATM call).
        3. Calculate "fair value" using Jump Diffusion Monte Carlo.
        4. If MC price > BS price * (1 + edge_threshold): BUY (invest the premium).
        5. Fast-forward ~30 days and settle using the real SPY price on expiration.

    Inputs:
        ticker: Ticker symbol (str)
        period: Historical data period (str)
        initial_capital: Starting cash (float)
        option_type: 'call' or 'put' (str)
        edge_threshold: Minimum edge % to trigger a trade (float, e.g. 0.10 = 10%)
        risk_free_rate: Annualized risk-free rate (float)
        n_sims: Number of MC simulations per trade (int)
        jump_intensity: Jump Diffusion lambda (float)
        jump_mean: Jump Diffusion mean jump size (float)
        jump_std: Jump Diffusion jump volatility (float)
        expiry_days: Days to expiration for each synthetic option (int)

    Output:
        dict with keys:
            'trades_df'       : pd.DataFrame of all trade details
            'equity_curve'    : pd.DataFrame with date and portfolio value
            'final_value'     : float
            'total_return_pct': float
            'win_rate'        : float
            'total_trades'    : int
            'winning_trades'  : int
    """
    # --- Step 1: Fetch historical data ---
    prices = fetch_historical_prices(ticker, period)
    if prices is None or len(prices) < 60:
        return None

    # Remove timezone info for clean processing
    prices.index = prices.index.tz_localize(None)

    # --- Step 2: Calculate rolling volatility ---
    rolling_vol = _calculate_rolling_vol(prices, window=30)

    # --- Step 3: Identify entry dates (1st trading day of each month) ---
    entry_dates = _get_first_trading_day_per_month(prices)

    # --- Step 4: Run the trading loop ---
    capital = initial_capital
    trades = []
    equity_points = [{'date': prices.index[0], 'value': capital}]

    T = expiry_days / 365.0  # Time to maturity in years

    for entry_date in entry_dates:
        # Get spot price and volatility on entry date
        S0 = float(prices.loc[entry_date])
        sigma = rolling_vol.get(entry_date, np.nan)

        # Skip if volatility is not available (early dates) or invalid
        if np.isnan(sigma) or sigma <= 0.01:
            continue

        sigma = float(min(max(sigma, 0.05), 2.0))  # Clamp to safe range
        K = S0  # ATM option (Strike = Spot)

        # --- Scan BOTH calls and puts, pick the best edge ---
        # Simulate terminal prices once (shared between call/put pricing)
        S_T_sim, _ = simulate_jump_diffusion(
            S0, T, risk_free_rate, sigma, n_sims,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std
        )

        best_edge = -999
        best_type = None
        best_bs = 0
        best_mc = 0

        for otype in ['call', 'put']:
            try:
                bs_p = float(black_scholes_price(S0, K, T, risk_free_rate, sigma, otype))
            except (ValueError, ZeroDivisionError):
                continue
            if bs_p <= 0.01:
                continue

            if otype == 'call':
                pay = np.maximum(S_T_sim - K, 0)
            else:
                pay = np.maximum(K - S_T_sim, 0)

            mc_p = float(np.exp(-risk_free_rate * T) * np.mean(pay))
            edge = (mc_p - bs_p) / bs_p

            if edge > best_edge:
                best_edge = edge
                best_type = otype
                best_bs = bs_p
                best_mc = mc_p

        # --- Decision: Is there enough edge? ---
        if best_type is None or best_edge < edge_threshold:
            equity_points.append({'date': entry_date, 'value': capital})
            continue

        # --- Can we afford this trade? ---
        cost = best_bs
        if cost > capital:
            equity_points.append({'date': entry_date, 'value': capital})
            continue

        # --- Find expiration date (~30 trading days later) ---
        entry_idx = prices.index.get_loc(entry_date)
        expiry_idx = min(entry_idx + expiry_days, len(prices) - 1)
        expiry_date = prices.index[expiry_idx]
        S_T_actual = float(prices.iloc[expiry_idx])

        # --- Settle the trade ---
        if best_type == 'call':
            payoff = max(S_T_actual - K, 0)
        else:
            payoff = max(K - S_T_actual, 0)

        pnl = payoff - cost
        capital = capital + pnl

        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'type': best_type.upper(),
            'spot_price': round(S0, 2),
            'strike': round(K, 2),
            'volatility': round(sigma * 100, 1),
            'bs_premium': round(best_bs, 4),
            'mc_fair_value': round(best_mc, 4),
            'edge_pct': round(best_edge * 100, 1),
            'actual_S_T': round(S_T_actual, 2),
            'payoff': round(payoff, 4),
            'pnl': round(pnl, 4),
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'balance': round(capital, 2)
        })

        equity_points.append({'date': expiry_date, 'value': round(capital, 2)})

    # --- Step 5: Build results ---
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_points)

    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0]) if trades else 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    total_return = ((capital - initial_capital) / initial_capital) * 100

    return {
        'trades_df': trades_df,
        'equity_curve': equity_df,
        'final_value': round(capital, 2),
        'total_return_pct': round(total_return, 2),
        'win_rate': round(win_rate, 1),
        'total_trades': total_trades,
        'winning_trades': winning_trades,
    }
