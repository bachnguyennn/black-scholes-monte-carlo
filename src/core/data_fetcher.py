"""
data_fetcher.py

Fetches live options chain data from Yahoo Finance for arbitrage scanning.
Returns structured DataFrames with Strikes, Bids, Asks, and Implied Volatility
for all available option contracts on a given ticker.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
import asyncio


def get_spot_and_vol(ticker_symbol):
    """
    Fetches the current spot price and historical volatility for a ticker.

    Inputs:
        ticker_symbol: Yahoo Finance ticker string (e.g., '^SPX' for S&P 500 index)

    Output:
        dict with keys: 'spot', 'historical_vol', 'name'
        Returns None if data cannot be fetched.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        history = tk.history(period="1y")
        if history.empty:
            return None

        spot = float(history['Close'].iloc[-1])
        log_returns = np.log(history['Close'] / history['Close'].shift(1)).dropna()
        hist_vol = float(log_returns.std() * np.sqrt(252))
        hist_vol = min(max(hist_vol, 0.05), 2.0)

        name = tk.info.get('longName', ticker_symbol)

        return {
            'spot': spot,
            'historical_vol': hist_vol,
            'name': name,
            'history': history['Close']
        }
    except Exception:
        return None


def get_risk_free_rate():
    """
    Fetches the current risk-free rate from the 13-week Treasury Bill (^IRX).
    Falls back to 5% if the fetch fails.

    Output:
        float: annualized risk-free rate (e.g., 0.05 for 5%)
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            # ^IRX is quoted as a percentage (e.g., 4.5 means 4.5%)
            return float(hist['Close'].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.05  # Default fallback


def get_available_expirations(ticker_symbol):
    """
    Fetches the list of all available options expiration dates for a ticker.
    
    Inputs:
        ticker_symbol: Yahoo Finance ticker string (e.g., '^SPX' for S&P 500 index)
        
    Output:
        list of str in 'YYYY-MM-DD' format, or empty list if failed.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        return list(tk.options)
    except Exception:
        return []

def get_options_chain(ticker_symbol, max_expirations=3, target_days=None, specific_expirations=None):
    """
    Fetches the live options chain for a ticker from Yahoo Finance.

    Inputs:
        ticker_symbol: Yahoo Finance ticker string (e.g., '^SPX' for S&P 500 index)
        max_expirations: Maximum number of expiration dates to fetch (int)
        target_days: If provided, ignores max_expirations and finds the one 
                     expiration date closest to this number of days (int).
        specific_expirations: Optional list of specific 'YYYY-MM-DD' strings to fetch. 
                              If provided, overrides max_expirations and target_days.

    Output:
        pd.DataFrame with columns:
            'type'       : 'call' or 'put'
            'strike'     : float
            'expiration' : str (YYYY-MM-DD)
            'T'          : float (time to maturity in years)
            'bid'        : float
            'ask'        : float
            'mid'        : float (midpoint of bid/ask)
            'market_iv'  : float (implied volatility from exchange)
            'volume'     : int
            'openInterest': int

        Returns empty DataFrame if data cannot be fetched.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        expirations = tk.options  # List of expiration date strings

        if not expirations:
            return pd.DataFrame()

        if specific_expirations:
            # Only keep requested expirations that actually exist
            selected = [d for d in specific_expirations if d in expirations]
        elif target_days is not None:
            # Find the expiration date closest to target_days
            today = datetime.now()
            days_list = []
            for d_str in expirations:
                d_obj = datetime.strptime(d_str, "%Y-%m-%d")
                diff = abs((d_obj - today).days - target_days)
                days_list.append((diff, d_str))
            
            # Sort by difference and take the single best match
            selected = [sorted(days_list)[0][1]]
        else:
            # Take the nearest N expirations (Existing logic)
            selected = list(expirations)[:max_expirations]

        all_rows = []
        today = datetime.now()

        for exp_str in selected:
            chain = tk.option_chain(exp_str)

            # Calculate T (time to maturity in years)
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            # Set time to end of trading day (approx 4 PM)
            exp_datetime = exp_date.replace(hour=16, minute=0)
            
            time_delta = exp_datetime - datetime.now()
            days_to_exp = time_delta.total_seconds() / (24 * 3600)
            
            if days_to_exp <= 0.01: # Skip if less than ~15 mins left
                continue
                
            T = days_to_exp / 365.0

            # Process Calls
            for _, row in chain.calls.iterrows():
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue  # Skip invalid quotes

                all_rows.append({
                    'type': 'call',
                    'strike': float(row['strike']),
                    'expiration': exp_str,
                    'T': T,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2.0,
                    'market_iv': float(row.get('impliedVolatility', 0)),
                    'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                    'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                })

            # Process Puts
            for _, row in chain.puts.iterrows():
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                if bid <= 0 or ask <= 0 or ask < bid:
                    continue

                all_rows.append({
                    'type': 'put',
                    'strike': float(row['strike']),
                    'expiration': exp_str,
                    'T': T,
                    'bid': bid,
                    'ask': ask,
                    'mid': (bid + ask) / 2.0,
                    'market_iv': float(row.get('impliedVolatility', 0)),
                    'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
                    'openInterest': int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0,
                })

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)

        # Filter to liquid options only (volume > 0 or open interest > 10)
        df = df[(df['volume'] > 0) | (df['openInterest'] > 10)]

        return df.reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching options chain: {e}")
        return pd.DataFrame()


async def get_options_chain_async(ticker_symbol, max_expirations=3):
    """
    Asynchronously fetches live options chain using `asyncio.to_thread` wrapped around `yfinance`.
    This prevents blocking the FastAPI event loop while still utilizing yfinance's robust Cookie/Crumb 
    handling to prevent Yahoo Finance 429 Rate Limit blocks.
    """
    return await asyncio.to_thread(get_options_chain, ticker_symbol, max_expirations)
