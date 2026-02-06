
import sys
import os
import numpy as np
from datetime import datetime

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm

def fetch_sp500_data():
    """
    Attempts to fetch real-time S&P 500 (SPY or ^GSPC) data.
    Returns: (S0, sigma)
    """
    try:
        import yfinance as yf
        print("Successfully imported yfinance. Fetching real-time data...")
        
        # Use SPY (ETF) as proxy for S&P 500 price/volatility availability
        ticker = yf.Ticker("SPY")
        
        # Get current price
        history = ticker.history(period="1d")
        if history.empty:
            raise ValueError("No data found for SPY")
            
        S0 = history['Close'].iloc[-1]
        
        # Estimate volatility (3 month historical)
        hist_3mo = ticker.history(period="3mo")
        returns = np.log(hist_3mo['Close'] / hist_3mo['Close'].shift(1))
        sigma = returns.std() * np.sqrt(252)
        
        print(f"Fetched Data: SPY Price={S0:.2f}, Volatility={sigma:.2%}")
        return S0, sigma
        
    except ImportError:
        print("Note: 'yfinance' not installed. Using hardcoded fallback values.")
    except Exception as e:
        print(f"Note: Data fetch failed ({e}). Using hardcoded fallback values.")
        
    # Fallback: Approximate values for S&P 500 (SPY proxy) as of late 2024/early 2025
    S0_fallback = 580.00  # Approx SPY price
    sigma_fallback = 0.12 # Approx VIX/Vol
    
    print(f"Fallback Data: SPY Price={S0_fallback:.2f}, Volatility={sigma_fallback:.2%}")
    return S0_fallback, sigma_fallback

def run_demo():
    print("--- S&P 500 (SPY) Option Pricing Demo ---")
    
    # 1. Get Data
    S0, sigma = fetch_sp500_data()
    
    # 2. Define Option Parameters
    # Let's price a 3-month At-The-Money (ATM) Call
    K = S0          # ATM
    T = 0.25        # 3 months
    r = 0.045       # Risk-free rate approc 4.5%
    
    n_sims = 100_000
    
    print(f"\nOption Parameters:")
    print(f"Type: European Call")
    print(f"Strike (K): {K:.2f}")
    print(f"Time (T):   {T:.2f} years")
    print(f"Rate (r):   {r:.2%}")
    print(f"Sims (N):   {n_sims}")
    
    # 3. Analytic Price
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type='call')
    
    # 4. Monte Carlo Price
    # simulate_gbm(S0, T, r, sigma, n_sims)
    S_T = simulate_gbm(S0, T, r, sigma, n_sims)
    payoffs = np.maximum(S_T - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    
    # 5. Output
    print(f"\n--- Results ---")
    print(f"Black-Scholes Price: ${bs_price:.4f}")
    print(f"Monte Carlo Price:   ${mc_price:.4f}")
    
    diff = bs_price - mc_price
    print(f"Difference:          ${diff:.4f} ({diff/bs_price:.4%})")

if __name__ == "__main__":
    run_demo()
