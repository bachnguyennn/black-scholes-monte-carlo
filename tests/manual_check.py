
import sys
import os
import numpy as np

# Add src to python path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm

def manual_check():
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    n_paths = 100000
    n_steps = 1  # For European option, 1 step is sufficient for end-simulation
    
    print(f"--- Manual Verification ---")
    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Simulation: n_paths={n_paths}, n_steps={n_steps}")
    
    # 1. Black-Scholes Price
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type='call')
    print(f"\n[Black-Scholes] Price: {bs_price:.6f}")
    
    # 2. Monte Carlo Price
    # Simulate terminal prices
    # Note: simulate_gbm now takes (S0, T, r, sigma, n_sims)
    S_T = simulate_gbm(S0, T, r, sigma, n_paths)
    
    # Calculate payoffs: max(S_T - K, 0)
    payoffs = np.maximum(S_T - K, 0)
    
    # Discount back to present value using mean payoff
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    
    # Calculate Standard Error
    std_dev = np.std(payoffs)
    standard_error = (std_dev / np.sqrt(n_paths)) * np.exp(-r * T)
    
    print(f"[Monte Carlo]   Price: {mc_price:.6f}")
    print(f"[Monte Carlo]   SE:    {standard_error:.6f}")
    
    # 3. Validation
    diff = abs(bs_price - mc_price)
    confidence_interval_low = mc_price - 1.96 * standard_error
    confidence_interval_high = mc_price + 1.96 * standard_error
    
    print(f"\nDifference: {diff:.6f}")
    print(f"95% Confidence Interval: [{confidence_interval_low:.6f}, {confidence_interval_high:.6f}]")
    
    if confidence_interval_low <= bs_price <= confidence_interval_high:
        print("\nSUCCESS: Black-Scholes price is within the 95% Confidence Interval.")
    else:
        print("\nWARNING: Black-Scholes price is OUTSIDE the 95% Confidence Interval.")
        
if __name__ == "__main__":
    manual_check()
