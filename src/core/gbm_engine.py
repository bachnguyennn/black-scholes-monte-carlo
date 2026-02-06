"""
gbm_engine.py

Simulates Geometric Brownian Motion paths.
Formula:
S_T = S_0 * exp((r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z)
where Z ~ N(0, 1)
"""

import numpy as np

def simulate_gbm(S0, T, r, sigma, n_sims):
    """
    Simulates terminal asset prices using Geometric Brownian Motion.
    
    Inputs:
        S0: Initial price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        n_sims: Number of simulations (int)
        
    Output:
        S_T: NumPy array of shape (n_sims,) representing terminal prices
    """
    # Generate random standard normals Z ~ N(0, 1)
    Z = np.random.standard_normal(n_sims)
    
    # Calculate terminal prices directly using the exact solution
    # S_T = S_0 * exp( (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z )
    
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    
    S_T = S0 * np.exp(drift + diffusion)
    
    return S_T
