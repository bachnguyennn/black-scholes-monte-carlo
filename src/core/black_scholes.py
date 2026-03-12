"""
black_scholes.py

Implements the Black-Scholes analytical formula for European options.
Formula:
C = S0 * N(d1) - K * exp(-rT) * N(d2)
d1 = (ln(S0/K) + (r + sigma^2/2)T) / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
"""

import numpy as np
from scipy.stats import norm

def black_scholes_price(S0, K, T, r, sigma, option_type='call', q=0.0):
    """
    Calculates the analytical Black-Scholes price for a European option.
    
    Formula:
    C = S0 * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)
    P = K * exp(-rT) * N(-d2) - S0 * exp(-qT) * N(-d1)
    
    where:
    d1 = (ln(S0/K) + (r + sigma^2/2)T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Inputs:
        S0: Current asset price (float or np.ndarray)
        K: Strike price (float or np.ndarray)
        T: Time to maturity (float or np.ndarray)
        r: Risk-free rate (float or np.ndarray)
        sigma: Volatility (float or np.ndarray)
        option_type: 'call' or 'put' (str)
        q: Continuous dividend yield (float, default 0.0)
        
    Output:
        price: NumPy array or scalar of option prices
    """
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    
    if option_type == 'call':
        price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    return price
