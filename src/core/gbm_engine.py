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
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")

    # Generate random standard normals Z ~ N(0, 1)
    Z = np.random.standard_normal(n_sims)
    
    # Calculate terminal prices directly using the exact solution
    # S_T = S_0 * exp( (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z )
    
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    
    S_T = S0 * np.exp(drift + diffusion)
    
    return S_T

def simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps=100):
    """
    Simulates full price paths using Geometric Brownian Motion.
     useful for visualization.
    
    Inputs:
        S0: Initial price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        n_paths: Number of simulation paths (int)
        n_steps: Number of time steps (int)
        
    Output:
        paths: NumPy array of shape (n_paths, n_steps + 1)
    """
    if T <= 0:
        raise ValueError("Time to maturity (T) must be positive.")
    if sigma <= 0:
        raise ValueError("Volatility (sigma) must be positive.")

    dt = T / n_steps
    # Z ~ N(0, 1)
    Z = np.random.standard_normal((n_paths, n_steps))
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    log_returns = drift + diffusion
    
    # Cumulative sum for paths
    cumulative_log_returns = np.zeros((n_paths, n_steps + 1))
    cumulative_log_returns[:, 1:] = np.cumsum(log_returns, axis=1)
    
    return S0 * np.exp(cumulative_log_returns)


def calculate_profit_probability(S_T, K, premium, option_type='call'):
    """
    Calculates the probability that the option trade will be profitable.
    
    For a Call: Profitable if S_T > K + premium
    For a Put: Profitable if S_T < K - premium
    
    Inputs:
        S_T: Array of terminal prices from simulation (np.ndarray)
        K: Strike price (float)
        premium: Option premium paid (float)
        option_type: 'call' or 'put' (str)
        
    Output:
        probability: Float between 0.0 and 1.0
    """
    if option_type.lower() == 'call':
        # Call is profitable if stock price exceeds strike + premium
        breakeven = K + premium
        profitable_paths = S_T > breakeven
    else:  # put
        # Put is profitable if stock price falls below strike - premium
        breakeven = K - premium
        profitable_paths = S_T < breakeven
    
    return np.mean(profitable_paths)


def calculate_breakeven_price(K, premium, option_type='call'):
    """
    Calculates the stock price needed at expiration to break even.
    
    This is the price where the intrinsic value equals the premium paid.
    
    Inputs:
        K: Strike price (float)
        premium: Option premium paid (float)
        option_type: 'call' or 'put' (str)
        
    Output:
        breakeven_price: Float
    """
    if option_type.lower() == 'call':
        # For Call: Need stock to be at Strike + Premium
        return K + premium
    else:  # put
        # For Put: Need stock to be at Strike - Premium
        return K - premium
