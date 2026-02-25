"""
greeks.py

Calculates option Greeks using finite difference methods.
Greeks measure the sensitivity of option prices to various parameters.

For Jump Diffusion models, analytical Greeks are not available,
so we use numerical approximation via finite differences.
"""

import numpy as np
from .jump_diffusion import simulate_jump_diffusion
from .gbm_engine import simulate_gbm


def price_option(S0, K, T, r, sigma, option_type='call', model='gbm', n_sims=50000, **kwargs):
    """
    Price an option using Monte Carlo simulation.
    
    Inputs:
        S0: Current asset price (float)
        K: Strike price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        model: 'gbm' or 'jump_diffusion' (str)
        n_sims: Number of simulations (int)
        **kwargs: Additional parameters for jump_diffusion model
        
    Output:
        price: Option price (float)
    """
    # Simulate terminal prices
    if model == 'gbm':
        S_T = simulate_gbm(S0, T, r, sigma, n_sims)
    elif model == 'jump_diffusion':
        S_T, _ = simulate_jump_diffusion(S0, T, r, sigma, n_sims, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    # Calculate payoffs
    if option_type.lower() == 'call':
        payoffs = np.maximum(S_T - K, 0)
    elif option_type.lower() == 'put':
        payoffs = np.maximum(K - S_T, 0)
    else:
        raise ValueError(f"Unknown option_type: {option_type}")
    
    # Discount to present value
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price


def calculate_delta(S0, K, T, r, sigma, option_type='call', 
                    model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate Delta using central finite difference method.
    
    Delta measures the rate of change of option price with respect to 
    the underlying asset price.
    
    Formula: Δ = (V(S+h) - V(S-h)) / (2h)
    
    Inputs:
        S0: Current asset price (float)
        K: Strike price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        model: 'gbm' or 'jump_diffusion' (str)
        h: Step size for finite difference (float, default 0.01 = 1%)
        n_sims: Number of simulations (int)
        **kwargs: Additional parameters for jump_diffusion model
        
    Output:
        delta: Delta value (float)
        
    Interpretation:
        - For calls: 0 < Δ < 1 (typically 0.5 for ATM)
        - For puts: -1 < Δ < 0 (typically -0.5 for ATM)
        - Δ = 0.7 means option price increases by $0.70 for $1 stock increase
    """
    # Price at S0 + h
    V_up = price_option(S0 * (1 + h), K, T, r, sigma, option_type, model, n_sims, **kwargs)
    
    # Price at S0 - h
    V_down = price_option(S0 * (1 - h), K, T, r, sigma, option_type, model, n_sims, **kwargs)
    
    # Central difference
    delta = (V_up - V_down) / (2 * S0 * h)
    
    return delta


def calculate_vega(S0, K, T, r, sigma, option_type='call',
                   model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate Vega using central finite difference method.
    
    Vega measures the rate of change of option price with respect to volatility.
    
    Formula: ν = (V(σ+h) - V(σ-h)) / (2h)
    
    Inputs:
        S0: Current asset price (float)
        K: Strike price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        model: 'gbm' or 'jump_diffusion' (str)
        h: Step size for finite difference (float, default 0.01 = 1%)
        n_sims: Number of simulations (int)
        **kwargs: Additional parameters for jump_diffusion model
        
    Output:
        vega: Vega value (float)
        
    Interpretation:
        - Vega is always positive for both calls and puts
        - Higher for ATM options, lower for deep ITM/OTM
        - Vega = 0.25 means option price increases by $0.25 for 1% vol increase
    """
    # Price at σ + h
    V_up = price_option(S0, K, T, r, sigma * (1 + h), option_type, model, n_sims, **kwargs)
    
    # Price at σ - h
    V_down = price_option(S0, K, T, r, sigma * (1 - h), option_type, model, n_sims, **kwargs)
    
    # Central difference (divide by sigma*h to get per-unit change)
    vega = (V_up - V_down) / (2 * sigma * h)
    
    return vega


def calculate_gamma(S0, K, T, r, sigma, option_type='call',
                    model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate Gamma using central finite difference method.
    
    Gamma measures the rate of change of Delta with respect to the underlying price.
    It represents the curvature of the option price curve.
    
    Formula: Γ = (V(S+h) - 2V(S) + V(S-h)) / h²
    
    Inputs:
        S0: Current asset price (float)
        K: Strike price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        model: 'gbm' or 'jump_diffusion' (str)
        h: Step size for finite difference (float, default 0.01 = 1%)
        n_sims: Number of simulations (int)
        **kwargs: Additional parameters for jump_diffusion model
        
    Output:
        gamma: Gamma value (float)
        
    Interpretation:
        - Gamma is always positive for both calls and puts
        - Highest for ATM options near expiration
        - Measures how fast Delta changes as stock price moves
    """
    # Price at three points
    V_up = price_option(S0 * (1 + h), K, T, r, sigma, option_type, model, n_sims, **kwargs)
    V_mid = price_option(S0, K, T, r, sigma, option_type, model, n_sims, **kwargs)
    V_down = price_option(S0 * (1 - h), K, T, r, sigma, option_type, model, n_sims, **kwargs)
    
    # Second derivative
    gamma = (V_up - 2 * V_mid + V_down) / ((S0 * h) ** 2)
    
    return gamma


def calculate_all_greeks(S0, K, T, r, sigma, option_type='call',
                         model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate all Greeks at once for efficiency.
    
    Returns a dictionary with Delta, Vega, and Gamma.
    
    Inputs:
        S0: Current asset price (float)
        K: Strike price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        model: 'gbm' or 'jump_diffusion' (str)
        h: Step size for finite difference (float)
        n_sims: Number of simulations (int)
        **kwargs: Additional parameters for jump_diffusion model
        
    Output:
        greeks: Dictionary with keys 'price', 'delta', 'vega', 'gamma'
    """
    # Calculate price and all Greeks
    price = price_option(S0, K, T, r, sigma, option_type, model, n_sims, **kwargs)
    delta = calculate_delta(S0, K, T, r, sigma, option_type, model, h, n_sims, **kwargs)
    vega = calculate_vega(S0, K, T, r, sigma, option_type, model, h, n_sims, **kwargs)
    gamma = calculate_gamma(S0, K, T, r, sigma, option_type, model, h, n_sims, **kwargs)
    
    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'gamma': gamma
    }
