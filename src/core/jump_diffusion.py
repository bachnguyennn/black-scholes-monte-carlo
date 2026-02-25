"""
jump_diffusion.py

Implements Merton's Jump Diffusion Model for option pricing.
This model extends GBM by adding sudden "jump" events (market crashes).

Mathematical Foundation:
dS_t = μ S_t dt + σ S_t dW_t + J_t S_t dN_t

Where:
- μ: drift (risk-free rate)
- σ: continuous volatility
- W_t: Brownian motion (normal market fluctuations)
- N_t: Poisson process (random crash events)
- J_t: jump size distribution (how bad the crash is)

Reference: Merton (1976) "Option pricing when underlying stock returns are discontinuous"
"""

import numpy as np

def simulate_jump_diffusion(S0, T, r, sigma, n_sims, 
                            jump_intensity=0.1,  # λ: expected crashes per year
                            jump_mean=-0.05,      # μ_J: average crash size (-5%)
                            jump_std=0.03,        # σ_J: crash volatility
                            n_steps=1):           # 1 step for terminal prices
    """
    Simulates asset prices using Jump Diffusion Model (Merton 1976).
    
    This extends standard GBM by modeling rare "crash" events that occur
    randomly according to a Poisson process.
    
    Inputs:
        S0: Initial price (float)
        T: Time to maturity in years (float)
        r: Risk-free rate (float)
        sigma: Continuous volatility (float)
        n_sims: Number of simulation paths (int)
        jump_intensity: Expected number of jumps per year (float)
        jump_mean: Average jump size as percentage (float, negative for crashes)
        jump_std: Standard deviation of jump sizes (float)
        n_steps: Number of time steps for path simulation (int)
        
    Output:
        S_T: Terminal prices, shape (n_sims,)
        crash_mask: Boolean array indicating which paths experienced crashes
    """
    dt = T / n_steps
    
    # Initialize price paths
    S = np.zeros((n_sims, n_steps + 1))
    S[:, 0] = S0
    
    # Track which simulations experienced crashes
    crash_occurred = np.zeros(n_sims, dtype=bool)
    
    for t in range(1, n_steps + 1):
        # Standard GBM component
        Z = np.random.standard_normal(n_sims)
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        # Jump component (Poisson process)
        # Number of jumps in this time step for each path
        n_jumps = np.random.poisson(jump_intensity * dt, n_sims)
        
        # Vectorized jump component calculation
        # Key insight: Sum of N normal variables ~ N(N*μ, sqrt(N)*σ)
        # This eliminates the Python loop for massive speedup
        has_jump = n_jumps > 0
        crash_occurred |= has_jump
        
        # For paths with jumps, sample from scaled normal distribution
        # sum(n_jumps[i] normals) ~ N(n_jumps*μ_J, sqrt(n_jumps)*σ_J)
        jump_component = np.where(
            has_jump,
            np.random.normal(0, 1, n_sims) * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps,
            0
        )
        
        # Combine GBM + Jumps
        log_return = drift + diffusion + jump_component
        S[:, t] = S[:, t-1] * np.exp(log_return)
    
    return S[:, -1], crash_occurred


def simulate_jump_diffusion_paths(S0, T, r, sigma, n_paths,
                                   jump_intensity=0.1,
                                   jump_mean=-0.05,
                                   jump_std=0.03,
                                   n_steps=100):
    """
    Simulates full price paths (not just terminal values) for visualization.
    
    Returns:
        paths: Array of shape (n_paths, n_steps + 1)
        crash_mask: Boolean array indicating which paths crashed
    """
    dt = T / n_steps
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    crash_occurred = np.zeros(n_paths, dtype=bool)
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        
        n_jumps = np.random.poisson(jump_intensity * dt, n_paths)
        
        # Vectorized jump component (same optimization as above)
        has_jump = n_jumps > 0
        crash_occurred |= has_jump
        
        jump_component = np.where(
            has_jump,
            np.random.normal(0, 1, n_paths) * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps,
            0
        )
        
        log_return = drift + diffusion + jump_component
        paths[:, t] = paths[:, t-1] * np.exp(log_return)
    
    return paths, crash_occurred
