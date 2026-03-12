"""
jump_diffusion.py

Implements Merton's Jump Diffusion Model for option pricing.
This model extends GBM by adding sudden "jump" events (market crashes).

Mathematical Foundation (Merton 1976):
dS_t = (r - q - λκ) S_t dt + σ S_t dW_t + (e^Y - 1) S_t dN_t

Where:
- r: risk-free rate
- q: dividend yield
- σ: continuous volatility
- λ: crash intensity (expected jumps per year)
- κ: expected jump size E[e^Y - 1] = exp(μ_J + 0.5 * σ_J^2) - 1
- W_t: Brownian motion
- N_t: Poisson process
- Y: Normal distribution of log-jump sizes ~ N(μ_J, σ_J^2)

Reference: Merton (1976) "Option pricing when underlying stock returns are discontinuous"
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

@njit(fastmath=True)
def simulate_jump_diffusion(S0, T, r, sigma, n_sims, 
                            jump_intensity=0.1,  # λ: expected crashes per year
                            jump_mean=-0.05,      # μ_J: average log-jump size
                            jump_std=0.03,        # σ_J: log-jump volatility
                            n_steps=1,            # 1 step for terminal prices
                            q=0.0,
                            seed=-1):           # RNG seed
    """
    Simulates asset prices using Merton's Jump Diffusion Model.
    
    This version includes the risk-neutral jump compensator to ensure
    the process is a martingale under the risk-neutral measure.
    
    Inputs:
        S0: Initial price (float)
        T: Time to maturity in years (float)
        r: Risk-free rate (float)
        sigma: Continuous volatility (float)
        n_sims: Number of simulation paths (int)
        jump_intensity: λ, expected jumps per year (float)
        jump_mean: μ_J, mean of log-jump size (float)
        jump_std: σ_J, std dev of log-jump size (float)
        n_steps: Number of time steps (int)
        q: Continuous dividend yield (float)
        seed: Random seed for reproducibility (int, default -1)
        
    Output:
        S_T: Terminal prices, shape (n_sims,)
        crash_mask: Boolean array indicating which paths experienced crashes
    """
    if seed != -1:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # Jump compensator: kappa = E[exp(Y) - 1] = exp(mu_j + 0.5 * sigma_j^2) - 1
    jump_comp = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    
    # Standard GBM component with compensator
    drift = (r - q - 0.5 * sigma**2 - jump_comp) * dt
    diffusion_std = sigma * np.sqrt(dt)
    
    # Accumulator for prices
    log_S = np.full(n_sims, np.log(S0))
    
    # Track crashes
    crash_occurred = np.zeros(n_sims, dtype=np.bool_)
    
    # Numba loops are fast
    for i in range(n_sims):
        for t in range(n_steps):
            Z = np.random.randn()
            n_jumps = np.random.poisson(jump_intensity * dt)
            
            if n_jumps > 0:
                crash_occurred[i] = True
                jump_component = np.random.randn() * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps
            else:
                jump_component = 0.0
                
            log_S[i] += drift + diffusion_std * Z + jump_component
            
    S_T = np.exp(log_S)
    return S_T, crash_occurred


@njit(fastmath=True)
def simulate_jump_diffusion_paths(S0, T, r, sigma, n_paths,
                                   jump_intensity=0.1,
                                   jump_mean=-0.05,
                                   jump_std=0.03,
                                   n_steps=100,
                                   q=0.0,
                                   seed=-1):
    """
    Simulates full price paths using Merton's Jump Diffusion Model.
    
    Returns:
        paths: Array of shape (n_paths, n_steps + 1)
        crash_mask: Boolean array indicating which paths crashed
    """
    if seed != -1:
        np.random.seed(seed)
        
    dt = T / n_steps
    
    # Jump compensator
    jump_comp = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    
    drift = (r - q - 0.5 * sigma**2 - jump_comp) * dt
    diffusion_std = sigma * np.sqrt(dt)
    
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    crash_occurred = np.zeros(n_paths, dtype=np.bool_)
    
    for i in range(n_paths):
        current_log_S = np.log(S0)
        for t in range(1, n_steps + 1):
            Z = np.random.randn()
            n_jumps = np.random.poisson(jump_intensity * dt)
            
            if n_jumps > 0:
                crash_occurred[i] = True
                jump_component = np.random.randn() * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps
            else:
                jump_component = 0.0
            
            current_log_S += drift + diffusion_std * Z + jump_component
            paths[i, t] = np.exp(current_log_S)
    
    return paths, crash_occurred
