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
from scipy.stats import norm

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def merton_jump_price(S0, K, T, r, sigma, option_type='call',
                      jump_intensity=0.1, jump_mean=-0.05, jump_std=0.03,
                      q=0.0, n_terms=60):
    """
    Analytic Merton (1976) jump-diffusion price.

    Conditional on exactly n jumps, the terminal price is lognormal, so the
    option price is a Poisson-weighted sum of Black-Scholes (Black-76 forward)
    prices:

        C = sum_{n>=0} e^{-lam*T} (lam*T)^n / n! * BS_forward(F_n, K, T, r, sigma_n)

    where, matching the compensated risk-neutral drift used by the Monte Carlo
    engine (r - q - lam*k, with k = E[e^Y - 1]):

        m       = jump_mean + 0.5*jump_std^2          (= ln E[e^Y])
        k       = e^m - 1
        sigma_n^2 = sigma^2 * T + n * jump_std^2       (total variance)
        F_n     = S0 * exp((r - q - lam*k) * T + n*m)  (forward given n jumps)

    Summed over n, the forwards satisfy E[F_n] = S0*e^{(r-q)T}, so the model is
    a martingale. This is exact (to the truncation of the Poisson series) and
    serves as the analytic ground truth for the jump-diffusion Monte Carlo.

    Inputs mirror simulate_jump_diffusion; n_terms caps the Poisson series.

    Output:
        float: option price
    """
    lam = jump_intensity
    disc = np.exp(-r * T)

    if T <= 0 or sigma <= 0:
        intrinsic = max(S0 - K, 0.0) if option_type == 'call' else max(K - S0, 0.0)
        return float(intrinsic)

    m = jump_mean + 0.5 * jump_std ** 2   # ln E[e^Y]
    k = np.exp(m) - 1.0
    lamT = lam * T

    price = 0.0
    log_w = -lamT                          # log Poisson weight, n = 0
    for n in range(n_terms):
        if n > 0:
            if lamT <= 0.0:
                break                      # no jumps: only the n = 0 term
            log_w += np.log(lamT) - np.log(n)
        w = np.exp(log_w)
        if n > lamT and w < 1e-14:
            break                          # tail weight negligible

        var_n = sigma ** 2 * T + n * jump_std ** 2
        sd_n = np.sqrt(var_n)
        F_n = S0 * np.exp((r - q - lam * k) * T + n * m)

        if sd_n < 1e-12:
            payoff = max(F_n - K, 0.0) if option_type == 'call' else max(K - F_n, 0.0)
            term = disc * payoff
        else:
            d1 = (np.log(F_n / K) + 0.5 * var_n) / sd_n
            d2 = d1 - sd_n
            if option_type == 'call':
                term = disc * (F_n * norm.cdf(d1) - K * norm.cdf(d2))
            else:
                term = disc * (K * norm.cdf(-d2) - F_n * norm.cdf(-d1))

        price += w * term

    return float(max(price, 0.0))

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
def simulate_jump_diffusion_antithetic(S0, T, r, sigma, n_pairs,
                                       jump_intensity=0.1,
                                       jump_mean=-0.05,
                                       jump_std=0.03,
                                       n_steps=1,
                                       q=0.0,
                                       seed=-1):
    """
    Simulates terminal prices in antithetic pairs for variance reduction.

    For each of the n_pairs draws, the Gaussian shocks (both the diffusion
    Brownian increment and the jump-size normals) are used with sign +Z and
    -Z, producing two negatively-correlated terminal prices from one set of
    random numbers. The Poisson jump *counts* are shared across the pair
    (counts cannot be antithetically mirrored), so the reduction comes from
    the dominant continuous component and the jump magnitudes.

    The same Brownian path is also evolved with no jumps and no jump
    compensator to give a pure-GBM terminal price. That price makes an
    excellent control variate: it is strongly correlated with the
    jump-diffusion payoff and its discounted expectation is exactly the
    Black-Scholes value.

    Returns:
        S_T_plus  : jump-diffusion terminal prices using +Z, shape (n_pairs,)
        S_T_minus : jump-diffusion terminal prices using -Z, shape (n_pairs,)
        S_bs_plus : pure-GBM terminal prices using +Z (control), shape (n_pairs,)
        S_bs_minus: pure-GBM terminal prices using -Z (control), shape (n_pairs,)
    """
    if seed != -1:
        np.random.seed(seed)

    dt = T / n_steps
    jump_comp = jump_intensity * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
    gbm_drift = (r - q - 0.5 * sigma**2) * dt          # pure GBM (control)
    jd_drift = gbm_drift - jump_comp * dt              # jump-compensated
    diffusion_std = sigma * np.sqrt(dt)

    S_T_plus = np.empty(n_pairs)
    S_T_minus = np.empty(n_pairs)
    S_bs_plus = np.empty(n_pairs)
    S_bs_minus = np.empty(n_pairs)

    for i in range(n_pairs):
        log_jd_p = np.log(S0)
        log_jd_m = np.log(S0)
        log_bs_p = np.log(S0)
        log_bs_m = np.log(S0)
        for t in range(n_steps):
            Z = np.random.randn()
            n_jumps = np.random.poisson(jump_intensity * dt)
            if n_jumps > 0:
                Zj = np.random.randn()
                mag = Zj * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps
                mag_anti = -Zj * jump_std * np.sqrt(n_jumps) + jump_mean * n_jumps
            else:
                mag = 0.0
                mag_anti = 0.0

            diff_p = diffusion_std * Z
            diff_m = -diffusion_std * Z
            log_jd_p += jd_drift + diff_p + mag
            log_jd_m += jd_drift + diff_m + mag_anti
            log_bs_p += gbm_drift + diff_p
            log_bs_m += gbm_drift + diff_m

        S_T_plus[i] = np.exp(log_jd_p)
        S_T_minus[i] = np.exp(log_jd_m)
        S_bs_plus[i] = np.exp(log_bs_p)
        S_bs_minus[i] = np.exp(log_bs_m)

    return S_T_plus, S_T_minus, S_bs_plus, S_bs_minus


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
