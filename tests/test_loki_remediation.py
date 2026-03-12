import pytest
import numpy as np
from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.heston_model import price_option_heston_fourier, price_option_heston

def test_jd_martingale():
    """Verify the jump-diffusion drift compensator preserves the martingale property."""
    S0 = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.2
    jump_intensity = 0.5
    jump_mean = -0.1
    jump_std = 0.1
    n_sims = 200000  # High sample for precision
    
    S_T, _ = simulate_jump_diffusion(S0, T, r, sigma, n_sims, jump_intensity, jump_mean, jump_std, q=q)
    
    # E[S_T] should be S0 * exp((r - q) * T)
    expected_E_ST = S0 * np.exp((r - q) * T)
    actual_E_ST = np.mean(S_T)
    
    # 99.7% confidence interval (3 sigma)
    std_err = np.std(S_T) / np.sqrt(n_sims)
    
    assert abs(actual_E_ST - expected_E_ST) < 3 * std_err, \
        f"Martingale property violated: expected {expected_E_ST:.4f}, got {actual_E_ST:.4f} (err: {abs(actual_E_ST - expected_E_ST):.4f})"

def test_heston_fourier_vs_mc():
    """Verify Heston Fourier pricing matches Monte Carlo within tolerance."""
    S0 = 100.0
    K = 105.0
    T = 0.5
    r = 0.03
    q = 0.01
    V0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    
    fourier_call = price_option_heston_fourier(S0, K, T, r, V0, kappa, theta, xi, rho, option_type='call', q=q)
    mc_res = price_option_heston(S0, K, T, r, V0, kappa, theta, xi, rho, option_type='call', n_sims=100000, q=q)
    
    mc_price = mc_res['price']
    std_err = mc_res['std_error']
    
    # Fourier should be within 3 std errors of MC
    assert abs(fourier_call - mc_price) < 3 * std_err, \
        f"Heston Fourier mismatch: Fourier {fourier_call:.4f}, MC {mc_price:.4f} ± {3*std_err:.4f}"

def test_heston_put_call_parity():
    """Verify put-call parity for Heston Fourier pricing."""
    S0 = 100.0
    K = 95.0
    T = 1.0
    r = 0.05
    q = 0.02
    V0 = 0.04
    kappa = 1.5
    theta = 0.04
    xi = 0.2
    rho = -0.6
    
    C = price_option_heston_fourier(S0, K, T, r, V0, kappa, theta, xi, rho, 'call', q=q)
    P = price_option_heston_fourier(S0, K, T, r, V0, kappa, theta, xi, rho, 'put', q=q)
    
    lhs = C - P
    rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    
    assert abs(lhs - rhs) < 1e-6, f"Heston Put-Call Parity violated: LHS {lhs:.6f}, RHS {rhs:.6f}"

def test_jd_put_call_parity():
    """Verify put-call parity for jump-diffusion Monte Carlo pricing."""
    S0 = 100.0
    K = 100.0
    T = 0.25
    r = 0.04
    q = 0.0
    sigma = 0.2
    jump_intensity = 0.2
    jump_mean = -0.1
    jump_std = 0.05
    n_sims = 100000
    
    # Estimate call and put prices from the same simulated terminal prices.
    S_T, _ = simulate_jump_diffusion(S0, T, r, sigma, n_sims, jump_intensity, jump_mean, jump_std, q=q)
    discount = np.exp(-r * T)
    C = discount * np.mean(np.maximum(S_T - K, 0))
    P = discount * np.mean(np.maximum(K - S_T, 0))
    
    lhs = C - P
    rhs = S0 * np.exp(-q * T) - K * np.exp(-r * T)
    
    # Tolerance based on Monte Carlo error
    std_err_C = discount * np.std(np.maximum(S_T - K, 0)) / np.sqrt(n_sims)
    std_err_P = discount * np.std(np.maximum(K - S_T, 0)) / np.sqrt(n_sims)
    combined_err = np.sqrt(std_err_C**2 + std_err_P**2)
    
    assert abs(lhs - rhs) < 3 * combined_err, f"JD Put-Call Parity violated: LHS {lhs:.4f}, RHS {rhs:.4f}"
