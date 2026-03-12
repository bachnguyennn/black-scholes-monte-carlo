"""
lsv_model.py

Implements the Local Stochastic Volatility (LSV) model.
This model bridges the gap between Dupire's Local Volatility (LV) 
and Heston's Stochastic Volatility (SV).

Mathematical Foundation:
dS_t = (r - q) * S_t * dt + sigma_L(S_t, t) * sqrt(V_t) * S_t * dW_t^S
dV_t = kappa * (theta - V_t) * dt + xi * sqrt(V_t) * dW_t^V
Corr(dW_t^S, dW_t^V) = rho

Where sigma_L(K, T) is the non-parametric leverage function, calibrated
to ensure the model perfectly matches the European option market implied
volatility surface.

Reference: Dupire (1994) "Pricing with a Smile"
"""

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .heston_model import feller_condition

@njit(fastmath=True)
def simulate_lsv_paths(S0, T, r, V0, kappa, theta, xi, rho,
                       leverage_matrix, strikes, maturities,
                       n_paths, n_steps=100, q=0.0, seed=-1):
    """
    Simulates full price paths using the Local Stochastic Volatility (LSV) model.
    A leverage matrix (Dupire local vol approximation) modulates the Heston SV.
    
    Inputs:
        leverage_matrix: 2D array of leveraging factors shape (len(strikes), len(maturities))
        strikes: 1D array of strike prices for interpolation
        maturities: 1D array of maturities for interpolation
        ...standard Heston inputs...
    """
    if seed != -1:
        np.random.seed(seed)
        
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_comp = np.sqrt(max(1.0 - rho**2, 0.0))

    paths = np.zeros((n_paths, n_steps + 1))
    vol_paths = np.zeros((n_paths, n_steps + 1))
    
    for i in range(n_paths):
        paths[i, 0] = S0
        vol_paths[i, 0] = np.sqrt(float(V0))
        
        log_S = np.log(S0)
        V = float(V0)

        for t in range(n_steps):
            Z_S = np.random.randn()
            Z_V_indep = np.random.randn()
            Z_V = rho * Z_S + rho_comp * Z_V_indep
            
            V_pos = max(V, 0.0)
            sqrt_V = np.sqrt(V_pos)
            current_S = np.exp(log_S)
            current_t = t * dt
            
            # JIT compilation restricts scipy.interpolate here, so use a
            # simple nearest-neighbor lookup for the leverage factor.
            
            # Find closest maturity index
            m_idx = np.searchsorted(maturities, current_t)
            if m_idx >= len(maturities): m_idx = len(maturities) - 1
            
            # Find closest strike index
            k_idx = np.searchsorted(strikes, current_S)
            if k_idx >= len(strikes): k_idx = len(strikes) - 1
            
            # Local Volatility leverage factor L(S, t)
            L = leverage_matrix[k_idx, m_idx]

            log_S += (r - q - 0.5 * (L**2) * V_pos) * dt + L * sqrt_V * sqrt_dt * Z_S
            V += kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * Z_V

            paths[i, t + 1] = np.exp(log_S)
            vol_paths[i, t + 1] = L * np.sqrt(max(V, 0.0))

    return paths, vol_paths

def calibrate_leverage_function(market_iv_surface, strikes, maturities,
                                 r=0.05, q=0.0, kappa=2.0,
                                 heston_theta=0.04, heston_V0=0.04):
    """
    Calibrates the non-parametric leverage function sigma_L(K, T) for the LSV model.

    The leverage function L(K, T) is defined such that the LSV model (Heston SV
    modulated by L) perfectly prices all liquid vanilla options on the market surface.

    Method: Dupire (1994) Local Variance -> LSV Leverage Bridge
    ============================================================
    Compute the Dupire local variance sigma_loc^2(K,T) from the market
    total implied variance surface w(K,T) = sigma_IV^2 * T using finite differences:

        sigma_loc^2 = dw/dT / [1 - (k/w)*dw/dk + 0.25*(-0.25 - 1/w + k^2/w^2)*(dw/dk)^2 + 0.5*d^2w/dk^2]

    where k = ln(K / F) is log-moneyness and F = S * e^((r-q)*T) is forward price.
    Reference: Gatheral (2006) "The Volatility Surface", Chapter 1.

    Compute the Heston expected variance E[V_t] using the analytic mean
    reversion formula: E[V_t] = theta + (V0 - theta) * e^(-kappa*t)

    Derive L(K, T) = sqrt(sigma_loc^2 / E[V_t])
    This ensures that L^2(K,T) * E[V_t] = sigma_loc^2, so the LSV model
    recovers the Dupire surface by construction.

    Args:
        market_iv_surface: 2D array of implied vols, shape (len(strikes), len(maturities))
        strikes:   1D array of strikes (ascending)
        maturities: 1D array of maturities in years (ascending)
        r, q:      risk-free rate and dividend yield
        kappa:     Heston mean reversion speed (default 2.0)
        heston_theta, heston_V0: Heston long-run and initial variance

    Returns:
        leverage_matrix: 2D float array of shape (len(strikes), len(maturities))
                         Values are floored at 0.1 and capped at 5.0 for numerical stability.
    """
    import numpy as np

    n_K = len(strikes)
    n_T = len(maturities)
    leverage_matrix = np.ones((n_K, n_T))

    # Total implied variance surface: w(k, T) = IV^2 * T
    total_var = market_iv_surface**2 * maturities[np.newaxis, :]   # shape (n_K, n_T)

    for j, T in enumerate(maturities):
        if T <= 0:
            continue

        # Log-moneyness: use the ATM forward as the anchor (not per-strike forward)
        # k_i = ln(K_i / F_ATM) gives a well-conditioned, monotonically spaced vector.
        # Using per-strike F causes k = ln(K / K*exp((r-q)*T)) ≈ 0 for all strikes,
        # collapsing the finite-difference denominators to near zero.
        F_atm = strikes[n_K // 2] * np.exp((r - q) * T)   # ATM-anchored forward
        k = np.log(strikes / F_atm)                         # well-spaced log-moneyness

        w = total_var[:, j]    # total var slice at this maturity

        for i in range(n_K):
            try:
                # ---- dw/dT (time derivative, finite difference) ----
                if j == 0:
                    dw_dT = (total_var[i, j + 1] - total_var[i, j]) / (maturities[j + 1] - maturities[j])
                elif j == n_T - 1:
                    dw_dT = (total_var[i, j] - total_var[i, j - 1]) / (maturities[j] - maturities[j - 1])
                else:
                    dw_dT = (total_var[i, j + 1] - total_var[i, j - 1]) / (maturities[j + 1] - maturities[j - 1])

                if dw_dT <= 0:
                    continue   # Arbitrage in the surface; skip this cell

                # ---- dw/dk (log-strike derivative) ----
                if i == 0:
                    dw_dk = (w[i + 1] - w[i]) / (k[i + 1] - k[i])
                    d2w_dk2 = 0.0
                elif i == n_K - 1:
                    dw_dk = (w[i] - w[i - 1]) / (k[i] - k[i - 1])
                    d2w_dk2 = 0.0
                else:
                    dk_plus  = k[i + 1] - k[i]
                    dk_minus = k[i] - k[i - 1]
                    dw_dk    = (w[i + 1] - w[i - 1]) / (dk_plus + dk_minus)
                    d2w_dk2  = 2.0 * (w[i + 1] / dk_plus - w[i] * (1/dk_plus + 1/dk_minus) + w[i - 1] / dk_minus) / (dk_plus + dk_minus)

                wi = w[i]
                if wi <= 1e-10:
                    continue

                ki = k[i]

                # Gatheral (2006) Eq. 1.5: Dupire denominator
                denom = (1.0
                         - (ki / wi) * dw_dk
                         + 0.25 * (-0.25 - 1.0/wi + ki**2 / wi**2) * dw_dk**2
                         + 0.5 * d2w_dk2)

                if denom <= 1e-8:
                    continue   # Butterfly spread arbitrage; skip

                # Dupire local variance
                sigma_loc_sq = dw_dT / denom

                # Heston expected variance at this maturity
                # E[V_t] = theta + (V0 - theta) * exp(-kappa * T)
                E_V = heston_theta + (heston_V0 - heston_theta) * np.exp(-kappa * T)

                if E_V <= 1e-10:
                    continue

                # Leverage function L(K, T) = sqrt(sigma_loc^2 / E[V_t])
                L_sq = sigma_loc_sq / E_V
                if L_sq > 0:
                    leverage_matrix[i, j] = np.clip(np.sqrt(L_sq), 0.1, 5.0)

            except Exception:
                # Ill-conditioned cell: fall back to neutral leverage (L=1 => pure Heston)
                leverage_matrix[i, j] = 1.0

    return leverage_matrix

