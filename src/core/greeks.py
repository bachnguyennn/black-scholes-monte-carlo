"""
greeks.py

Calculates option Greeks using finite difference methods.
Greeks measure the sensitivity of option prices to various parameters.

For Jump Diffusion models, analytical Greeks are not available,
so we use numerical approximation via finite differences.
"""

import numpy as np
from scipy.stats import norm

try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.stats import norm as jax_norm
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    jax_norm = None
    JAX_AVAILABLE = False

from .jump_diffusion import simulate_jump_diffusion, merton_jump_price
from .gbm_engine import simulate_gbm


def _jump_kwargs(kwargs):
    """Pull the Merton jump parameters (and dividend yield) out of **kwargs."""
    return dict(
        jump_intensity=kwargs.get('jump_intensity', 0.1),
        jump_mean=kwargs.get('jump_mean', -0.05),
        jump_std=kwargs.get('jump_std', 0.03),
        q=kwargs.get('q', 0.0),
    )


def _merton(S0, K, T, r, sigma, option_type, kwargs):
    return merton_jump_price(S0, K, T, r, sigma, option_type=option_type,
                             **_jump_kwargs(kwargs))

if JAX_AVAILABLE:
    @jax.jit
    def bs_price_jax(S, K, T, r, sigma, is_call=True):
        """
        JAX-compatible Black-Scholes pricing formula.
        Used exclusively for Automatic Differentiation to get exact, noise-free Greeks.
        """
        d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)

        call_price = S * jax_norm.cdf(d1) - K * jnp.exp(-r * T) * jax_norm.cdf(d2)
        put_price = K * jnp.exp(-r * T) * jax_norm.cdf(-d2) - S * jax_norm.cdf(-d1)

        return jnp.where(is_call, call_price, put_price)

    _jax_delta = jax.jit(jax.grad(bs_price_jax, argnums=0))
    _jax_vega = jax.jit(jax.grad(bs_price_jax, argnums=4))
    _jax_gamma = jax.jit(jax.grad(jax.grad(bs_price_jax, argnums=0), argnums=0))
else:
    bs_price_jax = None
    _jax_delta = None
    _jax_vega = None
    _jax_gamma = None


def _bs_d1_d2(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2, sqrt_T



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
    # Jump diffusion has an exact analytic price (Merton series) -- use it
    # rather than a noisy Monte Carlo estimate.
    if model == 'jump_diffusion':
        return _merton(S0, K, T, r, sigma, option_type.lower(), kwargs)

    if model == 'gbm':
        S_T = simulate_gbm(S0, T, r, sigma, n_sims)
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
    Calculate Delta.

    For 'jump_diffusion', Delta is a central finite difference on the exact
    Merton analytic price, so it reflects jump risk. For 'gbm'/Black-Scholes
    it uses JAX automatic differentiation (exact, noise-free).
    """
    opt = option_type.lower()
    if model == 'jump_diffusion':
        hS = max(S0 * 1e-3, 1e-4)
        up = _merton(S0 + hS, K, T, r, sigma, opt, kwargs)
        dn = _merton(S0 - hS, K, T, r, sigma, opt, kwargs)
        return (up - dn) / (2.0 * hS)

    is_call = opt == 'call'
    if JAX_AVAILABLE:
        return float(_jax_delta(S0, K, T, r, sigma, is_call))

    d1, _, _ = _bs_d1_d2(S0, K, T, r, sigma)
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


def calculate_vega(S0, K, T, r, sigma, option_type='call',
                   model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate Vega (sensitivity to the diffusion volatility), reported per 1%
    vol change.

    For 'jump_diffusion' this is a central finite difference on the Merton
    price w.r.t. the diffusion sigma; for 'gbm'/Black-Scholes it uses JAX
    automatic differentiation.
    """
    opt = option_type.lower()
    if model == 'jump_diffusion':
        hv = 0.01
        up = _merton(S0, K, T, r, sigma + hv, opt, kwargs)
        dn = _merton(S0, K, T, r, max(sigma - hv, 1e-6), opt, kwargs)
        raw_vega = (up - dn) / (2.0 * hv)   # per unit vol
        return raw_vega / 100.0

    is_call = opt == 'call'
    if JAX_AVAILABLE:
        raw_vega = float(_jax_vega(S0, K, T, r, sigma, is_call))
    else:
        d1, _, sqrt_T = _bs_d1_d2(S0, K, T, r, sigma)
        raw_vega = float(S0 * norm.pdf(d1) * sqrt_T)
    return raw_vega / 100.0


def calculate_gamma(S0, K, T, r, sigma, option_type='call',
                    model='gbm', h=0.01, n_sims=50000, **kwargs):
    """
    Calculate Gamma (second derivative w.r.t. spot).

    For 'jump_diffusion' this is a second-order central finite difference on
    the Merton price; for 'gbm'/Black-Scholes it uses JAX automatic
    differentiation.
    """
    opt = option_type.lower()
    if model == 'jump_diffusion':
        hS = max(S0 * 1e-3, 1e-4)
        up = _merton(S0 + hS, K, T, r, sigma, opt, kwargs)
        mid = _merton(S0, K, T, r, sigma, opt, kwargs)
        dn = _merton(S0 - hS, K, T, r, sigma, opt, kwargs)
        return (up - 2.0 * mid + dn) / (hS * hS)

    is_call = opt == 'call'
    if JAX_AVAILABLE:
        return float(_jax_gamma(S0, K, T, r, sigma, is_call))

    d1, _, sqrt_T = _bs_d1_d2(S0, K, T, r, sigma)
    return float(norm.pdf(d1) / (S0 * sigma * sqrt_T))


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
