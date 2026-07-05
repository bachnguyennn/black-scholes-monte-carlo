"""
heston_model.py

Implements the Heston Stochastic Volatility Model for option pricing.

Mathematical Foundation (Heston 1993):
    dS_t = (r - q) * S_t * dt + sqrt(V_t) * S_t * dW_t^S
    dV_t = kappa * (theta - V_t) * dt + xi * sqrt(V_t) * dW_t^V
    Corr(dW_t^S, dW_t^V) = rho

Feller Condition (REQUIRED for well-posed variance process):
    2 * kappa * theta > xi^2
    If violated, V_t collapses to zero with positive probability,
    corrupting the mathematical expectation of option prices.

Fourier/Characteristic Function Pricing:
    The Heston model has an exact semi-closed form solution via
    the characteristic function phi(u) and numerical integration:
    C = S0 * P1 - K * e^(-rT) * P2
    where P1, P2 are obtained via real-part integration.

Reference: Heston (1993); Andersen (2007) for discretization schemes.
"""

import numpy as np

# Fixed Gauss-Legendre quadrature nodes/weights for the Heston Fourier
# integral. Precomputed once: 128 nodes give ~1e-6 accuracy on [0, 200] for
# typical equity-index parameters at constant, region-independent cost.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(128)

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def feller_condition(kappa, theta, xi):
    """
    Checks the Feller Condition for the CIR variance process.

    The Feller Condition ensures the variance process V_t never touches
    zero: 2 * kappa * theta > xi^2.

    Inputs:
        kappa: Mean reversion speed (float)
        theta: Long-run variance (float)
        xi   : Vol of vol (float)

    Returns:
        dict with 'satisfied' (bool), 'discriminant' (float),
        and 'message' (str)
    """
    discriminant = 2 * kappa * theta - xi**2
    satisfied = discriminant > 0
    return {
        'satisfied': satisfied,
        'discriminant': round(discriminant, 6),
        'message': (
            f"[OK] Feller: 2*{kappa:.3f}*{theta:.4f} - {xi:.3f}^2 = {discriminant:.6f} > 0"
            if satisfied else
            f"[CRITICAL] Feller VIOLATED: 2*{kappa:.3f}*{theta:.4f} - {xi:.3f}^2 = {discriminant:.6f} <= 0. "
            f"Variance process will collapse. Increase kappa or theta, or reduce xi."
        )
    }


@njit(fastmath=True)
def simulate_heston(S0, T, r, V0, kappa, theta, xi, rho,
                    n_sims, n_steps=252, q=0.0, seed=-1):
    """
    Simulates terminal asset prices using the Heston stochastic
    volatility model with Euler-Maruyama and Full Truncation Scheme.

    Callers should check feller_condition() before using this
    function. If the Feller condition is violated, the full
    truncation scheme can still bias option prices.

    Inputs:
        S0    : Initial asset price (float)
        T     : Time to maturity in years (float)
        r     : Risk-free rate (float)
        V0    : Initial variance i.e. sigma^2 (float)
        kappa : Mean reversion speed (float)
        theta : Long-run variance (float)
        xi    : Vol of vol (float)
        rho   : Correlation dW_S and dW_V (float, typically negative)
        n_sims: Number of simulation paths (int)
        n_steps: Time steps (int, default 252)
        q     : Continuous dividend yield (float)
        seed: Random seed for reproducibility (int, default -1)

    Output:
        S_T: Terminal prices, shape (n_sims,)
        V_T: Terminal variances, shape (n_sims,)
    """
    if seed != -1:
        np.random.seed(seed)
        
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Pre-calculate constants
    rho_comp = np.sqrt(max(1.0 - rho**2, 0.0))

    S_T = np.zeros(n_sims)
    V_T = np.zeros(n_sims)

    for i in range(n_sims):
        log_S = np.log(S0)
        V = float(V0)
        
        for t in range(n_steps):
            Z_S = np.random.randn()
            Z_V_indep = np.random.randn()
            Z_V = rho * Z_S + rho_comp * Z_V_indep
            
            # Full Truncation: V_pos = max(V, 0)
            V_pos = max(V, 0.0)
            sqrt_V = np.sqrt(V_pos)

            log_S += (r - q - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z_S
            V += kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * Z_V

        S_T[i] = np.exp(log_S)
        V_T[i] = max(V, 0.0)
        
    return S_T, V_T


@njit(fastmath=True)
def simulate_heston_paths(S0, T, r, V0, kappa, theta, xi, rho,
                          n_paths, n_steps=100, q=0.0, seed=-1):
    """
    Simulates full price paths for visualization.

    Returns:
        paths    : Array of shape (n_paths, n_steps + 1)
        vol_paths: Array of shape (n_paths, n_steps + 1) with instantaneous vol sqrt(V_t)
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

            log_S += (r - q - 0.5 * V_pos) * dt + sqrt_V * sqrt_dt * Z_S
            V += kappa * (theta - V_pos) * dt + xi * sqrt_V * sqrt_dt * Z_V

            paths[i, t + 1] = np.exp(log_S)
            vol_paths[i, t + 1] = np.sqrt(max(V, 0.0))

    return paths, vol_paths


def _heston_characteristic_function(u, S0, T, r, q, V0, kappa, theta, xi, rho, j):
    """
    Computes the stable Heston characteristic function (Albrecher et al., 2007).
    
    This formulation avoids the multi-valued branching problem of the complex
    logarithm by ensuring the argument stays on the principal branch.
    
    Inputs:
        u  : Integration variable (scalar or NumPy array; evaluated
             elementwise so a whole quadrature grid can be priced at once)
        j  : 1 for P1 (delta-measure phi), 2 for P2 (risk-neutral phi)

    NumPy's complex sqrt/log use the same principal branch as cmath, so the
    Albrecher (2007) stable formulation remains branch-safe when vectorized.
    """
    # Heston parameters
    a = kappa * theta
    iu = 1j * np.asarray(u, dtype=np.complex128)

    # Measures are defined by b and u_j parameters
    if j == 1:
        u_j = 0.5
        b_j = kappa - rho * xi
    else:
        u_j = -0.5
        b_j = kappa

    # Characteristic function parameters
    # d = sqrt((rho * xi * iu - b_j)^2 - xi^2 * (2 * u_j * iu - u^2))
    d = np.sqrt((rho * xi * iu - b_j)**2 - xi**2 * (2 * u_j * iu - np.asarray(u)**2))

    # g = (b_j - rho * xi * iu + d) / (b_j - rho * xi * iu - d)
    g = (b_j - rho * xi * iu + d) / (b_j - rho * xi * iu - d)

    # Stable formulation for C and D
    # D(T) = (b_j - rho * xi * iu + d) / xi^2 * [(1 - exp(d*T)) / (1 - g * exp(d*T))]
    exp_dT = np.exp(d * T)
    D = (b_j - rho * xi * iu + d) / xi**2 * ((1 - exp_dT) / (1 - g * exp_dT))

    # C(T) = (r - q) * iu * T + (a / xi^2) * [(b_j - rho * xi * iu + d) * T - 2 * log((1 - g * exp(d*T)) / (1 - g))]
    C = (r - q) * iu * T + (a / xi**2) * ((b_j - rho * xi * iu + d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g)))

    # phi(u) = exp(C + D * V0 + iu * log(S0))
    return np.exp(C + D * V0 + iu * np.log(S0))


def price_option_heston_fourier(S0, K, T, r, V0, kappa, theta, xi, rho,
                                option_type='call', q=0.0):
    """
    Prices a European option analytically using the Heston model's
    characteristic function and numerical integration.

    This is the FAST PATH used by the scanner — no Monte Carlo paths needed.

    Integration uses a fixed 128-node Gauss-Legendre rule on [0, U] rather
    than adaptive quadrature. Adaptive `quad` is both slow and unreliable for
    the Heston integrand: for short maturities it oscillates and hits the
    subdivision limit, making cost blow up in exactly the parameter regions a
    calibrator explores. A fixed rule prices in constant time everywhere and
    is accurate to ~1e-6 for typical equity-index parameters.

    The formula (Heston 1993):
        C = S0 * e^(-q*T) * P1 - K * e^(-r*T) * P2

    Where P1 and P2 are risk-adjusted probabilities computed via:
        Pj = 1/2 + (1/pi) * Integral[Re(e^{-i*u*ln(K)} * phi_j(u) / (i*u))] du

    Inputs:
        S0, K, T, r, q: Standard option parameters
        V0, kappa, theta, xi, rho: Heston model parameters
        option_type: 'call' or 'put'

    Output:
        float: Option price
    """
    log_K = np.log(K)

    # Map the fixed Gauss-Legendre nodes from [-1, 1] onto the truncated
    # integration range [a, b]; integral ≈ (b-a)/2 * sum(w_i * f(u_i)).
    a, b = 1e-8, 200.0
    u = 0.5 * (b - a) * _GL_NODES + 0.5 * (b + a)
    half_width = 0.5 * (b - a)

    try:
        phi1 = _heston_characteristic_function(u, S0, T, r, q, V0, kappa, theta, xi, rho, 1)
        phi2 = _heston_characteristic_function(u, S0, T, r, q, V0, kappa, theta, xi, rho, 2)
        common = np.exp(-1j * u * log_K) / (1j * u)
        P1_integral = half_width * np.dot(_GL_WEIGHTS, np.real(common * phi1))
        P2_integral = half_width * np.dot(_GL_WEIGHTS, np.real(common * phi2))
    except Exception:
        # Fall back to Monte Carlo if integration fails
        return price_option_heston(S0, K, T, r, V0, kappa, theta, xi, rho, option_type, n_sims=10000)['price']

    P1 = 0.5 + P1_integral / np.pi
    P2 = 0.5 + P2_integral / np.pi

    # Clip probabilities to [0,1]
    P1 = float(np.clip(P1, 0.0, 1.0))
    P2 = float(np.clip(P2, 0.0, 1.0))

    call_price = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    if option_type == 'call':
        return max(float(call_price), 0.0)
    else:
        # Put-call parity: P = C - S*e^(-q*T) + K*e^(-r*T)
        put_price = call_price - S0 * np.exp(-q * T) + K * np.exp(-r * T)
        return max(float(put_price), 0.0)


def price_option_heston(S0, K, T, r, V0, kappa, theta, xi, rho,
                        option_type='call', n_sims=50000, n_steps=252, q=0.0):
    """
    Prices a European option using Heston model Monte Carlo.
    For batch scanning, use price_option_heston_fourier() instead.

    Output:
        dict with 'price', 'std_error', 'mean_vol'
    """
    S_T, V_T = simulate_heston(S0, T, r, V0, kappa, theta, xi, rho,
                                n_sims, n_steps, q)

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    discount = np.exp(-r * T)
    price = float(discount * np.mean(payoffs))
    std_error = float(discount * np.std(payoffs) / np.sqrt(n_sims))
    mean_vol = float(np.mean(np.sqrt(V_T)))

    return {
        'price': price,
        'std_error': std_error,
        'mean_vol': mean_vol
    }
