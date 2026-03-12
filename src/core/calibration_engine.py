"""
calibration_engine.py

Calibrates Heston model parameters to fit the live implied volatility surface.

Objective:
    Find (kappa, theta, xi, rho, V0) that minimize the weighted Sum of
    Squared Errors (SSE) between market-observed IVs and Heston model IVs:

    SSE = sum_i w_i * (IV_market(K_i, T_i) - IV_heston(K_i, T_i | params))^2

Optimizer:
    scipy.optimize.minimize with SLSQP method to respect parameter bounds.
    Bounds: kappa > 0, theta > 0, xi > 0, -1 < rho < 0, V0 > 0

Reference:
    Heston (1993); Gatheral (2006) "The Volatility Surface"
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from src.core.heston_model import price_option_heston_fourier, feller_condition
from src.core.black_scholes import black_scholes_price


def _implied_vol_from_price(price, S0, K, T, r, option_type='call', q=0.0):
    """
    Back-solves implied volatility from an option price using Brent's method.

    Uses binary search on the Black-Scholes formula to find sigma such that
    BS(sigma) == price.

    Inputs:
        price: Option price (float)
        S0, K, T, r: Standard option parameters
        option_type: 'call' or 'put'
        q: Dividend yield

    Output:
        float: Implied volatility, or NaN if no solution found
    """
    intrinsic = max(S0 - K, 0) if option_type == 'call' else max(K - S0, 0)
    if price <= intrinsic + 1e-6:
        return float('nan')

    lo, hi = 1e-4, 5.0

    for _ in range(100):
        mid = (lo + hi) / 2.0
        bs = black_scholes_price(S0, K, T, r, mid, option_type, q=q)
        if bs < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break

    iv = (lo + hi) / 2.0
    return iv if 0.001 < iv < 4.9 else float('nan')


def calibrate_heston(options_df, S0, r, q=0.0,
                     initial_params=None,
                     atm_moneyness_bounds=(0.80, 1.20)):
    """
    Calibrates Heston parameters to a live options chain DataFrame.

    For each option in options_df, the market IV is compared against the
    Heston model IV (computed via Fourier inversion). The optimizer
    minimizes the weighted SSE using SLSQP.

    Inputs:
        options_df: DataFrame from data_fetcher.get_options_chain()
        S0        : Current spot price (float)
        r         : Risk-free rate (float)
        q         : Continuous dividend yield (float)
        initial_params: dict with 'kappa', 'theta', 'xi', 'rho', 'V0'
                        If None, sensible defaults are used.
        atm_moneyness_bounds: tuple (lo, hi) -- only calibrate to options
                        within this moneyness range (default 80%-120%)

    Output:
        dict with:
            'kappa', 'theta', 'xi', 'rho', 'V0': Calibrated parameters
            'sse': Final sum of squared errors
            'n_contracts': Number of contracts used
            'feller': Feller condition check result
            'success': bool
            'message': str
    """
    # Filter to calibration universe
    calib_df = options_df.copy()
    calib_df = calib_df[calib_df['T'] > 3 / 365]  # Skip very short-dated
    calib_df = calib_df[calib_df['mid'] > 0.05]     # Skip illiquid
    # Moneyness filter
    lo_strike = S0 * atm_moneyness_bounds[0]
    hi_strike = S0 * atm_moneyness_bounds[1]
    calib_df = calib_df[(calib_df['strike'] >= lo_strike) & (calib_df['strike'] <= hi_strike)]

    if len(calib_df) < 5:
        return {
            'success': False,
            'message': f"Insufficient calibration data: only {len(calib_df)} contracts after filtering.",
            'n_contracts': len(calib_df)
        }

    # Build calibration targets (market IVs)
    strikes = calib_df['strike'].values
    maturities = calib_df['T'].values
    opt_types = calib_df['type'].str.lower().values
    market_ivs = calib_df['market_iv'].values

    # Only use contracts where market IV is meaningful
    valid = (market_ivs > 0.01) & (market_ivs < 3.0)
    strikes = strikes[valid]
    maturities = maturities[valid]
    opt_types = opt_types[valid]
    market_ivs = market_ivs[valid]

    if len(strikes) < 5:
        return {
            'success': False,
            'message': f"Insufficient valid market IVs after cleaning: {len(strikes)} contracts.",
            'n_contracts': len(strikes)
        }

    # Weight: higher volume contracts matter more (use moneyness-based weights)
    moneyness = strikes / S0
    weights = np.exp(-2 * (np.log(moneyness))**2)  # Gaussian weight centered at ATM
    weights = weights / weights.sum()

    # Initial parameters
    if initial_params is None:
        hist_var = (market_ivs.mean())**2
        initial_params = {
            'kappa': 2.0,
            'theta': hist_var,
            'xi': 0.3,
            'rho': -0.7,
            'V0': hist_var
        }

    x0 = [initial_params['kappa'], initial_params['theta'],
           initial_params['xi'], initial_params['rho'], initial_params['V0']]

    # Parameter bounds: kappa, theta, xi > 0; -1 < rho < 0; V0 > 0
    bounds = [(0.1, 20.0), (0.001, 2.0), (0.05, 3.0), (-0.999, -0.001), (0.0001, 2.0)]

    def objective(params):
        kappa, theta, xi, rho, V0 = params
        sse = 0.0
        for i in range(len(strikes)):
            try:
                heston_price = price_option_heston_fourier(
                    S0, strikes[i], maturities[i], r,
                    V0, kappa, theta, xi, rho,
                    option_type=opt_types[i], q=q
                )
                heston_iv = _implied_vol_from_price(
                    heston_price, S0, strikes[i], maturities[i], r, opt_types[i], q=q
                )
                if np.isnan(heston_iv):
                    sse += weights[i] * 1.0   # Heavy penalty for failed pricing
                    continue
                sse += weights[i] * (market_ivs[i] - heston_iv)**2
            except Exception:
                sse += weights[i] * 1.0
        return sse

    result = minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    if result.success or result.fun < 0.01:
        kappa, theta, xi, rho, V0 = result.x
        fc = feller_condition(kappa, theta, xi)
        return {
            'kappa': round(float(kappa), 4),
            'theta': round(float(theta), 6),
            'xi': round(float(xi), 4),
            'rho': round(float(rho), 4),
            'V0': round(float(V0), 6),
            'sse': round(float(result.fun), 8),
            'n_contracts': len(strikes),
            'feller': fc,
            'success': True,
            'message': f"Calibrated to {len(strikes)} contracts. SSE={result.fun:.6f}. {fc['message']}"
        }
    else:
        return {
            'success': False,
            'message': f"Optimizer did not converge: {result.message}",
            'n_contracts': len(strikes),
            'sse': float(result.fun)
        }


def build_iv_surface(options_df, S0, num_strikes=None, num_mats=None,
                     interpolation='linear', atm_moneyness_bounds=(0.80, 1.20)):
    """
    Builds a 2D regular IV surface from scattered options chain data.

    This function converts a flat DataFrame of options into a regular (strikes × maturities) grid
    using interpolation. Grid resolution is adaptive based on data density.

    Inputs:
        options_df: DataFrame with columns ['strike', 'T', 'market_iv', 'bid', 'ask', 'mid']
        S0: Spot price (float) - used for moneyness-based gridding
        num_strikes: Number of strike grid points (int).
                     If None, auto-detect based on data density.
        num_mats: Number of maturity grid points (int).
                  If None, auto-detect based on data density.
        interpolation: 'linear' (default) or 'cubic' for sparse data
        atm_moneyness_bounds: Filter to (lower, upper) moneyness range

    Returns:
        dict with:
            'strikes_grid': 1D sorted array of strike prices
            'maturities_grid': 1D sorted array of times in years
            'iv_surface': 2D array of shape (len(strikes_grid), len(maturities_grid))
            'coverage': float - percentage of grid cells with valid data (0-100)
            'n_contracts_input': int - input contracts
            'n_contracts_used': int - after filtering
            'message': str - status message
    """
    from scipy.interpolate import griddata
    import scipy.ndimage as ndimage

    # Filter and clean data
    df_clean = options_df.copy()

    # Remove bad IV data
    df_clean = df_clean[(df_clean['market_iv'] > 0.01) & (df_clean['market_iv'] < 3.0)]

    # Remove very short-dated
    df_clean = df_clean[df_clean['T'] > 1/365]

    # Moneyness filter
    lo_strike = S0 * atm_moneyness_bounds[0]
    hi_strike = S0 * atm_moneyness_bounds[1]
    df_clean = df_clean[(df_clean['strike'] >= lo_strike) & (df_clean['strike'] <= hi_strike)]

    # Remove illiquid (bid/ask spread too wide)
    df_clean['spread'] = df_clean['ask'] - df_clean['bid']
    df_clean['spread_pct'] = df_clean['spread'] / df_clean['mid']
    df_clean = df_clean[df_clean['spread_pct'] < 0.50]  # Less than 50% spread

    n_input = len(options_df)
    n_used = len(df_clean)

    if n_used < 5:
        return {
            'success': False,
            'strikes_grid': None,
            'maturities_grid': None,
            'iv_surface': None,
            'coverage': 0.0,
            'n_contracts_input': n_input,
            'n_contracts_used': n_used,
            'message': f"Insufficient data for IV surface: only {n_used} contracts after filtering (need >= 5)"
        }

    # Auto-detect grid resolution if not provided
    if num_strikes is None or num_mats is None:
        if n_used < 20:
            num_strikes = 10 if num_strikes is None else num_strikes
            num_mats = 8 if num_mats is None else num_mats
        elif n_used < 50:
            num_strikes = 15 if num_strikes is None else num_strikes
            num_mats = 10 if num_mats is None else num_mats
        elif n_used < 100:
            num_strikes = 20 if num_strikes is None else num_strikes
            num_mats = 12 if num_mats is None else num_mats
        else:
            num_strikes = 25 if num_strikes is None else num_strikes
            num_mats = 15 if num_mats is None else num_mats

    # Create regular grid points
    strikes_grid = np.exp(np.linspace(
        np.log(S0 * atm_moneyness_bounds[0]),
        np.log(S0 * atm_moneyness_bounds[1]),
        num_strikes
    ))

    min_T = df_clean['T'].min()
    max_T = df_clean['T'].max()
    maturities_grid = np.linspace(min_T, max_T, num_mats)

    # Interpolate scattered IV data to a regular grid
    points = df_clean[['strike', 'T']].values
    values = df_clean['market_iv'].values

    # Create meshgrid for interpolation target
    strike_mesh, mat_mesh = np.meshgrid(strikes_grid, maturities_grid, indexing='ij')
    xy = np.column_stack([strike_mesh.ravel(), mat_mesh.ravel()])

    # Interpolate with fallback strategy
    iv_surface = None
    last_error = None

    # Check data dimensionality first
    strike_range = df_clean['strike'].max() - df_clean['strike'].min()
    mat_range = df_clean['T'].max() - df_clean['T'].min()

    # If data is essentially 1D (flat in one dimension), use nearest neighbor
    if strike_range < 1.0 or mat_range < 1e-6:
        interpolation = 'nearest'

    # Try interpolation with fallback chain
    for method in [interpolation, 'linear', 'nearest']:
        try:
            iv_surface = griddata(points, values, xy, method=method)
            iv_surface = iv_surface.reshape(strike_mesh.shape)
            break  # Success, exit loop
        except Exception as e:
            last_error = str(e)
            continue  # Try next method

    if iv_surface is None:
        return {
            'success': False,
            'strikes_grid': None,
            'maturities_grid': None,
            'iv_surface': None,
            'coverage': 0.0,
            'n_contracts_input': n_input,
            'n_contracts_used': n_used,
            'message': f"Interpolation failed after all attempts: {last_error}"
        }

    # Fill NaN cells with nearby valid values
    nan_mask = np.isnan(iv_surface)
    if np.any(nan_mask):
        # Use forward-fill along each axis
        for i in range(iv_surface.shape[0]):
            valid_idx = np.where(~np.isnan(iv_surface[i, :]))[0]
            if len(valid_idx) > 0:
                iv_surface[i, nan_mask[i, :]] = np.interp(
                    np.where(nan_mask[i, :])[0],
                    valid_idx,
                    iv_surface[i, valid_idx]
                )

        # Handle remaining NaNs with spatial smoothing
        if np.any(np.isnan(iv_surface)):
            nan_mask = np.isnan(iv_surface)
            iv_surface[nan_mask] = np.nanmean(iv_surface)

    # Light Gaussian smoothing for stability
    iv_surface = ndimage.gaussian_filter(iv_surface, sigma=0.5)

    # Clip to reasonable bounds
    iv_surface = np.clip(iv_surface, 0.01, 3.0)

    # Calculate coverage (fraction of grid cells that had original data nearby)
    coverage = 100.0 * (1.0 - np.mean(nan_mask) if np.any(nan_mask) else 100.0)

    return {
        'success': True,
        'strikes_grid': strikes_grid,
        'maturities_grid': maturities_grid,
        'iv_surface': iv_surface,
        'coverage': float(coverage),
        'n_contracts_input': n_input,
        'n_contracts_used': n_used,
        'message': f"IV surface: {num_strikes}×{num_mats} grid, {coverage:.0f}% coverage"
    }


def calibrate_lsv(options_df, S0, r, q=0.0, heston_params=None,
                  num_strikes=None, num_mats=None,
                  interpolation='linear',
                  atm_moneyness_bounds=(0.80, 1.20)):
    """
    Full LSV calibration pipeline: IV surface → leverage function → results.

    This function orchestrates the complete LSV calibration process:
    1. Build IV surface from scattered options data
    2. Optionally calibrate Heston if params not provided
    3. Calibrate leverage function from surface
    4. Return calibrated leverage matrix and grid

    Inputs:
        options_df: DataFrame from data_fetcher.get_options_chain()
        S0: Current spot price (float)
        r: Risk-free rate (float)
        q: Dividend yield (float)
        heston_params: dict with 'kappa', 'theta', 'xi', 'rho', 'V0'
                       If None, uses defaults or calibrates from data
        num_strikes: IV surface grid strikes (auto-detect if None)
        num_mats: IV surface grid maturities (auto-detect if None)
        interpolation: 'linear' or 'cubic'
        atm_moneyness_bounds: Filter range for calibration

    Returns:
        dict with:
            'success': bool
            'leverage_matrix': 2D array of shape (num_strikes, num_mats) or None if failed
            'strikes_grid': 1D array of strikes
            'maturities_grid': 1D array of maturities
            'heston_kappa': float (mean reversion speed used)
            'heston_theta': float (long-run mean used)
            'heston_V0': float (initial variance used)
            'surface_coverage': float (% of IV grid with data)
            'n_contracts': int (input contracts)
            'n_used': int (after filtering)
            'message': str
    """
    from src.core.lsv_model import calibrate_leverage_function

    # Build IV surface
    surf_res = build_iv_surface(
        options_df, S0,
        num_strikes=num_strikes, num_mats=num_mats,
        interpolation=interpolation,
        atm_moneyness_bounds=atm_moneyness_bounds
    )

    if not surf_res.get('success', False):
        return {
            'success': False,
            'leverage_matrix': None,
            'message': surf_res['message'],
            'n_contracts': surf_res.get('n_contracts_input', 0),
            'n_used': surf_res.get('n_contracts_used', 0)
        }

    iv_surface = surf_res['iv_surface']
    strikes_grid = surf_res['strikes_grid']
    maturities_grid = surf_res['maturities_grid']
    coverage = surf_res['coverage']

    # Get Heston parameters
    if heston_params is None:
        # Try to calibrate Heston
        heston_res = calibrate_heston(options_df, S0, r, q=q)
        if heston_res.get('success', False):
            heston_params = {
                'kappa': heston_res['kappa'],
                'theta': heston_res['theta'],
                'xi': heston_res['xi'],
                'rho': heston_res['rho'],
                'V0': heston_res['V0']
            }
        else:
            # Use defaults
            heston_params = {
                'kappa': 2.0,
                'theta': (options_df['market_iv'].mean() ** 2) if 'market_iv' in options_df.columns else 0.04,
                'xi': 0.3,
                'rho': -0.7,
                'V0': (options_df['market_iv'].mean() ** 2) if 'market_iv' in options_df.columns else 0.04
            }

    kappa = heston_params.get('kappa', 2.0)
    theta = heston_params.get('theta', 0.04)
    V0 = heston_params.get('V0', 0.04)

    # Check surface quality
    if coverage < 30:
        return {
            'success': False,
            'leverage_matrix': None,
            'message': f"IV surface coverage too low ({coverage:.0f}%, need >= 30%)",
            'surface_coverage': coverage,
            'n_contracts': surf_res['n_contracts_input'],
            'n_used': surf_res['n_contracts_used']
        }

    # Calibrate leverage function
    try:
        leverage_matrix = calibrate_leverage_function(
            iv_surface, strikes_grid, maturities_grid,
            r=r, q=q,
            kappa=kappa,
            heston_theta=theta,
            heston_V0=V0
        )

        if leverage_matrix is None:
            return {
                'success': False,
                'leverage_matrix': None,
                'message': "Leverage calibration returned None",
                'surface_coverage': coverage,
                'n_contracts': surf_res['n_contracts_input'],
                'n_used': surf_res['n_contracts_used']
            }

        return {
            'success': True,
            'leverage_matrix': leverage_matrix,
            'strikes_grid': strikes_grid,
            'maturities_grid': maturities_grid,
            'heston_kappa': float(kappa),
            'heston_theta': float(theta),
            'heston_V0': float(V0),
            'surface_coverage': float(coverage),
            'n_contracts': surf_res['n_contracts_input'],
            'n_used': surf_res['n_contracts_used'],
            'message': f"LSV calibrated: {len(strikes_grid)}×{len(maturities_grid)} leverage matrix, {coverage:.0f}% surface coverage"
        }

    except Exception as e:
        return {
            'success': False,
            'leverage_matrix': None,
            'message': f"Leverage calibration error: {str(e)}",
            'surface_coverage': coverage,
            'n_contracts': surf_res['n_contracts_input'],
            'n_used': surf_res['n_contracts_used']
        }
