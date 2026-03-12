"""
scanner_engine.py

Batch pricing engine for scanning an entire options chain.
Calculates theoretical fair values, applies quote-quality filters,
and can optionally return structured diagnostics.
"""

import numpy as np
import pandas as pd
from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.black_scholes import black_scholes_price
from src.core.heston_model import price_option_heston_fourier
from src.core.lsv_model import simulate_lsv_paths
from src.core.config import (
    SCANNER_MONEYNESS_LOWER,
    SCANNER_MONEYNESS_UPPER,
    SCANNER_MIN_T_DAYS,
    SCANNER_MIN_MARKET_IV,
    SLIPPAGE_TIER_1_THRESHOLD,
    SLIPPAGE_TIER_2_THRESHOLD,
    SLIPPAGE_TIER_3_THRESHOLD,
    SLIPPAGE_TIER_1_PENALTY,
    SLIPPAGE_TIER_2_PENALTY,
    SLIPPAGE_TIER_3_PENALTY,
    SLIPPAGE_TIER_4_PENALTY,
)


def _initialize_scan_diagnostics(total_contracts, model, max_spread_pct, sigma_fallback):
    return {
        'total_contracts': int(total_contracts),
        'contracts_priced': 0,
        'contracts_filtered': 0,
        'reason_counts': {
            'ghost_contract': 0,
            'invalid_quote': 0,
            'moneyness': 0,
            'short_dte': 0,
            'wide_spread': 0,
            'pricing_error': 0,
        },
        'signal_counts': {
            'BUY': 0,
            'SELL': 0,
            'HOLD': 0,
        },
        'sigma_source_counts': {},
        'model_used': model,
        'max_spread_pct': float(max_spread_pct),
        'sigma_fallback': float(sigma_fallback),
    }


def price_single_option_mc(S0, K, T, r, sigma, option_type, n_sims=10000,
                           jump_intensity=0.1, jump_mean=-0.05, jump_std=0.03):
    """
    Prices a single option using Jump Diffusion Monte Carlo.

    Inputs:
        S0: Spot price (float)
        K: Strike price (float)
        T: Time to maturity in years (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        option_type: 'call' or 'put' (str)
        n_sims: Number of simulations (int)
        jump_intensity: Crash intensity lambda (float)
        jump_mean: Average crash size (float, negative)
        jump_std: Crash volatility (float)

    Output:
        dict with 'mc_price' and 'bs_price'
    """
    # Monte Carlo (Jump Diffusion)
    S_T, _ = simulate_jump_diffusion(
        S0, T, r, sigma, n_sims,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std
    )

    if option_type == 'call':
        payoffs = np.maximum(S_T - K, 0)
    else:
        payoffs = np.maximum(K - S_T, 0)

    mc_price = float(np.exp(-r * T) * np.mean(payoffs))

    # Black-Scholes (for comparison)
    bs_price_val = black_scholes_price(S0, K, T, r, sigma, option_type=option_type)

    return {
        'mc_price': mc_price,
        'bs_price': bs_price_val
    }


def scan_for_arbitrage(options_df, S0, r, q=0.0,
                       model='heston',
                       n_sims=10000,
                       leverage_matrix=None,
                       leverage_strikes=None,
                       leverage_maturities=None,
                       jump_intensity=0.1, jump_mean=-0.05, jump_std=0.03,
                       heston_V0=0.04, heston_kappa=2.0, heston_theta=0.04,
                       heston_xi=0.3, heston_rho=-0.7,
                       sigma_fallback=0.2,
                       max_spread_pct=0.25,
                       return_diagnostics=False):
    """
    Scans an entire options chain DataFrame for arbitrage opportunities.

    Uses realistic Bid-Ask spread logic instead of midpoint comparison:
        - BUY signal: mc_price > ask (you must pay the ask to buy)
        - SELL signal: mc_price < bid (you receive the bid when selling)

    Applies quote-quality and consistency filters:
        - Drops contracts with 0 volume AND 0 open interest.
        - Drops contracts where Bid <= 0 or Ask <= Bid.
        - Drops deep OTM/ITM contracts (>15% from spot).
        - Switches to Analytical Fourier Pricing for Heston to bypass
          the GIL and execute instantly.

    Inputs:
        options_df: DataFrame from data_fetcher.get_options_chain()
        S0: Current spot price (float)
        r: Risk-free rate (float)
        q: Dividend yield (float)
        model: 'gbm', 'jump_diffusion', 'heston', or 'lsv'
        n_sims: Number of Monte Carlo simulations
        leverage_matrix: For LSV -- 2D array of leverage factors (optional, fallback to L=1)
        leverage_strikes: For LSV -- 1D array of strike grid points
        leverage_maturities: For LSV -- 1D array of maturity grid points
        ... model params ...
        sigma_fallback: Volatility to use if market_iv is missing/invalid for JD/GBM
        max_spread_pct: Maximum allowable spread width
    """
    diagnostics = _initialize_scan_diagnostics(len(options_df), model, max_spread_pct, sigma_fallback)

    if options_df.empty:
        empty_df = pd.DataFrame()
        if return_diagnostics:
            return empty_df, diagnostics
        return empty_df

    results = []

    for idx, row in options_df.iterrows():
        K = row['strike']
        T = row['T']
        opt_type = row['type']
        bid = row['bid']
        ask = row['ask']
        mid = row['mid']
        vol = row.get('volume', 0)
        oi = row.get('openInterest', 0)

        # Quote-quality filtering
        
        # 1. Volume/OI Culling: Do not price ghost contracts
        if vol == 0 and oi == 0:
            diagnostics['reason_counts']['ghost_contract'] += 1
            continue
            
        # 2. Invalid Quote Check
        if bid <= 0 or ask <= bid:
            diagnostics['reason_counts']['invalid_quote'] += 1
            continue
            
        # 3. Moneyness Bounds (config: SCANNER_MONEYNESS_LOWER / UPPER)
        if K < S0 * SCANNER_MONEYNESS_LOWER or K > S0 * SCANNER_MONEYNESS_UPPER:
            diagnostics['reason_counts']['moneyness'] += 1
            continue

        # Skip very short-dated options (config: SCANNER_MIN_T_DAYS)
        if T < SCANNER_MIN_T_DAYS / 365:
            diagnostics['reason_counts']['short_dte'] += 1
            continue

        # Liquidity filter: skip contracts with excessively wide spreads
        spread = ask - bid
        spread_pct = spread / mid if mid > 0 else 999
        if spread_pct > max_spread_pct:
            diagnostics['reason_counts']['wide_spread'] += 1
            continue

        try:
            # Sigma Hierarchy: use market IV if above minimum validity threshold
            # (config: SCANNER_MIN_MARKET_IV)
            mkt_iv = float(row.get('market_iv', 0))
            if mkt_iv > SCANNER_MIN_MARKET_IV:
                sigma_used = mkt_iv
                sigma_source = "Market IV"
            else:
                sigma_used = sigma_fallback
                sigma_source = "Fallback"

            # Route to the selected pricing model
            if model == 'heston':
                 # Analytical fast-path
                 mc_price = price_option_heston_fourier(
                     S0, K, T, r, heston_V0, heston_kappa, heston_theta,
                     heston_xi, heston_rho, option_type=opt_type, q=q
                 )
                 bs_price = black_scholes_price(S0, K, T, r, sigma_used, opt_type, q=q)
                 sigma_source = "Heston Surface (V0)" if sigma_source == "Fallback" else sigma_source
            elif model == 'lsv':
                 # LSV Simulation Path with calibrated leverage (or fallback to pure Heston)
                 if leverage_matrix is None:
                     # No calibrated leverage provided; fallback to pure Heston (L=1 everywhere)
                     lsv_leverage = np.ones((100, 100))
                     lsv_strikes = np.linspace(S0 * 0.5, S0 * 1.5, 100)
                     lsv_mats = np.linspace(0.01, 2.0, 100)
                     sigma_source = "LSV (No Calibration - Pure Heston)"
                 else:
                     # Use calibrated leverage
                     lsv_leverage = leverage_matrix
                     lsv_strikes = leverage_strikes
                     lsv_mats = leverage_maturities
                     sigma_source = "LSV (Calibrated)"

                 paths, _ = simulate_lsv_paths(
                     S0, T, r, heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
                     lsv_leverage, lsv_strikes, lsv_mats,
                     n_paths=n_sims, n_steps=50, q=q
                 )
                 S_T_lsv = paths[:, -1]
                 if opt_type == 'call':
                     payoffs = np.maximum(S_T_lsv - K, 0)
                 else:
                     payoffs = np.maximum(K - S_T_lsv, 0)

                 mc_price = float(np.exp(-r * T) * np.mean(payoffs))
                 bs_price = black_scholes_price(S0, K, T, r, sigma_used, opt_type, q=q)
                 sigma_source = "LSV Local Vol Surface" if sigma_source == "Fallback" else sigma_source
            elif model == 'gbm':
                 # Standard Black-Scholes (GBM)
                 mc_price = black_scholes_price(S0, K, T, r, sigma_used, opt_type, q=q)
                 bs_price = mc_price
            else:
                 # Standard JD Monte Carlo
                 prices = price_single_option_mc(
                     S0, K, T, r, sigma_used, opt_type, n_sims,
                     jump_intensity, jump_mean, jump_std
                 )
                 mc_price = prices['mc_price']
                 bs_price = prices['bs_price']

            # Order-book depth slippage penalty (config: SLIPPAGE_TIER_* constants)
            # Liquidity score = max(daily volume, open interest)
            liquidity_score = max(vol, oi)
            if liquidity_score < SLIPPAGE_TIER_1_THRESHOLD:
                penalty_pct = SLIPPAGE_TIER_1_PENALTY
            elif liquidity_score < SLIPPAGE_TIER_2_THRESHOLD:
                penalty_pct = SLIPPAGE_TIER_2_PENALTY
            elif liquidity_score < SLIPPAGE_TIER_3_THRESHOLD:
                penalty_pct = SLIPPAGE_TIER_3_PENALTY
            else:
                penalty_pct = SLIPPAGE_TIER_4_PENALTY

            effective_ask = ask * (1.0 + penalty_pct)
            effective_bid = bid * (1.0 - penalty_pct)

            # Realistic edge calculation using bid-ask crossing and slippage penalty
            # To BUY: you must cross the effective ask -> edge = mc_price - effective_ask
            buy_edge = mc_price - effective_ask
            # To SELL: you receive the effective bid -> edge = effective_bid - mc_price
            sell_edge = effective_bid - mc_price

            # Signal logic based on crossing the spread
            if buy_edge > 0:
                edge = buy_edge
                edge_pct = (edge / ask) * 100
                signal = "[ BUY ]"
            elif sell_edge > 0:
                edge = sell_edge
                edge_pct = (edge / bid) * 100
                signal = "[ SELL ]"
            else:
                # No actionable edge after crossing the spread
                edge = mc_price - mid
                edge_pct = (edge / mid) * 100 if mid > 0 else 0.0
                signal = "[ HOLD ]"

            signal_key = signal.replace('[', '').replace(']', '').strip()
            diagnostics['signal_counts'][signal_key] += 1
            diagnostics['sigma_source_counts'][sigma_source] = diagnostics['sigma_source_counts'].get(sigma_source, 0) + 1

            results.append({
                'type': opt_type.upper(),
                'strike': K,
                'expiration': row['expiration'],
                'T_days': int(round(T * 365)),
                'bid': bid,
                'ask': ask,
                'mid': mid,
                'spread': round(spread, 4),
                'spread_pct': round(spread_pct * 100, 1),
                'mc_price': round(mc_price, 4),
                'bs_price': round(bs_price, 4),
                'edge': round(edge, 4),
                'edge_pct': round(edge_pct, 1),
                'signal': signal,
                'volume': row['volume'],
                'openInterest': oi,
                'market_iv': round(row['market_iv'] * 100, 1),
                'sigma_used': round(sigma_used * 100, 1),
                'sigma_source': sigma_source
            })

        except Exception:
            diagnostics['reason_counts']['pricing_error'] += 1
            continue

    diagnostics['contracts_priced'] = len(results)
    diagnostics['contracts_filtered'] = diagnostics['total_contracts'] - diagnostics['contracts_priced']

    if not results:
        empty_df = pd.DataFrame()
        if return_diagnostics:
            return empty_df, diagnostics
        return empty_df

    result_df = pd.DataFrame(results)

    # Sort by absolute edge percentage (biggest opportunities first)
    result_df['abs_edge_pct'] = result_df['edge_pct'].abs()
    result_df = result_df.sort_values('abs_edge_pct', ascending=False).drop(columns='abs_edge_pct')

    result_df = result_df.reset_index(drop=True)
    if return_diagnostics:
        return result_df, diagnostics
    return result_df
