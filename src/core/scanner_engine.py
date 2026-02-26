"""
scanner_engine.py

Batch Monte Carlo pricing engine for scanning an entire options chain.
Takes a DataFrame of option contracts and calculates the theoretical
Fair Value for each using the Jump Diffusion model, then identifies
arbitrage opportunities by comparing against market prices.
"""

import numpy as np
import pandas as pd
from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.black_scholes import black_scholes_price


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


def scan_for_arbitrage(options_df, S0, r, sigma,
                       n_sims=10000,
                       jump_intensity=0.1,
                       jump_mean=-0.05,
                       jump_std=0.03):
    """
    Scans an entire options chain DataFrame for arbitrage opportunities.

    For each option contract, it calculates the Monte Carlo Fair Value
    using Jump Diffusion and compares it to the market mid price.

    Inputs:
        options_df: DataFrame from data_fetcher.get_options_chain()
        S0: Current spot price (float)
        r: Risk-free rate (float)
        sigma: Historical volatility (float)
        n_sims: Number of MC simulations per option (int)
        jump_intensity: Crash intensity (float)
        jump_mean: Average crash size (float)
        jump_std: Crash volatility (float)

    Output:
        pd.DataFrame with additional columns:
            'mc_price'     : Monte Carlo fair value
            'bs_price'     : Black-Scholes price
            'edge'         : mc_price - market mid price (positive = undervalued)
            'edge_pct'     : edge as percentage of mid price
            'signal'       : 'BUY', 'SELL', or 'HOLD'
    """
    if options_df.empty:
        return pd.DataFrame()

    results = []

    for idx, row in options_df.iterrows():
        K = row['strike']
        T = row['T']
        opt_type = row['type']
        mid = row['mid']

        # Skip deep OTM options with near-zero mid prices
        if mid < 0.05:
            continue

        # Skip options expiring in less than ~15 mins (T < 0.00003 years)
        if T < 0.0001:
            continue

        try:
            prices = price_single_option_mc(
                S0, K, T, r, sigma, opt_type, n_sims,
                jump_intensity, jump_mean, jump_std
            )

            mc_price = prices['mc_price']
            bs_price = prices['bs_price']

            # Edge = how much the model thinks the option is worth vs market
            edge = mc_price - mid
            edge_pct = (edge / mid) * 100 if mid > 0 else 0.0

            # Signal logic
            if edge_pct > 10:
                signal = "🟢 BUY"
            elif edge_pct < -10:
                signal = "🔴 SELL"
            else:
                signal = "⚪ HOLD"

            results.append({
                'type': opt_type.upper(),
                'strike': K,
                'expiration': row['expiration'],
                'T_days': int(round(T * 365)),
                'bid': row['bid'],
                'ask': row['ask'],
                'mid': mid,
                'mc_price': round(mc_price, 4),
                'bs_price': round(bs_price, 4),
                'edge': round(edge, 4),
                'edge_pct': round(edge_pct, 1),
                'signal': signal,
                'volume': row['volume'],
                'market_iv': round(row['market_iv'] * 100, 1),
            })

        except Exception:
            continue  # Skip options that cause numerical issues

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Sort by absolute edge percentage (biggest opportunities first)
    result_df['abs_edge_pct'] = result_df['edge_pct'].abs()
    result_df = result_df.sort_values('abs_edge_pct', ascending=False).drop(columns='abs_edge_pct')

    return result_df.reset_index(drop=True)
