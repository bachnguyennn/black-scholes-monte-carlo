"""
backtester.py

Research backtester for model-driven option trades.

The engine avoids look-ahead bias in its volatility estimation and runs a
daily delta-hedging routine, but it does not replay historical option
quotes. Entry prices are proxied with Black-Scholes using no-look-ahead
rolling volatility, so results should be presented as controlled research
evidence rather than executable historical fills.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from src.core.heston_model import price_option_heston_fourier
from src.core.jump_diffusion import simulate_jump_diffusion


def fetch_historical_prices(ticker='^SPX', period='2y'):
    try:
        tk = yf.Ticker(ticker)
        history = tk.history(period=period)
        if history.empty:
            return None
        if 'Adj Close' in history.columns:
            return history['Adj Close']
        return history['Close']
    except Exception as e:
        print(f"ERROR: fetch_historical_prices failed for {ticker}: {str(e)}")
        return None


def _calculate_rolling_vol_no_lookahead(prices, current_date, window=30):
    """
    Calculates historical volatility strictly using data BEFORE current_date.
    No leakage allowed.
    """
    past_prices = prices.loc[:current_date - pd.Timedelta(days=1)]
    if len(past_prices) < window:
        return np.nan
    
    # Assert Look-Ahead bias is contained
    assert max(past_prices.index) < current_date, "CRITICAL: Look-ahead bias detected!"

    log_returns = np.log(past_prices / past_prices.shift(1)).dropna()
    if len(log_returns) < window - 1:
        return np.nan
        
    return log_returns.iloc[-window:].std() * np.sqrt(252)


def _approx_heston_delta(S, K, T, r, V0, kappa, theta, xi, rho, option_type, q=0.0):
    """
    Finite difference approximation of Heston Delta using the Fourier pricer.
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
            
    dS = S * 0.01
    price_up = price_option_heston_fourier(S + dS, K, T, r, V0, kappa, theta, xi, rho, option_type, q)
    price_dn = price_option_heston_fourier(S - dS, K, T, r, V0, kappa, theta, xi, rho, option_type, q)
    return (price_up - price_dn) / (2 * dS)


def run_synthetic_backtest(
    ticker='^SPX',
    period='2y',
    initial_capital=10000.0,
    option_type='call',
    edge_threshold=0.05,        # 5% edge required to enter
    risk_free_rate=0.05,
    n_sims=10000,
    model='heston',
    jump_intensity=0.1, jump_mean=-0.05, jump_std=0.03,
    heston_V0=0.04, heston_kappa=2.0, heston_theta=0.04, heston_xi=0.3, heston_rho=-0.7,
    expiry_days_list=[30],      # List of target expirations to scan (e.g., [30, 60, 90])
    tx_cost_bps=5.0,            # 5 bps cost on option premium
    hedge_cost_bps=1.0,         # 1 bps cost on stock turnover
    slippage_pct=0.01,          # 1% slippage on option premium
    hedge_rebalance_delta_threshold=0.02,
    dividend_yield=0.0,
    seed=None
):
    
    prices = fetch_historical_prices(ticker, period)
    if prices is None:
        print(f"ERROR: No data returned for {ticker} with period {period}")
        return None
    if len(prices) < 60:
        print(f"ERROR: Not enough data for {ticker} ({len(prices)} days). Need at least 60.")
        return None

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    
    # Find active trading days (approx 1st of month)
    monthly_groups = prices.groupby([prices.index.year, prices.index.month])
    entry_dates = [group.index[0] for _, group in monthly_groups]

    capital = initial_capital
    net_capital = initial_capital
    trades = []
    trade_returns = []
    equity_points = [{'date': prices.index[0], 'value': capital, 'net_value': net_capital}]
    total_entry_costs = 0.0
    total_hedge_costs = 0.0

    for entry_date in entry_dates:
        # 1. No Look-Ahead Vol calculation
        try:
            sigma = _calculate_rolling_vol_no_lookahead(prices, entry_date, window=30)
        except AssertionError:
            continue
            
        if np.isnan(sigma) or sigma <= 0.01:
            continue
            
        sigma = float(min(max(sigma, 0.05), 2.0))
        
        entry_idx = prices.index.get_loc(entry_date)
        S0 = float(prices.iloc[entry_idx])
        K = S0 
        
        # Phase 7: Scan term structure and pick highest edge
        best_edge = -float('inf')
        best_exp = None
        best_fair_p = 0
        best_mkt_p = 0
        best_expiry_idx = 0
        
        # Ensure expiry_days_list is a list
        if not isinstance(expiry_days_list, (list, tuple)):
            expiry_days_list = [expiry_days_list]
            
        for expiry_days in expiry_days_list:
            expiry_idx = min(entry_idx + int(expiry_days), len(prices) - 1)
            if expiry_idx <= entry_idx:
                continue
                
            T_initial = int(expiry_days) / 365.0
            
            # 2. SEVER BS DEPENDENCY. Price theoretically via Heston.
            if model == 'heston':
                V0_est = sigma ** 2
                try:
                    fair_p = price_option_heston_fourier(
                        S0, K, T_initial, risk_free_rate, 
                        V0_est, heston_kappa, heston_theta, heston_xi, heston_rho, 
                        option_type=option_type, q=dividend_yield
                    )
                except Exception:
                    continue
            else:
                # Fallback to Jump Diffusion MC
                sim_seed = int(seed) if seed is not None else -1
                res = simulate_jump_diffusion(
                    S0, T_initial, risk_free_rate, sigma, n_sims,
                    jump_intensity, jump_mean, jump_std, seed=sim_seed, q=dividend_yield
                )
                payoffs = np.maximum(res[0] - K, 0) if option_type == 'call' else np.maximum(K - res[0], 0)
                fair_p = float(np.exp(-risk_free_rate * T_initial) * np.mean(payoffs))
    
            # Proxy for "Market Price" (Standard BS)
            from src.core.black_scholes import black_scholes_price
            try:
                mkt_p = black_scholes_price(S0, K, T_initial, risk_free_rate, sigma, option_type, q=dividend_yield)
            except:
                mkt_p = fair_p
    
            # Relative pricing edge versus the proxy market price
            current_edge = (fair_p - mkt_p) / mkt_p if mkt_p > 0 else 0
            if current_edge > best_edge:
                best_edge = current_edge
                best_exp = int(expiry_days)
                best_fair_p = fair_p
                best_mkt_p = mkt_p
                best_expiry_idx = expiry_idx

        # Edge Entry Gating against the best opportunity
        if best_edge < edge_threshold or best_exp is None:
            equity_points.append({'date': entry_date, 'value': capital, 'net_value': net_capital})
            continue

        expiry_idx = best_expiry_idx
        expiry_date = prices.index[expiry_idx]
        T_initial = best_exp / 365.0
        fair_p = best_fair_p
        mkt_p = best_mkt_p
        edge = best_edge

        # Real Execution Price (with slippage)
        exec_price = mkt_p * (1 + slippage_pct)
        entry_cost_tx = exec_price * (tx_cost_bps / 10000.0)
        total_entry_costs += entry_cost_tx
        
        if exec_price > net_capital or exec_price <= 0.01:
            equity_points.append({'date': entry_date, 'value': capital, 'net_value': net_capital})
            continue

        # Daily delta hedging loop
        cash_account = -exec_price
        net_cash_account = -exec_price - entry_cost_tx
        shares_held = 0.0
        total_hedge_cost = 0.0
        
        path_prices = prices.iloc[entry_idx : expiry_idx + 1]
        
        for i in range(len(path_prices) - 1):
            current_date = path_prices.index[i]
            S_t = float(path_prices.iloc[i])
            T_t = (expiry_idx - (entry_idx + i)) / 365.0
            
            if model == 'heston':
                delta_t = _approx_heston_delta(
                    S_t, K, T_t, risk_free_rate, V0_est, 
                    heston_kappa, heston_theta, heston_xi, heston_rho, option_type, q=dividend_yield
                )
            else:
                # BS Delta proxy for JD
                from scipy.stats import norm
                if T_t > 0:
                    d1 = (np.log(S_t / K) + (risk_free_rate + 0.5 * sigma**2) * T_t) / (sigma * np.sqrt(T_t))
                    if option_type == 'call':
                        delta_t = norm.cdf(d1)
                    else:
                        delta_t = norm.cdf(d1) - 1.0
                else:
                    if option_type == 'call':
                        delta_t = 1.0 if S_t > K else 0.0
                    else:
                        delta_t = -1.0 if S_t < K else 0.0
            
            # Rebalance
            shares_needed = delta_t - shares_held
            if abs(shares_needed) < hedge_rebalance_delta_threshold:
                continue
            trade_cost = shares_needed * S_t
            
            # Hedge transaction costs
            h_cost = abs(shares_needed * S_t) * (hedge_cost_bps / 10000.0)
            total_hedge_cost += h_cost
            total_hedge_costs += h_cost
            
            cash_account -= trade_cost
            net_cash_account -= (trade_cost + h_cost)
            shares_held = delta_t
            
        # Expiration Settlement
        S_T_actual = float(path_prices.iloc[-1])
        payoff = max(S_T_actual - K, 0) if option_type == 'call' else max(K - S_T_actual, 0)
        
        # Liquidate shares
        share_value = shares_held * S_T_actual
        cash_account += share_value
        net_cash_account += share_value
        shares_held = 0.0
        
        # Receive payoff
        cash_account += payoff
        net_cash_account += payoff
        
        pnl = cash_account
        net_pnl = net_cash_account
        
        capital_before = net_capital
        capital += pnl
        net_capital += net_pnl
        if capital_before > 0:
            trade_returns.append(net_pnl / capital_before)
        
        trades.append({
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'expiry_date': expiry_date.strftime('%Y-%m-%d'),
            'type': f'{option_type.upper()}_HEDGED',
            'spot_price': round(S0, 2),
            'strike': round(K, 2),
            'dte': best_exp,
            'volatility': round(sigma * 100, 1),
            'mc_fair_value': round(fair_p, 4),
            'market_proxy_price': round(mkt_p, 4),
            'execution_price': round(exec_price, 4),
            'entry_edge_pct': round(edge * 100, 2),
            'actual_S_T': round(S_T_actual, 2),
            'payoff': round(payoff, 4),
            'pnl_gross': round(pnl, 4),
            'pnl_net': round(net_pnl, 4),
            'entry_tx_cost': round(entry_cost_tx, 4),
            'hedge_cost': round(total_hedge_cost, 4),
            'result': 'WIN' if net_pnl > 0 else 'LOSS',
            'balance': round(net_capital, 2)
        })

        equity_points.append({'date': expiry_date, 'value': round(capital, 2), 'net_value': round(net_capital, 2)})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_points)

    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl_net'] > 0]) if trades else 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    trade_pnls = [t['pnl_net'] for t in trades] if trades else []
    gross_trade_pnls = [t['pnl_gross'] for t in trades] if trades else []
    trade_costs = [(t['pnl_gross'] - t['pnl_net']) for t in trades] if trades else []
    
    # Calculate Sharpe on net trade returns, annualized by trade frequency
    if len(trade_returns) > 1:
        mean_ret = np.mean(trade_returns)
        std_ret = np.std(trade_returns, ddof=1)
        if std_ret > 0 and len(equity_points) > 1:
            total_days = (equity_points[-1]['date'] - equity_points[0]['date']).days
            years = total_days / 365.0 if total_days > 0 else 0.0
            ann_factor = (total_trades / years) if years > 0 else 0.0
            sharpe = (mean_ret / std_ret) * np.sqrt(ann_factor) if ann_factor > 0 else 0.0
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
        
    # Calculate Max DD
    eq = np.array([pt['net_value'] for pt in equity_points], dtype=float)
    if len(eq) > 1:
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak * 100
        max_dd = float(np.min(drawdown))
    else:
        max_dd = 0.0

    # Cost sensitivity analysis from realized trades (no re-simulation)
    sensitivity_multipliers = [0.5, 1.0, 1.5]
    cost_sensitivity = {}
    for mult in sensitivity_multipliers:
        scenario_final = initial_capital + sum(g - mult * c for g, c in zip(gross_trade_pnls, trade_costs))
        scenario_return = ((scenario_final - initial_capital) / initial_capital) * 100
        cost_sensitivity[f"cost_x{mult:.1f}"] = {
            'final_value': round(float(scenario_final), 2),
            'return_pct': round(float(scenario_return), 2)
        }

    avg_entry_edge = float(np.mean([t['entry_edge_pct'] for t in trades])) if trades else 0.0
    total_cost_impact = float(capital - net_capital)

    return {
        'trades_df': trades_df,
        'equity_curve': equity_df,
        'final_value': round(net_capital, 2),
        'gross_value': round(capital, 2),
        'total_return_pct': round(((net_capital - initial_capital) / initial_capital) * 100, 2),
        'win_rate': round(win_rate, 1),
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'sharpe_ratio': round(sharpe, 2),
        'max_drawdown_pct': round(max_dd, 1),
        'cost_summary': {
            'total_entry_costs': round(total_entry_costs, 2),
            'total_hedge_costs': round(total_hedge_costs, 2),
            'total_cost_impact': round(total_cost_impact, 2),
            'avg_entry_edge_pct': round(avg_entry_edge, 2),
            'hedge_rebalance_delta_threshold': hedge_rebalance_delta_threshold
        },
        'cost_sensitivity': cost_sensitivity,
        'methodology': {
            'fair_value_model': model,
            'entry_market_price_source': 'black_scholes_proxy_from_rolling_vol',
            'uses_historical_option_quotes': False,
            'lookahead_guard': True,
            'delta_hedged': True,
            'disclosure': 'Controlled research simulator: model fair value is compared against a Black-Scholes proxy entry price derived from no-look-ahead rolling volatility, not historical option quotes.'
        },
    }
