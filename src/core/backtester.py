"""
backtester.py

Research backtester for model-driven option trades.

Includes a historical quote mode using local SPX options bid/ask data and
an older synthetic proxy mode retained for backward compatibility.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from src.core.heston_model import price_option_heston_fourier
from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.lsv_model import simulate_lsv_paths


DEFAULT_HISTORICAL_OPTIONS_CSV = Path(__file__).resolve().parents[2] / "combined_options_data.csv"
SUPPORTED_HISTORICAL_OPTION_TICKERS = {"SPX", "^SPX"}
HISTORICAL_OPTION_MULTIPLIER = 100
DEFAULT_MAX_SPREAD_PCT = 0.35
DEFAULT_MIN_BID = 0.05
FULL_HISTORICAL_PERIOD = "full_csv"


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
    delta = (price_up - price_dn) / (2 * dS)
    if option_type == 'call':
        return float(np.clip(delta, 0.0, 1.0))
    return float(np.clip(delta, -1.0, 0.0))


def has_historical_option_quotes(csv_path=DEFAULT_HISTORICAL_OPTIONS_CSV):
    return Path(csv_path).exists()


@lru_cache(maxsize=2)
def load_historical_option_quotes(csv_path=str(DEFAULT_HISTORICAL_OPTIONS_CSV)):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Historical options CSV not found: {path}")

    columns = [
        "QUOTE_DATE",
        "UNDERLYING_LAST",
        "EXPIRE_DATE",
        "DTE",
        "STRIKE",
        "C_BID",
        "C_ASK",
        "C_IV",
        "C_DELTA",
        "C_LAST",
        "C_VOLUME",
        "P_BID",
        "P_ASK",
        "P_IV",
        "P_DELTA",
        "P_LAST",
        "P_VOLUME",
    ]
    df = pd.read_csv(
        path,
        usecols=columns,
        parse_dates=["QUOTE_DATE", "EXPIRE_DATE"],
        dayfirst=True,
    )
    numeric_cols = [col for col in columns if col not in {"QUOTE_DATE", "EXPIRE_DATE"}]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"]).dt.tz_localize(None).dt.normalize()
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE", "UNDERLYING_LAST", "STRIKE", "DTE"])
    return df.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).reset_index(drop=True)


def get_historical_option_quote_range(csv_path=DEFAULT_HISTORICAL_OPTIONS_CSV):
    quotes = load_historical_option_quotes(str(csv_path))
    if quotes.empty:
        return None, None
    return (
        pd.Timestamp(quotes["QUOTE_DATE"].min()).normalize(),
        pd.Timestamp(quotes["QUOTE_DATE"].max()).normalize(),
    )


def _resolve_period_window(index_dates, period):
    if len(index_dates) == 0:
        raise ValueError("Cannot resolve period on an empty date index.")

    if period in {None, "", FULL_HISTORICAL_PERIOD}:
        normalized = pd.to_datetime(pd.Index(index_dates)).tz_localize(None).normalize()
        return pd.Timestamp(normalized.min()), pd.Timestamp(normalized.max())

    period_map = {
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "5y": pd.DateOffset(years=5),
    }
    latest = pd.Timestamp(max(index_dates)).normalize()
    offset = period_map.get(period, pd.DateOffset(years=2))
    start = (latest - offset).normalize()
    return start, latest


def _option_quote_columns(option_type):
    prefix = "C" if option_type == "call" else "P"
    return {
        "bid": f"{prefix}_BID",
        "ask": f"{prefix}_ASK",
        "iv": f"{prefix}_IV",
        "last": f"{prefix}_LAST",
        "volume": f"{prefix}_VOLUME",
        "delta": f"{prefix}_DELTA",
    }


def _valid_entry_rows(
    entry_rows,
    option_type,
    max_spread_pct=DEFAULT_MAX_SPREAD_PCT,
    min_bid=DEFAULT_MIN_BID,
):
    cols = _option_quote_columns(option_type)
    bid_col = cols["bid"]
    ask_col = cols["ask"]
    iv_col = cols["iv"]

    valid = entry_rows.dropna(subset=[bid_col, ask_col, "UNDERLYING_LAST", "STRIKE", "DTE", "EXPIRE_DATE"]).copy()
    valid["market_mid"] = (valid[bid_col] + valid[ask_col]) / 2.0
    valid["spread"] = valid[ask_col] - valid[bid_col]
    valid["spread_pct"] = np.where(valid["market_mid"] > 0, valid["spread"] / valid["market_mid"], np.nan)
    valid = valid[
        (valid[ask_col] > valid[bid_col])
        & (valid[bid_col] >= min_bid)
        & (valid["market_mid"] > 0)
        & (valid["spread_pct"] <= max_spread_pct)
        & (valid["DTE"] > 0)
        & pd.notna(valid[iv_col])
        & (valid[iv_col] > 0)
    ]
    return valid


def _row_to_contract(chosen, option_type):
    cols = _option_quote_columns(option_type)
    bid_col = cols["bid"]
    ask_col = cols["ask"]
    iv_col = cols["iv"]
    last_col = cols["last"]
    volume_col = cols["volume"]
    delta_col = cols["delta"]
    return {
        "quote_date": pd.Timestamp(chosen["QUOTE_DATE"]).normalize(),
        "expiry_date": pd.Timestamp(chosen["EXPIRE_DATE"]).normalize(),
        "spot": float(chosen["UNDERLYING_LAST"]),
        "strike": float(chosen["STRIKE"]),
        "dte": float(chosen["DTE"]),
        "market_bid": float(chosen[bid_col]),
        "market_ask": float(chosen[ask_col]),
        "market_mid": float(chosen["market_mid"]),
        "market_spread": float(chosen["spread"]),
        "market_spread_pct": float(chosen["spread_pct"]),
        "market_iv": float(chosen[iv_col]) if pd.notna(chosen[iv_col]) else np.nan,
        "market_last": float(chosen[last_col]) if pd.notna(chosen[last_col]) else np.nan,
        "market_volume": float(chosen[volume_col]) if pd.notna(chosen[volume_col]) else np.nan,
        "market_delta": float(chosen[delta_col]) if pd.notna(chosen[delta_col]) else np.nan,
    }


def _select_contract_for_target_dte(
    entry_rows,
    option_type,
    target_dte,
    max_spread_pct=DEFAULT_MAX_SPREAD_PCT,
    min_bid=DEFAULT_MIN_BID,
):
    valid = _valid_entry_rows(entry_rows, option_type, max_spread_pct=max_spread_pct, min_bid=min_bid)
    if valid.empty:
        return None

    expiries = valid[["EXPIRE_DATE", "DTE"]].drop_duplicates().copy()
    expiries["dte_gap"] = (expiries["DTE"] - float(target_dte)).abs()
    expiry_choice = expiries.sort_values(["dte_gap", "DTE", "EXPIRE_DATE"]).iloc[0]

    expiry_rows = valid[valid["EXPIRE_DATE"] == expiry_choice["EXPIRE_DATE"]].copy()
    spot = float(expiry_rows["UNDERLYING_LAST"].iloc[0])
    expiry_rows["strike_gap"] = (expiry_rows["STRIKE"] - spot).abs()
    chosen = expiry_rows.sort_values(["strike_gap", "spread_pct", "spread", "STRIKE"]).iloc[0]
    return _row_to_contract(chosen, option_type)


def _list_contract_candidates(entry_rows, option_type, max_spread_pct=DEFAULT_MAX_SPREAD_PCT, min_bid=DEFAULT_MIN_BID):
    valid = _valid_entry_rows(entry_rows, option_type, max_spread_pct=max_spread_pct, min_bid=min_bid)
    if valid.empty:
        return []

    valid = valid.copy()
    valid["strike_gap"] = (valid["STRIKE"] - valid["UNDERLYING_LAST"]).abs()
    valid = valid.sort_values(["strike_gap", "spread_pct", "DTE", "STRIKE"])
    return [_row_to_contract(row, option_type) for _, row in valid.iterrows()]


def _build_contract_history(quotes, option_type, expiry_date, strike):
    cols = _option_quote_columns(option_type)
    history = quotes[(quotes["EXPIRE_DATE"] == expiry_date) & (quotes["STRIKE"] == strike)].copy()
    if history.empty:
        return pd.DataFrame()

    history["bid"] = pd.to_numeric(history[cols["bid"]], errors="coerce")
    history["ask"] = pd.to_numeric(history[cols["ask"]], errors="coerce")
    history["mid"] = (history["bid"] + history["ask"]) / 2.0
    history["iv"] = pd.to_numeric(history[cols["iv"]], errors="coerce")
    history["last"] = pd.to_numeric(history[cols["last"]], errors="coerce")
    history["volume"] = pd.to_numeric(history[cols["volume"]], errors="coerce")
    history["delta"] = pd.to_numeric(history[cols["delta"]], errors="coerce")
    history = history[(history["ask"] >= history["bid"]) & (history["ask"] > 0)].copy()
    history = history.sort_values("QUOTE_DATE").drop_duplicates(subset=["QUOTE_DATE"], keep="last")
    return history.set_index("QUOTE_DATE")[["bid", "ask", "mid", "iv", "last", "volume", "delta"]]


def _quote_asof(contract_history, current_date):
    if contract_history.empty:
        return None
    available = contract_history.loc[:current_date]
    if available.empty:
        return None
    row = available.iloc[-1]
    return {
        "bid": float(row["bid"]) if pd.notna(row["bid"]) else np.nan,
        "ask": float(row["ask"]) if pd.notna(row["ask"]) else np.nan,
        "mid": float(row["mid"]) if pd.notna(row["mid"]) else np.nan,
        "iv": float(row["iv"]) if pd.notna(row["iv"]) else np.nan,
        "last": float(row["last"]) if pd.notna(row["last"]) else np.nan,
        "volume": float(row["volume"]) if pd.notna(row["volume"]) else np.nan,
        "delta": float(row["delta"]) if pd.notna(row["delta"]) else np.nan,
    }


def _price_and_delta(
    model,
    option_type,
    S,
    K,
    T,
    risk_free_rate,
    sigma,
    n_sims,
    jump_intensity,
    jump_mean,
    jump_std,
    heston_kappa,
    heston_theta,
    heston_xi,
    heston_rho,
    leverage_matrix,
    leverage_strikes,
    leverage_maturities,
    dividend_yield,
    seed,
):
    if model == "heston":
        V0_est = sigma ** 2
        fair_p = price_option_heston_fourier(
            S, K, T, risk_free_rate,
            V0_est, heston_kappa, heston_theta, heston_xi, heston_rho,
            option_type=option_type, q=dividend_yield
        )
        delta_t = _approx_heston_delta(
            S, K, T, risk_free_rate, V0_est,
            heston_kappa, heston_theta, heston_xi, heston_rho, option_type, q=dividend_yield
        )
        return float(fair_p), float(delta_t)

    if model == "lsv":
        V0_est = sigma ** 2
        if leverage_matrix is None or leverage_strikes is None or leverage_maturities is None:
            lsv_leverage = np.ones((100, 100))
            lsv_strikes = np.linspace(S * 0.5, S * 1.5, 100)
            lsv_maturities = np.linspace(0.01, max(T, 0.02), 100)
        else:
            lsv_leverage = leverage_matrix
            lsv_strikes = leverage_strikes
            lsv_maturities = leverage_maturities

        paths, _ = simulate_lsv_paths(
            S, T, risk_free_rate, V0_est, heston_kappa, heston_theta, heston_xi, heston_rho,
            lsv_leverage, lsv_strikes, lsv_maturities,
            n_paths=n_sims, n_steps=50, q=dividend_yield, seed=int(seed) if seed is not None else -1
        )
        S_T = paths[:, -1]
        payoffs = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)
        fair_p = float(np.exp(-risk_free_rate * T) * np.mean(payoffs))

        # Use Heston delta as the hedge proxy for LSV to keep hedge costs stable.
        delta_t = _approx_heston_delta(
            S, K, T, risk_free_rate, V0_est,
            heston_kappa, heston_theta, heston_xi, heston_rho, option_type, q=dividend_yield
        )
        return fair_p, float(delta_t)

    from scipy.stats import norm

    sim_seed = int(seed) if seed is not None else -1
    res = simulate_jump_diffusion(
        S, T, risk_free_rate, sigma, n_sims,
        jump_intensity, jump_mean, jump_std, seed=sim_seed, q=dividend_yield
    )
    payoffs = np.maximum(res[0] - K, 0) if option_type == "call" else np.maximum(K - res[0], 0)
    fair_p = float(np.exp(-risk_free_rate * T) * np.mean(payoffs))

    if T > 0:
        d1 = (np.log(S / K) + (risk_free_rate + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta_t = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
    else:
        if option_type == "call":
            delta_t = 1.0 if S > K else 0.0
        else:
            delta_t = -1.0 if S < K else 0.0

    return fair_p, float(delta_t)


def _hedge_target_shares(option_delta, contracts, option_multiplier):
    """Short the option delta for long options; negative put deltas become long underlying hedges."""
    return -float(option_delta) * float(option_multiplier) * float(contracts)


def run_historical_quotes_backtest(
    ticker='^SPX',
    period=FULL_HISTORICAL_PERIOD,
    initial_capital=10000.0,
    option_type='call',
    edge_threshold=0.05,
    risk_free_rate=0.05,
    n_sims=10000,
    model='heston',
    jump_intensity=0.1,
    jump_mean=-0.05,
    jump_std=0.03,
    heston_V0=0.04,
    heston_kappa=2.0,
    heston_theta=0.04,
    heston_xi=0.3,
    heston_rho=-0.7,
    leverage_matrix=None,
    leverage_strikes=None,
    leverage_maturities=None,
    expiry_days_list=None,
    tx_cost_bps=5.0,
    hedge_cost_bps=1.0,
    slippage_pct=0.01,
    hedge_rebalance_delta_threshold=0.02,
    dividend_yield=0.0,
    seed=None,
    csv_path=DEFAULT_HISTORICAL_OPTIONS_CSV,
    option_multiplier=HISTORICAL_OPTION_MULTIPLIER,
    max_spread_pct=DEFAULT_MAX_SPREAD_PCT,
    min_quote_bid=DEFAULT_MIN_BID,
    hedge_margin_ratio=0.15,
    max_open_positions=1,
    liquidate_on_equity_breach=True,
):
    if str(ticker).upper() not in SUPPORTED_HISTORICAL_OPTION_TICKERS:
        raise ValueError(f"Historical quote backtest currently supports {sorted(SUPPORTED_HISTORICAL_OPTION_TICKERS)} only.")

    quotes = load_historical_option_quotes(str(csv_path))
    if quotes.empty:
        return None

    start_date, end_date = _resolve_period_window(quotes["QUOTE_DATE"], period)
    quotes = quotes[(quotes["QUOTE_DATE"] >= start_date) & (quotes["QUOTE_DATE"] <= end_date)].copy()
    if quotes.empty:
        return None

    underlying = (
        quotes.groupby("QUOTE_DATE", sort=True)["UNDERLYING_LAST"]
        .first()
        .astype(float)
        .sort_index()
    )
    if len(underlying) < 60:
        return None

    monthly_groups = underlying.groupby([underlying.index.year, underlying.index.month])
    entry_dates = [group.index[0] for _, group in monthly_groups]
    entry_date_set = set(entry_dates)

    free_cash_gross = initial_capital
    free_cash_net = initial_capital
    trades = []
    equity_points = []
    total_entry_costs = 0.0
    total_hedge_costs = 0.0
    open_positions = []
    forced_liquidations = 0
    solvency_breach_triggered = False

    def current_equity(spot_price, current_date):
        gross = free_cash_gross
        net = free_cash_net
        for position in open_positions:
            quote = _quote_asof(position["quote_history"], current_date)
            option_value = position["contracts"] * option_multiplier * (
                quote["mid"] if quote and np.isfinite(quote["mid"]) and current_date < position["settlement_date"]
                else max(spot_price - position["strike"], 0) if position["option_type"] == "call"
                else max(position["strike"] - spot_price, 0)
            )
            hedge_value = position["hedge_pnl_gross"]
            hedge_value_net = position["hedge_pnl_net"]
            margin_reserve = position["hedge_margin_reserve"]
            gross += option_value + hedge_value + margin_reserve
            net += option_value + hedge_value_net + margin_reserve
        return gross, net

    def close_position(position, current_date, spot_price, exit_reason="expiry"):
        nonlocal free_cash_gross, free_cash_net, forced_liquidations

        quote = _quote_asof(position["quote_history"], current_date)
        intrinsic = max(spot_price - position["strike"], 0) if position["option_type"] == "call" else max(position["strike"] - spot_price, 0)
        if exit_reason == "expiry":
            option_cashflow = intrinsic * option_multiplier * position["contracts"]
        elif quote and np.isfinite(quote["bid"]) and current_date < position["settlement_date"]:
            option_cashflow = quote["bid"] * option_multiplier * position["contracts"]
        else:
            option_cashflow = intrinsic * option_multiplier * position["contracts"]

        final_gross = option_cashflow + position["hedge_pnl_gross"] + position["hedge_margin_reserve"]
        final_net = option_cashflow + position["hedge_pnl_net"] + position["hedge_margin_reserve"]
        free_cash_gross += final_gross
        free_cash_net += final_net

        pnl = final_gross - position["entry_cash_commitment_gross"]
        net_pnl = final_net - position["entry_cash_commitment_net"]
        if position in open_positions:
            open_positions.remove(position)
        if exit_reason == "equity_breach":
            forced_liquidations += 1
        portfolio_balance = current_equity(spot_price, current_date)[1]
        trades.append({
            'entry_date': position["entry_date"].strftime('%Y-%m-%d'),
            'exit_date': current_date.strftime('%Y-%m-%d'),
            'expiry_date': position["expiry_date"].strftime('%Y-%m-%d'),
            'exit_reason': exit_reason,
            'type': f'{position["option_type"].upper()}_HEDGED',
            'spot_price': round(position["spot_price"], 2),
            'strike': round(position["strike"], 2),
            'contracts': int(position["contracts"]),
            'option_multiplier': int(option_multiplier),
            'entry_notional': round(position["entry_notional_net"], 2),
            'entry_cash_commitment': round(position["entry_cash_commitment_net"], 2),
            'target_dte': round(float(position["target_dte"]), 1),
            'actual_dte': round(float(position["actual_dte"]), 1),
            'volatility': round(position["sigma"] * 100, 1),
            'mc_fair_value': round(position["fair_value"], 4),
            'market_bid': round(position["market_bid"], 4),
            'market_ask': round(position["market_ask"], 4),
            'market_mid': round(position["market_mid"], 4),
            'market_iv': round(position["market_iv"], 4) if not np.isnan(position["market_iv"]) else np.nan,
            'market_spread_pct': round(position["market_spread_pct"] * 100, 2),
            'execution_price': round(position["execution_price"], 4),
            'entry_edge_pct': round(position["entry_edge_pct"], 2),
            'initial_hedge_shares': round(position["initial_hedge_shares"], 4),
            'actual_S_T': round(spot_price, 2),
            'payoff': round(option_cashflow, 4),
            'pnl_gross': round(pnl, 4),
            'pnl_net': round(net_pnl, 4),
            'entry_tx_cost': round(position["entry_tx_cost"], 4),
            'hedge_cost': round(position["hedge_cost_total"], 4),
            'hedge_margin_reserve': round(position["hedge_margin_reserve"], 4),
            'result': 'WIN' if net_pnl > 0 else 'LOSS',
            'balance': round(portfolio_balance, 2),
        })

    for current_date, spot in underlying.items():
        spot = float(spot)

        for position in list(open_positions):
            if current_date < position["entry_date"]:
                continue

            if current_date > position["entry_date"]:
                hedge_move = position["hedge_shares"] * (spot - position["last_hedge_spot"])
                position["hedge_pnl_gross"] += hedge_move
                position["hedge_pnl_net"] += hedge_move
                position["last_hedge_spot"] = spot

            T_t = max((position["expiry_date"] - current_date).days / 365.0, 0.0)
            _, delta_t = _price_and_delta(
                model=model,
                option_type=position["option_type"],
                S=spot,
                K=position["strike"],
                T=T_t,
                risk_free_rate=risk_free_rate,
                sigma=position["sigma"],
                n_sims=n_sims,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_std=jump_std,
                heston_kappa=heston_kappa,
                heston_theta=heston_theta,
                heston_xi=heston_xi,
                heston_rho=heston_rho,
                leverage_matrix=leverage_matrix,
                leverage_strikes=leverage_strikes,
                leverage_maturities=leverage_maturities,
                dividend_yield=dividend_yield,
                seed=seed,
            )
            target_shares = _hedge_target_shares(delta_t, position["contracts"], option_multiplier)
            shares_needed = target_shares - position["hedge_shares"]
            delta_gap = abs(shares_needed) / max(option_multiplier * position["contracts"], 1)
            if delta_gap >= hedge_rebalance_delta_threshold:
                hedge_cost = abs(shares_needed * spot) * (hedge_cost_bps / 10000.0)
                position["hedge_pnl_net"] -= hedge_cost
                position["hedge_shares"] = target_shares
                position["hedge_cost_total"] += hedge_cost
                total_hedge_costs += hedge_cost

            if current_date >= position["settlement_date"]:
                close_position(position, current_date, spot, exit_reason="expiry")

        if liquidate_on_equity_breach and open_positions:
            gross_equity, net_equity = current_equity(spot, current_date)
            if net_equity <= 0 or gross_equity <= 0:
                solvency_breach_triggered = True
                for position in list(open_positions):
                    close_position(position, current_date, spot, exit_reason="equity_breach")
                free_cash_gross = max(free_cash_gross, 0.0)
                free_cash_net = max(free_cash_net, 0.0)

        if current_date in entry_date_set and len(open_positions) < max_open_positions and free_cash_net > 0:
            sigma = _calculate_rolling_vol_no_lookahead(underlying, current_date, window=30)
            if not np.isnan(sigma) and sigma > 0.01:
                sigma = float(min(max(sigma, 0.05), 2.0))
                entry_rows = quotes[quotes["QUOTE_DATE"] == current_date]
                best_candidate = None

                if isinstance(expiry_days_list, (list, tuple)) and len(expiry_days_list) > 0:
                    contract_candidates = []
                    for expiry_days in expiry_days_list:
                        contract = _select_contract_for_target_dte(
                            entry_rows,
                            option_type,
                            expiry_days,
                            max_spread_pct=max_spread_pct,
                            min_bid=min_quote_bid,
                        )
                        if contract is not None:
                            contract_candidates.append((contract, expiry_days))
                else:
                    contract_candidates = [(contract, None) for contract in _list_contract_candidates(
                        entry_rows,
                        option_type,
                        max_spread_pct=max_spread_pct,
                        min_bid=min_quote_bid,
                    )]

                for contract, target_dte in contract_candidates:
                    T_initial = max(contract["dte"], 1.0) / 365.0
                    fair_p, entry_delta = _price_and_delta(
                        model=model,
                        option_type=option_type,
                        S=contract["spot"],
                        K=contract["strike"],
                        T=T_initial,
                        risk_free_rate=risk_free_rate,
                        sigma=sigma,
                        n_sims=n_sims,
                        jump_intensity=jump_intensity,
                        jump_mean=jump_mean,
                        jump_std=jump_std,
                        heston_kappa=heston_kappa,
                        heston_theta=heston_theta,
                        heston_xi=heston_xi,
                        heston_rho=heston_rho,
                        leverage_matrix=leverage_matrix,
                        leverage_strikes=leverage_strikes,
                        leverage_maturities=leverage_maturities,
                        dividend_yield=dividend_yield,
                        seed=seed,
                    )
                    settlement_candidates = underlying.index[underlying.index <= contract["expiry_date"]]
                    if len(settlement_candidates) == 0:
                        continue
                    settlement_date = settlement_candidates[-1]
                    if settlement_date <= current_date:
                        continue

                    execution_price = contract["market_ask"] * (1 + slippage_pct)
                    if execution_price <= 0:
                        continue
                    current_edge = (fair_p - execution_price) / execution_price
                    if current_edge < edge_threshold:
                        continue

                    entry_tx_cost_per_contract = execution_price * option_multiplier * (tx_cost_bps / 10000.0)
                    hedge_margin_per_contract = abs(_hedge_target_shares(entry_delta, 1, option_multiplier) * contract["spot"]) * hedge_margin_ratio
                    entry_notional_gross_per_contract = execution_price * option_multiplier
                    entry_notional_net_per_contract = entry_notional_gross_per_contract + entry_tx_cost_per_contract
                    entry_cash_commitment_per_contract = entry_notional_net_per_contract + hedge_margin_per_contract
                    contracts = int(np.floor(free_cash_net / entry_cash_commitment_per_contract)) if entry_cash_commitment_per_contract > 0 else 0
                    if contracts < 1:
                        continue

                    if best_candidate is None or current_edge > best_candidate["edge"]:
                        best_candidate = {
                            "contract": contract,
                            "fair_value": fair_p,
                            "target_dte": target_dte if target_dte is not None else contract["dte"],
                            "entry_delta": entry_delta,
                            "edge": current_edge,
                            "settlement_date": settlement_date,
                            "execution_price": execution_price,
                            "entry_tx_cost_per_contract": entry_tx_cost_per_contract,
                            "hedge_margin_per_contract": hedge_margin_per_contract,
                            "entry_notional_gross_per_contract": entry_notional_gross_per_contract,
                            "entry_notional_net_per_contract": entry_notional_net_per_contract,
                            "entry_cash_commitment_per_contract": entry_cash_commitment_per_contract,
                            "contracts": contracts,
                        }

                if best_candidate is not None:
                    best_contract = best_candidate["contract"]
                    contracts = best_candidate["contracts"]
                    hedge_margin_reserve = best_candidate["hedge_margin_per_contract"] * contracts
                    initial_hedge_shares = _hedge_target_shares(best_candidate["entry_delta"], contracts, option_multiplier)
                    initial_hedge_cost = 0.0
                    initial_delta_gap = abs(initial_hedge_shares) / max(option_multiplier * contracts, 1)
                    if initial_delta_gap >= hedge_rebalance_delta_threshold:
                        initial_hedge_cost = abs(initial_hedge_shares * spot) * (hedge_cost_bps / 10000.0)

                    free_cash_gross -= (best_candidate["entry_notional_gross_per_contract"] * contracts + hedge_margin_reserve)
                    free_cash_net -= (best_candidate["entry_notional_net_per_contract"] * contracts + hedge_margin_reserve + initial_hedge_cost)
                    total_entry_costs += best_candidate["entry_tx_cost_per_contract"] * contracts
                    total_hedge_costs += initial_hedge_cost
                    open_positions.append({
                        "entry_date": current_date,
                        "settlement_date": best_candidate["settlement_date"],
                        "expiry_date": best_contract["expiry_date"],
                        "option_type": option_type,
                        "contracts": contracts,
                        "spot_price": best_contract["spot"],
                        "strike": best_contract["strike"],
                        "sigma": sigma,
                        "fair_value": best_candidate["fair_value"],
                        "target_dte": best_candidate["target_dte"],
                        "actual_dte": best_contract["dte"],
                        "market_bid": best_contract["market_bid"],
                        "market_ask": best_contract["market_ask"],
                        "market_mid": best_contract["market_mid"],
                        "market_iv": best_contract["market_iv"],
                        "market_spread_pct": best_contract["market_spread_pct"],
                        "execution_price": best_candidate["execution_price"],
                        "entry_edge_pct": best_candidate["edge"] * 100.0,
                        "entry_tx_cost": best_candidate["entry_tx_cost_per_contract"] * contracts,
                        "hedge_margin_reserve": hedge_margin_reserve,
                        "entry_notional_gross": best_candidate["entry_notional_gross_per_contract"] * contracts,
                        "entry_notional_net": best_candidate["entry_notional_net_per_contract"] * contracts,
                        "entry_cash_commitment_gross": best_candidate["entry_notional_gross_per_contract"] * contracts + hedge_margin_reserve,
                        "entry_cash_commitment_net": best_candidate["entry_notional_net_per_contract"] * contracts + hedge_margin_reserve + initial_hedge_cost,
                        "hedge_pnl_gross": 0.0,
                        "hedge_pnl_net": 0.0,
                        "hedge_shares": initial_hedge_shares,
                        "initial_hedge_shares": initial_hedge_shares,
                        "last_hedge_spot": spot,
                        "hedge_cost_total": initial_hedge_cost,
                        "quote_history": _build_contract_history(quotes, option_type, best_contract["expiry_date"], best_contract["strike"]),
                    })

        gross_value, net_value = current_equity(spot, current_date)
        equity_points.append({'date': current_date, 'value': round(gross_value, 2), 'net_value': round(net_value, 2)})

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_df = pd.DataFrame(equity_points)
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl_net'] > 0]) if trades else 0
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    gross_trade_pnls = [t['pnl_gross'] for t in trades] if trades else []
    trade_costs = [(t['pnl_gross'] - t['pnl_net']) for t in trades] if trades else []
    if not equity_df.empty and (equity_df["net_value"] > 0).all():
        daily_returns = equity_df["net_value"].pct_change().dropna()
    else:
        daily_returns = pd.Series(dtype=float)
    if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std(ddof=1)) * np.sqrt(252))
    else:
        sharpe = np.nan if solvency_breach_triggered else 0.0

    eq = np.array([pt['net_value'] for pt in equity_points], dtype=float)
    if len(eq) > 1:
        peak = np.maximum.accumulate(eq)
        drawdown = (eq - peak) / peak * 100
        max_dd = float(np.min(drawdown))
    else:
        max_dd = 0.0

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
    gross_final = float(equity_df["value"].iloc[-1]) if not equity_df.empty else free_cash_gross
    net_final = float(equity_df["net_value"].iloc[-1]) if not equity_df.empty else free_cash_net
    total_cost_impact = float(gross_final - net_final)

    return {
        'trades_df': trades_df,
        'equity_curve': equity_df,
        'final_value': round(net_final, 2),
        'gross_value': round(gross_final, 2),
        'total_return_pct': round(((net_final - initial_capital) / initial_capital) * 100, 2),
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
            'hedge_rebalance_delta_threshold': hedge_rebalance_delta_threshold,
            'option_multiplier': int(option_multiplier),
            'hedge_margin_ratio': hedge_margin_ratio,
            'max_open_positions': int(max_open_positions),
            'forced_liquidations': int(forced_liquidations),
            'solvency_breach_triggered': bool(solvency_breach_triggered),
        },
        'cost_sensitivity': cost_sensitivity,
        'methodology': {
            'fair_value_model': model,
            'entry_market_price_source': 'historical_option_bid_ask_mid',
            'uses_historical_option_quotes': True,
            'lookahead_guard': True,
            'delta_hedged': True,
            'overlapping_positions': max_open_positions > 1,
            'max_open_positions': int(max_open_positions),
            'mark_to_market_basis': 'daily_option_mid_and_hedge_close',
            'sharpe_basis': 'daily_mark_to_mid_equity',
            'hedge_bookkeeping': 'futures_style_mark_to_market',
            'hedge_margin_ratio': hedge_margin_ratio,
            'liquidate_on_equity_breach': bool(liquidate_on_equity_breach),
            'solvency_breach_triggered': bool(solvency_breach_triggered),
            'data_start': start_date.strftime('%Y-%m-%d'),
            'data_end': end_date.strftime('%Y-%m-%d'),
            'quote_filters': {
                'min_bid': min_quote_bid,
                'max_spread_pct': max_spread_pct,
                'requires_positive_iv': True,
                'requires_uncrossed_market': True,
            },
            'disclosure': 'Daily EOD SPX option quotes: entries use historical bid/ask from combined_options_data.csv, signals are filtered against executable ask-side entry prices, positions are sized with the 100x contract multiplier plus a segregated hedge-margin reserve, concurrent positions are capped, and the engine force-liquidates if marked equity breaches zero before continuing.'
        },
    }


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
            except Exception:
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
            target_shares = -delta_t
            shares_needed = target_shares - shares_held
            if abs(shares_needed) < hedge_rebalance_delta_threshold:
                continue
            trade_cost = shares_needed * S_t
            
            # Hedge transaction costs
            h_cost = abs(shares_needed * S_t) * (hedge_cost_bps / 10000.0)
            total_hedge_cost += h_cost
            total_hedge_costs += h_cost
            
            cash_account -= trade_cost
            net_cash_account -= (trade_cost + h_cost)
            shares_held = target_shares
            
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
