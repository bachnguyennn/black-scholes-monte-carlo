import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core import backtester as bt


def _synthetic_prices(n=260, start=100.0, drift=0.0005):
    dates = pd.bdate_range("2025-01-01", periods=n)
    prices = start * np.exp(np.cumsum(np.full(n, drift)))
    return pd.Series(prices, index=dates)


def _historical_option_panel():
    dates = pd.bdate_range("2024-09-02", periods=140)
    rows = []
    for i, date in enumerate(dates):
        spot = 100.0 + 0.15 * i + 3.5 * np.sin(i / 4.0)
        expiry = date + pd.Timedelta(days=30)
        strike = round(spot)
        rows.append(
            {
                "QUOTE_DATE": date,
                "UNDERLYING_LAST": spot,
                "EXPIRE_DATE": expiry,
                "DTE": float((expiry - date).days),
                "STRIKE": float(strike),
                "C_BID": 4.0 + 0.01 * i,
                "C_ASK": 4.4 + 0.01 * i,
                "C_IV": 0.22,
                "C_DELTA": 0.52,
                "C_LAST": 4.2 + 0.01 * i,
                "C_VOLUME": 10.0,
                "P_BID": 3.8 + 0.01 * i,
                "P_ASK": 4.2 + 0.01 * i,
                "P_IV": 0.24,
                "P_DELTA": -0.48,
                "P_LAST": 4.0 + 0.01 * i,
                "P_VOLUME": 12.0,
            }
        )
    return pd.DataFrame(rows)


def _historical_option_panel_with_affordability_gap():
    dates = pd.bdate_range("2024-09-02", periods=140)
    rows = []
    for i, date in enumerate(dates):
        spot = 100.0 + 0.3 * i + 3.0 * np.sin(i / 4.0)
        expiry = date + pd.Timedelta(days=30)
        atm_strike = float(round(spot))
        cheap_strike = atm_strike + 5.0
        rows.append(
            {
                "QUOTE_DATE": date,
                "UNDERLYING_LAST": spot,
                "EXPIRE_DATE": expiry,
                "DTE": float((expiry - date).days),
                "STRIKE": atm_strike,
                "C_BID": 119.0,
                "C_ASK": 120.0,
                "C_IV": 0.22,
                "C_DELTA": 0.55,
                "C_LAST": 119.5,
                "C_VOLUME": 10.0,
                "P_BID": 3.0,
                "P_ASK": 3.2,
                "P_IV": 0.24,
                "P_DELTA": -0.45,
                "P_LAST": 3.1,
                "P_VOLUME": 10.0,
            }
        )
        rows.append(
            {
                "QUOTE_DATE": date,
                "UNDERLYING_LAST": spot,
                "EXPIRE_DATE": expiry,
                "DTE": float((expiry - date).days),
                "STRIKE": cheap_strike,
                "C_BID": 1.4,
                "C_ASK": 1.5,
                "C_IV": 0.20,
                "C_DELTA": 0.35,
                "C_LAST": 1.45,
                "C_VOLUME": 10.0,
                "P_BID": 3.5,
                "P_ASK": 3.7,
                "P_IV": 0.25,
                "P_DELTA": -0.50,
                "P_LAST": 3.6,
                "P_VOLUME": 10.0,
            }
        )
    return pd.DataFrame(rows)


def _historical_option_panel_for_breach():
    dates = pd.bdate_range("2024-09-02", periods=140)
    rows = []
    for i, date in enumerate(dates):
        spot = max(20.0, 115.0 - 0.8 * i + 4.0 * np.sin(i / 3.0))
        expiry = date + pd.Timedelta(days=30)
        strike = float(round(spot))
        rows.append(
            {
                "QUOTE_DATE": date,
                "UNDERLYING_LAST": spot,
                "EXPIRE_DATE": expiry,
                "DTE": float((expiry - date).days),
                "STRIKE": strike,
                "C_BID": 0.9,
                "C_ASK": 1.0,
                "C_IV": 0.40,
                "C_DELTA": 1.0,
                "C_LAST": 0.95,
                "C_VOLUME": 20.0,
                "P_BID": 0.9,
                "P_ASK": 1.0,
                "P_IV": 0.40,
                "P_DELTA": -1.0,
                "P_LAST": 0.95,
                "P_VOLUME": 20.0,
            }
        )
    return pd.DataFrame(rows)


def test_backtester_put_path_and_cost_reports(monkeypatch):
    prices = _synthetic_prices()
    monkeypatch.setattr(bt, "fetch_historical_prices", lambda ticker, period: prices)

    res = bt.run_synthetic_backtest(
        ticker="TEST",
        period="1y",
        option_type="put",
        model="heston",
        edge_threshold=-1.0,
        hedge_rebalance_delta_threshold=0.0,
    )

    assert res is not None
    assert "cost_summary" in res
    assert "cost_sensitivity" in res
    assert "methodology" in res
    assert res["methodology"]["uses_historical_option_quotes"] is False
    if not res["trades_df"].empty:
        assert all(res["trades_df"]["type"].str.contains("PUT_HEDGED"))
        assert "market_proxy_price" in res["trades_df"].columns
        assert "execution_price" in res["trades_df"].columns


def test_rebalance_threshold_reduces_hedge_cost(monkeypatch):
    prices = _synthetic_prices()
    monkeypatch.setattr(bt, "fetch_historical_prices", lambda ticker, period: prices)

    low_thr = bt.run_synthetic_backtest(
        ticker="TEST",
        period="1y",
        option_type="call",
        model="heston",
        edge_threshold=-1.0,
        hedge_rebalance_delta_threshold=0.0,
    )

    high_thr = bt.run_synthetic_backtest(
        ticker="TEST",
        period="1y",
        option_type="call",
        model="heston",
        edge_threshold=-1.0,
        hedge_rebalance_delta_threshold=0.5,
    )

    assert low_thr is not None and high_thr is not None
    assert high_thr["cost_summary"]["total_hedge_costs"] <= low_thr["cost_summary"]["total_hedge_costs"]


def test_historical_quote_backtest_uses_real_bid_ask(monkeypatch):
    monkeypatch.setattr(bt, "load_historical_option_quotes", lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _historical_option_panel())

    res = bt.run_historical_quotes_backtest(
        ticker="SPX",
        period=bt.FULL_HISTORICAL_PERIOD,
        option_type="call",
        model="heston",
        edge_threshold=-1.0,
        hedge_rebalance_delta_threshold=0.0,
        expiry_days_list=None,
        csv_path="ignored.csv",
    )

    assert res is not None
    assert res["methodology"]["uses_historical_option_quotes"] is True
    assert res["methodology"]["entry_market_price_source"] == "historical_option_bid_ask_mid"
    assert res["methodology"]["overlapping_positions"] is False
    assert res["methodology"]["max_open_positions"] == 1
    assert res["methodology"]["sharpe_basis"] == "daily_mark_to_mid_equity"
    assert not res["trades_df"].empty
    assert len(res["equity_curve"]) > res["total_trades"]
    assert {"market_bid", "market_ask", "market_mid", "contracts", "option_multiplier"}.issubset(res["trades_df"].columns)
    assert (res["trades_df"]["option_multiplier"] == 100).all()


def test_historical_quote_backtest_uses_short_call_hedges_and_reserved_margin(monkeypatch):
    monkeypatch.setattr(bt, "load_historical_option_quotes", lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _historical_option_panel())
    monkeypatch.setattr(bt, "_price_and_delta", lambda **kwargs: (kwargs["S"] * 0.08, 0.50))

    res = bt.run_historical_quotes_backtest(
        ticker="SPX",
        period=bt.FULL_HISTORICAL_PERIOD,
        option_type="call",
        model="heston",
        edge_threshold=0.01,
        hedge_rebalance_delta_threshold=0.0,
        csv_path="ignored.csv",
    )

    assert res is not None
    assert not res["trades_df"].empty
    assert (res["trades_df"]["initial_hedge_shares"] < 0).all()
    assert (res["trades_df"]["hedge_margin_reserve"] > 0).all()
    assert (res["trades_df"]["entry_cash_commitment"] >= res["trades_df"]["entry_notional"]).all()
    assert res["max_drawdown_pct"] >= -100.0


def test_historical_quote_backtest_falls_back_to_best_affordable_contract(monkeypatch):
    monkeypatch.setattr(
        bt,
        "load_historical_option_quotes",
        lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _historical_option_panel_with_affordability_gap(),
    )

    def _stub_price_and_delta(**kwargs):
        strike_gap = kwargs["K"] - kwargs["S"]
        if strike_gap <= 1.0:
            return 200.0, 0.55
        return 2.4, 0.35

    monkeypatch.setattr(bt, "_price_and_delta", _stub_price_and_delta)

    res = bt.run_historical_quotes_backtest(
        ticker="SPX",
        period=bt.FULL_HISTORICAL_PERIOD,
        option_type="call",
        model="heston",
        edge_threshold=0.10,
        hedge_rebalance_delta_threshold=0.0,
        csv_path="ignored.csv",
    )

    assert res is not None
    assert not res["trades_df"].empty
    assert (res["trades_df"]["strike"] > res["trades_df"]["spot_price"]).all()


def test_historical_quote_backtest_force_liquidates_on_equity_breach(monkeypatch):
    monkeypatch.setattr(
        bt,
        "load_historical_option_quotes",
        lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _historical_option_panel_for_breach(),
    )
    monkeypatch.setattr(bt, "_price_and_delta", lambda **kwargs: (5.0, 1.0))
    monkeypatch.setattr(bt, "_hedge_target_shares", lambda option_delta, contracts, option_multiplier: float(option_delta) * float(option_multiplier) * float(contracts))

    res = bt.run_historical_quotes_backtest(
        ticker="SPX",
        period=bt.FULL_HISTORICAL_PERIOD,
        option_type="call",
        model="heston",
        edge_threshold=0.01,
        hedge_rebalance_delta_threshold=0.0,
        csv_path="ignored.csv",
    )

    assert res is not None
    assert res["cost_summary"]["forced_liquidations"] > 0
    assert res["cost_summary"]["solvency_breach_triggered"] is True
    assert res["final_value"] == 0.0
    assert res["max_drawdown_pct"] >= -100.0
    assert pd.isna(res["sharpe_ratio"])
    assert "equity_breach" in set(res["trades_df"]["exit_reason"])


def _panel_atm_and_otm():
    """Each date offers a near-ATM strike and a deep-OTM strike."""
    dates = pd.bdate_range("2024-09-02", periods=140)
    rows = []
    for i, date in enumerate(dates):
        spot = 100.0 + 0.15 * i + 3.5 * np.sin(i / 4.0)  # oscillate -> nonzero realized vol
        expiry = date + pd.Timedelta(days=30)
        # (strike, bid, ask): ATM ~8% spread, OTM ~20% spread (both under the
        # 35% validity cap so both are candidates before the selection policy).
        for strike, bid, ask in [(round(spot), 4.8, 5.2), (round(spot * 1.4), 0.9, 1.1)]:
            rows.append({
                "QUOTE_DATE": date, "UNDERLYING_LAST": spot, "EXPIRE_DATE": expiry,
                "DTE": 30.0, "STRIKE": float(strike),
                "C_BID": bid, "C_ASK": ask, "C_IV": 0.20, "C_DELTA": 0.5,
                "C_LAST": (bid + ask) / 2, "C_VOLUME": 10.0,
                "P_BID": bid, "P_ASK": ask, "P_IV": 0.20, "P_DELTA": -0.5,
                "P_LAST": (bid + ask) / 2, "P_VOLUME": 10.0,
            })
    return pd.DataFrame(rows)


def _panel_rich_worthless_calls():
    """Calls quoted rich (bid 5.0) on an underlying that steadily declines, so
    the options selected at entry expire out-of-the-money (worthless)."""
    dates = pd.bdate_range("2024-09-02", periods=220)
    rows = []
    for i, date in enumerate(dates):
        spot = 140.0 - 0.4 * i + 2.0 * np.sin(i / 4.0)  # declining + wiggle
        expiry = date + pd.Timedelta(days=30)
        strike = float(round(spot))
        rows.append({
            "QUOTE_DATE": date, "UNDERLYING_LAST": spot, "EXPIRE_DATE": expiry,
            "DTE": 30.0, "STRIKE": strike,
            "C_BID": 5.0, "C_ASK": 5.4, "C_IV": 0.20, "C_DELTA": 0.5,
            "C_LAST": 5.2, "C_VOLUME": 10.0,
            "P_BID": 5.0, "P_ASK": 5.4, "P_IV": 0.20, "P_DELTA": -0.5,
            "P_LAST": 5.2, "P_VOLUME": 10.0,
        })
    return pd.DataFrame(rows)


def test_short_vol_profits_where_long_correctly_abstains(monkeypatch):
    """The calls are richer than the model's fair value, so a long buyer should
    never touch them, while a short seller collects the premium as they expire
    worthless. Verifies the short side's cashflow signs end to end."""
    monkeypatch.setattr(bt, "load_historical_option_quotes",
                        lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _panel_rich_worthless_calls())
    # Model fair value (3.0) below the 5.0 bid; delta 0 -> isolate the option
    # cashflow from hedging for a clean, deterministic check.
    monkeypatch.setattr(bt, "_price_and_delta", lambda **kwargs: (3.0, 0.0))

    common = dict(ticker="SPX", period=bt.FULL_HISTORICAL_PERIOD, option_type="call",
                  model="heston", edge_threshold=0.05, hedge_rebalance_delta_threshold=0.0,
                  initial_capital=10000.0, max_capital_fraction_per_trade=0.5,
                  csv_path="ignored.csv")

    long_res = bt.run_historical_quotes_backtest(strategy_side="long", **common)
    short_res = bt.run_historical_quotes_backtest(strategy_side="short", **common)

    # A long buyer will not pay 5.4 for something the model values at 3.0.
    assert long_res["trades_df"].empty
    assert long_res["final_value"] == 10000.0

    # The short seller collects premium on options that expire worthless.
    assert not short_res["trades_df"].empty
    assert short_res["final_value"] > 10000.0
    assert short_res["methodology"]["strategy_side"] == "short"
    assert short_res["trades_df"]["type"].str.startswith("SHORT").all()
    assert (short_res["trades_df"]["pnl_net"] > 0).all()


def test_selection_policy_keeps_near_atm_and_caps_edge(monkeypatch):
    """The moneyness band + edge cap must stop the engine from selecting the
    deep-OTM contract whose tiny premium yields a huge relative edge."""
    monkeypatch.setattr(bt, "load_historical_option_quotes",
                        lambda csv_path=str(bt.DEFAULT_HISTORICAL_OPTIONS_CSV): _panel_atm_and_otm())
    # Constant fair value: moderate edge on the ATM contract (~12%), enormous
    # relative edge on the cheap OTM contract (~430%).
    monkeypatch.setattr(bt, "_price_and_delta", lambda **kwargs: (5.9, 0.5))

    common = dict(ticker="SPX", period=bt.FULL_HISTORICAL_PERIOD, option_type="call",
                  model="heston", edge_threshold=0.05, hedge_rebalance_delta_threshold=0.0,
                  csv_path="ignored.csv")

    # No policy: engine picks the highest-edge contract -> the OTM junk.
    unguarded = bt.run_historical_quotes_backtest(**common)
    assert not unguarded["trades_df"].empty
    assert (unguarded["trades_df"]["strike"] > unguarded["trades_df"]["spot_price"] * 1.2).any()

    # With policy: only near-ATM trades survive.
    guarded = bt.run_historical_quotes_backtest(
        moneyness_band=0.07, max_edge=0.30, **common)
    assert not guarded["trades_df"].empty
    mny = guarded["trades_df"]["strike"] / guarded["trades_df"]["spot_price"]
    assert (mny <= 1.07).all() and (mny >= 0.93).all()
