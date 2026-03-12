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
