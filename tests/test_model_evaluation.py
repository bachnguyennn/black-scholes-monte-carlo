import math

import pandas as pd

from src.core.model_evaluation import build_live_surface_evaluation


def test_live_surface_evaluation_reports_core_metrics():
    priced_df = pd.DataFrame(
        [
            {
                "type": "CALL",
                "strike": 100.0,
                "T_days": 30,
                "bid": 4.5,
                "ask": 5.5,
                "mid": 5.0,
                "mc_price": 5.1,
                "market_iv": 20.0,
            },
            {
                "type": "PUT",
                "strike": 100.0,
                "T_days": 30,
                "bid": 4.0,
                "ask": 5.0,
                "mid": 4.5,
                "mc_price": 4.4,
                "market_iv": 22.0,
            },
        ]
    )

    metrics = build_live_surface_evaluation(priced_df, S0=100.0, r=0.05)

    assert metrics["success"] is True
    assert metrics["contracts_evaluated"] == 2
    assert math.isfinite(metrics["price_mae"])
    assert math.isfinite(metrics["price_rmse"])
    assert math.isfinite(metrics["mean_abs_error_in_spreads"])
    assert 0.0 <= metrics["within_nbbo_pct"] <= 100.0
    assert metrics["iv_contracts_evaluated"] >= 1


def test_live_surface_evaluation_rejects_missing_columns():
    metrics = build_live_surface_evaluation(pd.DataFrame([{"strike": 100.0}]), S0=100.0, r=0.05)

    assert metrics["success"] is False
    assert "bid" in metrics["missing_columns"]
