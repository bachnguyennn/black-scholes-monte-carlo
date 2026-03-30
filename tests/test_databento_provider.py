import pytest
import pandas as pd

from src.core.databento_provider import (
    build_rebalance_dates,
    default_spot_symbol_for_option_root,
    normalize_option_parent_symbol,
    select_near_atm_contracts,
)


def test_normalize_option_parent_symbol_adds_opt_suffix():
    assert normalize_option_parent_symbol("spx") == "SPX.OPT"
    assert normalize_option_parent_symbol("XSP") == "XSP.OPT"


def test_normalize_option_parent_symbol_preserves_opt_suffix():
    assert normalize_option_parent_symbol("SPX.OPT") == "SPX.OPT"


def test_normalize_option_parent_symbol_rejects_unexpected_dot_format():
    with pytest.raises(ValueError):
        normalize_option_parent_symbol("SPX.FUT")


def test_select_near_atm_contracts_keeps_nearest_expiry_and_atm_strikes():
    definitions = pd.DataFrame(
        [
            {"raw_symbol": "C1", "expiration": "2026-03-20", "strike_price": 5750.0, "instrument_class": "C"},
            {"raw_symbol": "C2", "expiration": "2026-03-20", "strike_price": 5800.0, "instrument_class": "C"},
            {"raw_symbol": "C3", "expiration": "2026-03-20", "strike_price": 5900.0, "instrument_class": "C"},
            {"raw_symbol": "P1", "expiration": "2026-03-20", "strike_price": 5750.0, "instrument_class": "P"},
            {"raw_symbol": "P2", "expiration": "2026-03-20", "strike_price": 5800.0, "instrument_class": "P"},
            {"raw_symbol": "P3", "expiration": "2026-03-20", "strike_price": 5900.0, "instrument_class": "P"},
            {"raw_symbol": "FAR", "expiration": "2026-05-15", "strike_price": 5800.0, "instrument_class": "C"},
        ]
    )

    selected = select_near_atm_contracts(
        definitions_df=definitions,
        target_spot=5825.0,
        as_of="2026-03-03",
        min_dte_days=7,
        max_dte_days=45,
        expiries_count=1,
        strikes_per_type=2,
    )

    assert list(selected["raw_symbol"]) == ["C2", "C1", "P2", "P1"]
    assert all(selected["dte_days"] > 7)


def test_default_spot_symbol_for_option_root_maps_index_products():
    assert default_spot_symbol_for_option_root("SPX") == "^SPX"
    assert default_spot_symbol_for_option_root("XSP") == "^SPX"
    assert default_spot_symbol_for_option_root("SPY") == "SPY"


def test_build_rebalance_dates_business_month_start():
    dates = build_rebalance_dates("2026-01-01", "2026-03-31", frequency="BMS")
    assert [ts.strftime("%Y-%m-%d") for ts in dates] == ["2026-01-01", "2026-02-02", "2026-03-02"]
