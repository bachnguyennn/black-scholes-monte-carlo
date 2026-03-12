import pandas as pd

from src.core.scanner_engine import scan_for_arbitrage


def _sample_options_df():
    return pd.DataFrame(
        [
            {
                "type": "call",
                "strike": 100.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 4.8,
                "ask": 5.2,
                "mid": 5.0,
                "volume": 100,
                "openInterest": 200,
                "market_iv": 0.20,
            }
        ]
    )


def test_scanner_gbm_path_returns_rows():
    df = _sample_options_df()
    out = scan_for_arbitrage(df, S0=100.0, r=0.03, q=0.01, model="gbm")
    assert not out.empty
    assert out.iloc[0]["type"] == "CALL"


def test_scanner_heston_path_returns_rows():
    df = _sample_options_df()
    out = scan_for_arbitrage(df, S0=100.0, r=0.03, q=0.01, model="heston")
    assert not out.empty
    assert out.iloc[0]["type"] == "CALL"


def test_scanner_returns_diagnostics_for_filtered_contracts():
    df = pd.DataFrame(
        [
            {
                "type": "call",
                "strike": 100.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 4.8,
                "ask": 5.2,
                "mid": 5.0,
                "volume": 100,
                "openInterest": 200,
                "market_iv": 0.20,
            },
            {
                "type": "call",
                "strike": 140.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 0.4,
                "ask": 0.6,
                "mid": 0.5,
                "volume": 10,
                "openInterest": 20,
                "market_iv": 0.22,
            },
        ]
    )

    out, diagnostics = scan_for_arbitrage(
        df,
        S0=100.0,
        r=0.03,
        q=0.01,
        model="gbm",
        return_diagnostics=True,
    )

    assert not out.empty
    assert diagnostics["total_contracts"] == 2
    assert diagnostics["contracts_priced"] == 1
    assert diagnostics["contracts_filtered"] == 1
    assert diagnostics["reason_counts"]["moneyness"] == 1
    assert sum(diagnostics["signal_counts"].values()) == 1

