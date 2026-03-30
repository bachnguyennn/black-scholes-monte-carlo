import pandas as pd

from src.core import data_fetcher


def _history_series(days_ago_start: int, periods: int = 40) -> pd.Series:
    end = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_ago_start)
    index = pd.date_range(end=end, periods=periods, freq="D", tz="UTC")
    values = [100.0 + idx for idx in range(periods)]
    return pd.Series(values, index=index, name="Close")


def test_market_data_summary_falls_back_when_polygon_key_missing(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)

    summary = data_fetcher.get_market_data_runtime_summary()

    assert summary["provider_preference"] == "polygon"
    assert summary["polygon_configured"] is False
    assert summary["spot_history_provider"] == "yfinance"
    assert summary["options_chain_provider"] == "yfinance"


def test_get_spot_and_vol_prefers_polygon_when_configured(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    expected = {
        "spot": 501.25,
        "historical_vol": 0.18,
        "name": "SPY",
        "history": pd.Series([500.0, 501.25]),
    }

    monkeypatch.setattr(
        data_fetcher.PolygonMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: dict(expected),
    )
    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: (_ for _ in ()).throw(AssertionError("yfinance should not be used")),
    )

    result = data_fetcher.get_spot_and_vol("SPY")

    assert result is not None
    assert result["provider"] == "polygon"
    assert result["requested_provider"] == "polygon"
    assert result["fallback_from"] is None
    assert result["history_points"] == 2
    assert "Limited history returned" in " ".join(result["validation_warnings"])
    assert result["spot"] == expected["spot"]


def test_get_spot_and_vol_falls_back_to_yfinance_when_polygon_returns_none(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    monkeypatch.setattr(
        data_fetcher.PolygonMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: None,
    )
    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: {
            "spot": 499.9,
            "historical_vol": 0.21,
            "name": ticker,
            "history": pd.Series([498.0, 499.9]),
        },
    )

    result = data_fetcher.get_spot_and_vol("SPY")

    assert result is not None
    assert result["provider"] == "yfinance"
    assert result["requested_provider"] == "polygon"
    assert result["fallback_from"] == "polygon"
    assert "Fell back from polygon to yfinance" in result["provider_note"]


def test_get_available_expirations_prefers_polygon_reference_data(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    monkeypatch.setattr(
        data_fetcher.PolygonMarketDataProvider,
        "get_available_expirations",
        lambda self, ticker: ["2026-04-17", "2026-05-15"],
    )
    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_available_expirations",
        lambda self, ticker: (_ for _ in ()).throw(AssertionError("yfinance should not be used")),
    )

    expirations = data_fetcher.get_available_expirations("SPY")

    assert expirations == ["2026-04-17", "2026-05-15"]


def test_get_options_chain_stays_on_yfinance_when_polygon_is_preferred(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    expected = pd.DataFrame(
        [
            {
                "type": "call",
                "strike": 500.0,
                "expiration": "2026-04-17",
                "T": 0.1,
                "bid": 10.0,
                "ask": 10.5,
                "mid": 10.25,
                "market_iv": 0.2,
                "volume": 10,
                "openInterest": 100,
            }
        ]
    )

    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_options_chain",
        lambda self, ticker_symbol, max_expirations=3, target_days=None, specific_expirations=None: expected,
    )
    monkeypatch.setattr(
        data_fetcher.PolygonMarketDataProvider,
        "get_options_chain",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(AssertionError("Polygon should not be used")),
    )

    result = data_fetcher.get_options_chain("SPY", specific_expirations=["2026-04-17"])

    pd.testing.assert_frame_equal(result, expected)


def test_get_spot_snapshot_falls_back_when_polygon_data_is_stale(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    monkeypatch.setenv("MARKET_DATA_MAX_STALENESS_DAYS", "2")

    monkeypatch.setattr(
        data_fetcher.PolygonMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: {
            "spot": 500.0,
            "historical_vol": 0.2,
            "name": ticker,
            "history": _history_series(days_ago_start=10),
        },
    )
    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: {
            "spot": 505.0,
            "historical_vol": 0.19,
            "name": ticker,
            "history": _history_series(days_ago_start=0),
        },
    )

    result = data_fetcher.get_spot_and_vol("SPY")

    assert result is not None
    assert result["provider"] == "yfinance"
    assert result["fallback_from"] == "polygon"
    assert result["is_stale"] is False


def test_get_spot_snapshot_returns_normalized_metadata(monkeypatch):
    monkeypatch.setenv("MARKET_DATA_PROVIDER", "yfinance")
    monkeypatch.setenv("MARKET_DATA_MIN_HISTORY_POINTS", "30")

    monkeypatch.setattr(
        data_fetcher.YFinanceMarketDataProvider,
        "get_spot_and_vol",
        lambda self, ticker: {
            "name": ticker,
            "history": _history_series(days_ago_start=0),
        },
    )

    result = data_fetcher.get_spot_and_vol("SPY")

    assert result is not None
    assert result["provider"] == "yfinance"
    assert result["spot"] > 0
    assert result["historical_vol"] > 0
    assert isinstance(result["as_of"], pd.Timestamp)
    assert isinstance(result["history_start"], pd.Timestamp)
    assert isinstance(result["history_end"], pd.Timestamp)
    assert result["history_points"] == 40
    assert result["is_stale"] is False
    assert "Spot price was normalized from the close history." in result["provider_note"]
