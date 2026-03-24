"""
Shared market-data access with provider selection and explicit fallbacks.

Polygon is only used for workflows that are practical on a free/basic key in
this repo today:
- spot + recent historical closes for supported tickers
- option expiration discovery via reference contracts when available

Full options-chain fetches remain on yfinance because the app depends on live
chain/expiration workflows that Polygon's free/basic tier does not cover well
enough for this UI.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
import yfinance as yf


POLYGON_BASE_URL = "https://api.polygon.io"
SUPPORTED_MARKET_DATA_PROVIDERS = {"auto", "polygon", "yfinance"}
OPTIONS_CHAIN_FALLBACK_NOTE = (
    "Full options-chain and live options quote workflows still use yfinance. "
    "Polygon free/basic is only used here for supported REST spot/history and "
    "reference-style metadata."
)


class MarketDataError(RuntimeError):
    """Base error for provider failures."""


class UnsupportedWorkflowError(MarketDataError):
    """Raised when a provider is intentionally not used for a workflow."""


@dataclass(frozen=True)
class MarketDataConfig:
    provider_preference: str
    polygon_api_key: str | None

    @property
    def polygon_enabled(self) -> bool:
        return bool(self.polygon_api_key)

    @property
    def polygon_preferred(self) -> bool:
        if self.provider_preference == "polygon":
            return self.polygon_enabled
        return self.provider_preference == "auto" and self.polygon_enabled


class YFinanceMarketDataProvider:
    name = "yfinance"

    def get_spot_and_vol(self, ticker_symbol: str) -> dict[str, Any] | None:
        try:
            tk = yf.Ticker(ticker_symbol)
            history = tk.history(period="1y")
            if history.empty:
                return None

            spot = float(history["Close"].iloc[-1])
            log_returns = np.log(history["Close"] / history["Close"].shift(1)).dropna()
            hist_vol = float(log_returns.std() * np.sqrt(252))
            hist_vol = min(max(hist_vol, 0.05), 2.0)

            return {
                "spot": spot,
                "historical_vol": hist_vol,
                "name": tk.info.get("longName", ticker_symbol),
                "history": history["Close"],
            }
        except Exception:
            return None

    def get_available_expirations(self, ticker_symbol: str) -> list[str]:
        try:
            tk = yf.Ticker(ticker_symbol)
            return list(tk.options)
        except Exception:
            return []

    def get_options_chain(
        self,
        ticker_symbol: str,
        max_expirations: int = 3,
        target_days: int | None = None,
        specific_expirations: list[str] | None = None,
    ) -> pd.DataFrame:
        try:
            tk = yf.Ticker(ticker_symbol)
            expirations = tk.options
            if not expirations:
                return pd.DataFrame()

            if specific_expirations:
                selected = [d for d in specific_expirations if d in expirations]
            elif target_days is not None:
                days_list: list[tuple[int, str]] = []
                now = datetime.now()
                for exp_str in expirations:
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                    diff = abs((exp_dt - now).days - target_days)
                    days_list.append((diff, exp_str))
                selected = [sorted(days_list)[0][1]]
            else:
                selected = list(expirations)[:max_expirations]

            all_rows: list[dict[str, Any]] = []
            for exp_str in selected:
                chain = tk.option_chain(exp_str)
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                exp_datetime = exp_date.replace(hour=16, minute=0)
                days_to_exp = (exp_datetime - datetime.now()).total_seconds() / (24 * 3600)
                if days_to_exp <= 0.01:
                    continue

                T = days_to_exp / 365.0
                all_rows.extend(_normalize_option_rows(chain.calls, exp_str, T, "call"))
                all_rows.extend(_normalize_option_rows(chain.puts, exp_str, T, "put"))

            if not all_rows:
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)
            df = df[(df["volume"] > 0) | (df["openInterest"] > 10)]
            return df.reset_index(drop=True)
        except Exception as exc:
            print(f"Error fetching options chain from yfinance: {exc}")
            return pd.DataFrame()


class PolygonMarketDataProvider:
    name = "polygon"

    def __init__(self, api_key: str, timeout: float = 10.0):
        self.api_key = api_key
        self.timeout = timeout

    def get_spot_and_vol(self, ticker_symbol: str) -> dict[str, Any] | None:
        history = self._fetch_daily_close_history(ticker_symbol, lookback_days=400)
        if history.empty:
            return None

        spot = float(history.iloc[-1])
        log_returns = np.log(history / history.shift(1)).dropna()
        hist_vol = float(log_returns.std() * np.sqrt(252))
        hist_vol = min(max(hist_vol, 0.05), 2.0)

        return {
            "spot": spot,
            "historical_vol": hist_vol,
            "name": self._fetch_ticker_name(ticker_symbol),
            "history": history,
        }

    def get_available_expirations(self, ticker_symbol: str) -> list[str]:
        params = {
            "underlying_ticker": _polygon_reference_symbol(ticker_symbol),
            "expired": "false",
            "limit": 1000,
            "sort": "expiration_date",
            "order": "asc",
        }

        expirations: set[str] = set()
        next_url: str | None = f"{POLYGON_BASE_URL}/v3/reference/options/contracts"

        while next_url:
            payload = self._request_json(next_url, params=params)
            for row in payload.get("results", []):
                expiration = row.get("expiration_date")
                if expiration:
                    expirations.add(str(expiration))

            raw_next_url = payload.get("next_url")
            next_url = self._with_api_key(raw_next_url) if raw_next_url else None
            params = None

        return sorted(expirations)

    def get_options_chain(
        self,
        ticker_symbol: str,
        max_expirations: int = 3,
        target_days: int | None = None,
        specific_expirations: list[str] | None = None,
    ) -> pd.DataFrame:
        raise UnsupportedWorkflowError(OPTIONS_CHAIN_FALLBACK_NOTE)

    def _fetch_daily_close_history(self, ticker_symbol: str, lookback_days: int) -> pd.Series:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=lookback_days)
        polygon_ticker = _polygon_aggs_symbol(ticker_symbol)
        url = f"{POLYGON_BASE_URL}/v2/aggs/ticker/{polygon_ticker}/range/1/day/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"
        payload = self._request_json(
            url,
            params={"adjusted": "true", "sort": "asc", "limit": 5000},
        )

        results = payload.get("results", [])
        if not results:
            return pd.Series(dtype=float)

        timestamps = pd.to_datetime([row["t"] for row in results], unit="ms", utc=True)
        closes = [float(row["c"]) for row in results]
        series = pd.Series(closes, index=timestamps, name="Close")
        return series

    def _fetch_ticker_name(self, ticker_symbol: str) -> str:
        if _is_index_symbol(ticker_symbol):
            return ticker_symbol

        symbol = _polygon_reference_symbol(ticker_symbol)
        url = f"{POLYGON_BASE_URL}/v3/reference/tickers/{symbol}"
        try:
            payload = self._request_json(url)
            return payload.get("results", {}).get("name", ticker_symbol)
        except Exception:
            return ticker_symbol

    def _request_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        params = dict(params or {})
        params["apiKey"] = self.api_key
        response = requests.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        status = str(payload.get("status", "")).upper()
        if status == "ERROR":
            raise MarketDataError(payload.get("error", "Polygon request failed"))
        return payload

    def _with_api_key(self, url: str) -> str:
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query.setdefault("apiKey", self.api_key)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_market_data_config() -> MarketDataConfig:
    preference = os.getenv("MARKET_DATA_PROVIDER", "auto").strip().lower() or "auto"
    if preference not in SUPPORTED_MARKET_DATA_PROVIDERS:
        preference = "auto"

    polygon_api_key = os.getenv("POLYGON_API_KEY", "").strip() or None
    return MarketDataConfig(provider_preference=preference, polygon_api_key=polygon_api_key)


def get_market_data_runtime_summary() -> dict[str, Any]:
    config = get_market_data_config()
    requested = config.provider_preference
    polygon_configured = config.polygon_enabled

    spot_history_provider = "polygon" if config.polygon_preferred else "yfinance"
    expirations_provider = "polygon" if config.polygon_preferred else "yfinance"

    note = (
        "Polygon is preferred for supported REST spot/history workflows."
        if config.polygon_preferred
        else "yfinance is the active market-data source."
    )

    if requested == "polygon" and not polygon_configured:
        note = "MARKET_DATA_PROVIDER=polygon is set, but POLYGON_API_KEY is missing. Falling back to yfinance."

    return {
        "provider_preference": requested,
        "polygon_configured": polygon_configured,
        "spot_history_provider": spot_history_provider,
        "expirations_provider": expirations_provider,
        "options_chain_provider": "yfinance",
        "options_chain_note": OPTIONS_CHAIN_FALLBACK_NOTE,
        "note": note,
    }


def get_spot_and_vol(ticker_symbol: str) -> dict[str, Any] | None:
    requested_provider = _requested_provider_label()
    primary_provider = _build_primary_provider_for_workflow("spot_history")
    primary_data = primary_provider.get_spot_and_vol(ticker_symbol)

    if primary_data:
        return _attach_provider_metadata(
            primary_data,
            provider=primary_provider.name,
            requested_provider=requested_provider,
            fallback_from=None,
        )

    if primary_provider.name == "polygon":
        fallback_provider = YFinanceMarketDataProvider()
        fallback_data = fallback_provider.get_spot_and_vol(ticker_symbol)
        if fallback_data:
            return _attach_provider_metadata(
                fallback_data,
                provider=fallback_provider.name,
                requested_provider=requested_provider,
                fallback_from="polygon",
            )

    return None


def get_risk_free_rate() -> float:
    """
    Fetches the current risk-free rate from the 13-week Treasury Bill (^IRX).
    Falls back to 5% if the fetch fails.
    """
    try:
        irx = yf.Ticker("^IRX")
        hist = irx.history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.05


def get_available_expirations(ticker_symbol: str) -> list[str]:
    primary_provider = _build_primary_provider_for_workflow("expirations")

    try:
        expirations = primary_provider.get_available_expirations(ticker_symbol)
        if expirations:
            return expirations
    except Exception:
        pass

    if primary_provider.name == "polygon":
        return YFinanceMarketDataProvider().get_available_expirations(ticker_symbol)
    return []


def get_options_chain(
    ticker_symbol: str,
    max_expirations: int = 3,
    target_days: int | None = None,
    specific_expirations: list[str] | None = None,
) -> pd.DataFrame:
    # Polygon free/basic is not used here on purpose. The scanner and risk tabs
    # depend on liquid, full-chain style fetches that remain more reliable via
    # yfinance in this repo.
    provider = YFinanceMarketDataProvider()
    return provider.get_options_chain(
        ticker_symbol,
        max_expirations=max_expirations,
        target_days=target_days,
        specific_expirations=specific_expirations,
    )


async def get_options_chain_async(ticker_symbol: str, max_expirations: int = 3) -> pd.DataFrame:
    return await asyncio.to_thread(get_options_chain, ticker_symbol, max_expirations)


def _build_primary_provider_for_workflow(workflow: str) -> YFinanceMarketDataProvider | PolygonMarketDataProvider:
    config = get_market_data_config()
    if workflow == "options_chain":
        return YFinanceMarketDataProvider()
    if config.polygon_preferred:
        return PolygonMarketDataProvider(config.polygon_api_key or "")
    return YFinanceMarketDataProvider()


def _requested_provider_label() -> str:
    config = get_market_data_config()
    if config.provider_preference == "auto":
        return "polygon" if config.polygon_preferred else "yfinance"
    return config.provider_preference


def _attach_provider_metadata(
    payload: dict[str, Any],
    provider: str,
    requested_provider: str,
    fallback_from: str | None,
) -> dict[str, Any]:
    enriched = dict(payload)
    enriched["provider"] = provider
    enriched["requested_provider"] = requested_provider
    enriched["fallback_from"] = fallback_from
    if fallback_from:
        enriched["provider_note"] = f"Fell back from {fallback_from} to {provider} for this request."
    else:
        enriched["provider_note"] = f"Served by {provider}."
    return enriched


def _normalize_option_rows(df: pd.DataFrame, expiration: str, T: float, option_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        bid = float(row.get("bid", 0))
        ask = float(row.get("ask", 0))
        if bid <= 0 or ask <= 0 or ask < bid:
            continue

        rows.append(
            {
                "type": option_type,
                "strike": float(row["strike"]),
                "expiration": expiration,
                "T": T,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "market_iv": float(row.get("impliedVolatility", 0)),
                "volume": int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                "openInterest": int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
            }
        )
    return rows


def _is_index_symbol(ticker_symbol: str) -> bool:
    clean_symbol = ticker_symbol.strip().upper()
    return clean_symbol.startswith("^") or clean_symbol.startswith("I:")


def _polygon_reference_symbol(ticker_symbol: str) -> str:
    clean_symbol = ticker_symbol.strip().upper()
    if clean_symbol.startswith("^"):
        return clean_symbol[1:]
    if clean_symbol.startswith("I:"):
        return clean_symbol[2:]
    return clean_symbol


def _polygon_aggs_symbol(ticker_symbol: str) -> str:
    symbol = _polygon_reference_symbol(ticker_symbol)
    if _is_index_symbol(ticker_symbol):
        return f"I:{symbol}"
    return symbol
