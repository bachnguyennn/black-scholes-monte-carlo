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
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd
import requests
import yfinance as yf


logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class MarketDataSnapshot:
    ticker_symbol: str
    spot: float
    historical_vol: float
    name: str
    history: pd.Series
    provider: str
    requested_provider: str
    fallback_from: str | None
    provider_note: str
    as_of: pd.Timestamp
    history_start: pd.Timestamp
    history_end: pd.Timestamp
    history_points: int
    is_stale: bool
    validation_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker_symbol": self.ticker_symbol,
            "spot": self.spot,
            "historical_vol": self.historical_vol,
            "name": self.name,
            "history": self.history,
            "provider": self.provider,
            "requested_provider": self.requested_provider,
            "fallback_from": self.fallback_from,
            "provider_note": self.provider_note,
            "as_of": self.as_of,
            "history_start": self.history_start,
            "history_end": self.history_end,
            "history_points": self.history_points,
            "is_stale": self.is_stale,
            "validation_warnings": list(self.validation_warnings),
        }


class YFinanceMarketDataProvider:
    name = "yfinance"

    def get_spot_and_vol(self, ticker_symbol: str) -> dict[str, Any] | None:
        try:
            tk = yf.Ticker(ticker_symbol)
            history = tk.history(period="1y")
            if history.empty:
                return None

            close_history = history["Close"]
            spot = float(close_history.iloc[-1])
            hist_vol = _compute_history_volatility(close_history)

            return {
                "spot": spot,
                "historical_vol": hist_vol,
                "name": tk.info.get("longName", ticker_symbol),
                "history": close_history,
            }
        except Exception as exc:
            logger.warning("yfinance spot/history fetch failed for %s: %s", ticker_symbol, exc)
            return None

    def get_available_expirations(self, ticker_symbol: str) -> list[str]:
        try:
            tk = yf.Ticker(ticker_symbol)
            return list(tk.options)
        except Exception as exc:
            logger.warning("yfinance expiration fetch failed for %s: %s", ticker_symbol, exc)
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
            logger.warning("yfinance options-chain fetch failed for %s: %s", ticker_symbol, exc)
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

        return {
            "spot": float(history.iloc[-1]),
            "historical_vol": _compute_history_volatility(history),
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
        return pd.Series(closes, index=timestamps, name="Close")

    def _fetch_ticker_name(self, ticker_symbol: str) -> str:
        if _is_index_symbol(ticker_symbol):
            return ticker_symbol

        symbol = _polygon_reference_symbol(ticker_symbol)
        url = f"{POLYGON_BASE_URL}/v3/reference/tickers/{symbol}"
        try:
            payload = self._request_json(url)
            return payload.get("results", {}).get("name", ticker_symbol)
        except Exception as exc:
            logger.info("polygon ticker-name lookup failed for %s: %s", ticker_symbol, exc)
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
        "max_staleness_days": get_market_data_max_staleness_days(),
        "min_history_points": get_market_data_min_history_points(),
        "note": note,
    }


def get_market_data_max_staleness_days() -> int:
    return _read_positive_int_env("MARKET_DATA_MAX_STALENESS_DAYS", default=5)


def get_market_data_min_history_points() -> int:
    return _read_positive_int_env("MARKET_DATA_MIN_HISTORY_POINTS", default=30)


def get_spot_snapshot(ticker_symbol: str) -> MarketDataSnapshot | None:
    requested_provider = _requested_provider_label()
    primary_provider = _build_primary_provider_for_workflow("spot_history")
    primary_raw = _safe_fetch_spot_payload(primary_provider, ticker_symbol)
    primary_snapshot = _normalize_market_snapshot(
        raw_payload=primary_raw,
        ticker_symbol=ticker_symbol,
        provider=primary_provider.name,
        requested_provider=requested_provider,
        fallback_from=None,
    )

    if primary_snapshot and _snapshot_supports_research_workflow(primary_snapshot):
        _log_snapshot_resolution(primary_snapshot, degraded=False)
        return primary_snapshot

    if primary_provider.name == "polygon":
        fallback_reason = _snapshot_issue_summary(primary_snapshot) if primary_snapshot else "primary provider returned no usable snapshot"
        logger.warning(
            "market_data_fallback ticker=%s from=%s to=yfinance reason=%s",
            ticker_symbol,
            primary_provider.name,
            fallback_reason,
        )
        fallback_raw = _safe_fetch_spot_payload(YFinanceMarketDataProvider(), ticker_symbol)
        fallback_snapshot = _normalize_market_snapshot(
            raw_payload=fallback_raw,
            ticker_symbol=ticker_symbol,
            provider="yfinance",
            requested_provider=requested_provider,
            fallback_from="polygon",
        )
        if fallback_snapshot:
            _log_snapshot_resolution(fallback_snapshot, degraded=not _snapshot_supports_research_workflow(fallback_snapshot))
            return fallback_snapshot

    if primary_snapshot:
        _log_snapshot_resolution(primary_snapshot, degraded=True)
        return primary_snapshot

    logger.error("market_data_unavailable ticker=%s workflow=spot_history requested=%s", ticker_symbol, requested_provider)
    return None


def get_spot_and_vol(ticker_symbol: str) -> dict[str, Any] | None:
    snapshot = get_spot_snapshot(ticker_symbol)
    return snapshot.to_dict() if snapshot else None


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
        expirations = _normalize_expiration_list(primary_provider.get_available_expirations(ticker_symbol))
        if expirations:
            logger.info(
                "market_data_expirations_resolved ticker=%s provider=%s count=%s",
                ticker_symbol,
                primary_provider.name,
                len(expirations),
            )
            return expirations
    except Exception as exc:
        logger.warning(
            "market_data_expirations_failed ticker=%s provider=%s error=%s",
            ticker_symbol,
            primary_provider.name,
            exc,
        )

    if primary_provider.name == "polygon":
        fallback_expirations = _normalize_expiration_list(YFinanceMarketDataProvider().get_available_expirations(ticker_symbol))
        if fallback_expirations:
            logger.warning(
                "market_data_expirations_fallback ticker=%s from=polygon to=yfinance count=%s",
                ticker_symbol,
                len(fallback_expirations),
            )
        return fallback_expirations
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


def _safe_fetch_spot_payload(provider: YFinanceMarketDataProvider | PolygonMarketDataProvider, ticker_symbol: str) -> dict[str, Any] | None:
    try:
        return provider.get_spot_and_vol(ticker_symbol)
    except Exception as exc:
        logger.warning(
            "market_data_fetch_failed ticker=%s provider=%s workflow=spot_history error=%s",
            ticker_symbol,
            provider.name,
            exc,
        )
        return None


def _normalize_market_snapshot(
    raw_payload: dict[str, Any] | None,
    ticker_symbol: str,
    provider: str,
    requested_provider: str,
    fallback_from: str | None,
) -> MarketDataSnapshot | None:
    if not raw_payload:
        return None

    history = _coerce_history_series(raw_payload.get("history"))
    if history is None or history.empty:
        logger.warning("market_data_invalid_history ticker=%s provider=%s", ticker_symbol, provider)
        return None

    history_start = history.index.min()
    history_end = history.index.max()
    history_points = len(history)
    as_of = history_end
    warnings: list[str] = []

    if history_points < get_market_data_min_history_points():
        warnings.append(f"Limited history returned ({history_points} points).")

    age = pd.Timestamp.now(tz="UTC") - as_of
    is_stale = age > pd.Timedelta(days=get_market_data_max_staleness_days())
    if is_stale:
        warnings.append(f"Market data is stale ({int(age.total_seconds() // 86400)}d old).")

    spot = _coerce_positive_float(raw_payload.get("spot"))
    last_close = float(history.iloc[-1])
    if spot is None:
        spot = last_close
        warnings.append("Spot price was normalized from the close history.")

    hist_vol = _coerce_positive_float(raw_payload.get("historical_vol"))
    if hist_vol is None:
        hist_vol = _compute_history_volatility(history)
        warnings.append("Historical volatility was recomputed from the close history.")
    if not np.isfinite(hist_vol) or hist_vol <= 0:
        hist_vol = 0.2
        warnings.append("Historical volatility fallback of 20% was used due to insufficient history.")

    hist_vol = float(min(max(hist_vol, 0.05), 2.0))
    provider_note = _build_provider_note(provider, fallback_from, warnings)

    return MarketDataSnapshot(
        ticker_symbol=ticker_symbol,
        spot=float(spot),
        historical_vol=hist_vol,
        name=str(raw_payload.get("name") or ticker_symbol),
        history=history,
        provider=provider,
        requested_provider=requested_provider,
        fallback_from=fallback_from,
        provider_note=provider_note,
        as_of=as_of,
        history_start=history_start,
        history_end=history_end,
        history_points=history_points,
        is_stale=is_stale,
        validation_warnings=tuple(warnings),
    )


def _snapshot_supports_research_workflow(snapshot: MarketDataSnapshot) -> bool:
    return not snapshot.is_stale and snapshot.history_points >= get_market_data_min_history_points()


def _snapshot_issue_summary(snapshot: MarketDataSnapshot | None) -> str:
    if snapshot is None:
        return "snapshot normalization failed"
    if snapshot.validation_warnings:
        return "; ".join(snapshot.validation_warnings)
    return "snapshot failed validation"


def _log_snapshot_resolution(snapshot: MarketDataSnapshot, degraded: bool) -> None:
    logger.log(
        logging.WARNING if degraded else logging.INFO,
        "market_data_snapshot_resolved ticker=%s provider=%s requested=%s fallback_from=%s stale=%s history_points=%s warnings=%s",
        snapshot.ticker_symbol,
        snapshot.provider,
        snapshot.requested_provider,
        snapshot.fallback_from or "-",
        snapshot.is_stale,
        snapshot.history_points,
        len(snapshot.validation_warnings),
    )


def _coerce_history_series(raw_history: Any) -> pd.Series | None:
    if isinstance(raw_history, pd.Series):
        history = raw_history.copy()
    elif isinstance(raw_history, pd.DataFrame) and "Close" in raw_history.columns:
        history = raw_history["Close"].copy()
    else:
        return None

    history = pd.to_numeric(history, errors="coerce").dropna()
    if history.empty:
        return None

    index = pd.to_datetime(history.index, errors="coerce", utc=True)
    valid_mask = ~index.isna()
    history = history[valid_mask]
    index = index[valid_mask]
    if history.empty:
        return None

    history.index = index
    history = history.sort_index()
    history = history[~history.index.duplicated(keep="last")]
    history.name = "Close"
    return history


def _compute_history_volatility(history: pd.Series) -> float:
    log_returns = np.log(history / history.shift(1)).dropna()
    if log_returns.empty:
        return 0.2
    return float(log_returns.std() * np.sqrt(252))


def _build_provider_note(provider: str, fallback_from: str | None, warnings: list[str]) -> str:
    if fallback_from:
        prefix = f"Fell back from {fallback_from} to {provider} for this request."
    else:
        prefix = f"Served by {provider}."

    if warnings:
        return f"{prefix} Validation: {' '.join(warnings)}"
    return prefix


def _normalize_expiration_list(expirations: list[str]) -> list[str]:
    normalized: set[str] = set()
    for exp_str in expirations:
        try:
            normalized.add(datetime.strptime(exp_str, "%Y-%m-%d").strftime("%Y-%m-%d"))
        except Exception:
            continue
    return sorted(normalized)


def _coerce_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


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


def _read_positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


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
