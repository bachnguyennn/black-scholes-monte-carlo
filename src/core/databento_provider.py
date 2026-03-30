"""
databento_provider.py

Minimal Databento historical-options helpers focused on OPRA parent-symbol
requests. The design is intentionally cost-first: estimate before download.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import databento as db
import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_DATABENTO_DATASET = "OPRA.PILLAR"
DEFAULT_DATABENTO_STYPE_IN = "parent"
DEFAULT_DATABENTO_SCHEMAS = ("definition", "cbbo-1m")
DEFAULT_DATABENTO_OUTPUT_ROOT = Path("data/databento")


def normalize_option_parent_symbol(symbol: str) -> str:
    """Normalizes an option parent symbol into Databento's `[ROOT].OPT` form."""
    cleaned = (symbol or "").strip().upper()
    if not cleaned:
        raise ValueError("A non-empty symbol is required.")
    if cleaned.endswith(".OPT"):
        return cleaned
    if "." in cleaned:
        raise ValueError(f"Unexpected Databento parent symbol format: {symbol}")
    return f"{cleaned}.OPT"


def default_spot_symbol_for_option_root(symbol: str) -> str:
    """Maps an option root to a practical spot-history ticker."""
    root = (symbol or "").strip().upper().removesuffix(".OPT")
    if root in {"SPX", "XSP"}:
        return "^SPX"
    return root


def get_historical_client(api_key: str | None = None) -> db.Historical:
    """Builds a Databento historical client from an explicit key or env."""
    resolved_key = api_key or os.getenv("DATABENTO_API_KEY", "").strip()
    if not resolved_key:
        raise ValueError("DATABENTO_API_KEY is not set.")
    return db.Historical(resolved_key)


def get_dataset_range(
    api_key: str | None = None,
    dataset: str = DEFAULT_DATABENTO_DATASET,
) -> dict[str, Any]:
    client = get_historical_client(api_key=api_key)
    return client.metadata.get_dataset_range(dataset)


def build_rebalance_dates(
    start: str,
    end: str,
    frequency: str = "BMS",
) -> list[pd.Timestamp]:
    """
    Builds a rebalance-date schedule in UTC-normalized naive dates.

    Common frequencies:
    - `BMS`: business month start
    - `W-FRI`: weekly Friday
    - `B`: every business day
    """
    schedule = pd.date_range(start=start, end=end, freq=frequency)
    return [pd.Timestamp(ts).normalize() for ts in schedule]


def fetch_spot_history(
    ticker: str,
    start: str,
    end: str,
) -> pd.Series:
    """Fetches daily close history for contract selection."""
    history = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if history.empty:
        return pd.Series(dtype=float)

    if isinstance(history.columns, pd.MultiIndex):
        close_col = ("Adj Close", ticker) if ("Adj Close", ticker) in history.columns else ("Close", ticker)
        series = history.loc[:, close_col].copy()
    else:
        close_col = "Adj Close" if "Adj Close" in history.columns else "Close"
        series = history[close_col].copy()

    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]

    series.index = pd.to_datetime(series.index).tz_localize(None)
    return pd.to_numeric(series, errors="coerce").dropna()


def estimate_request(
    symbol: str,
    start: str,
    end: str,
    schema: str = "cbbo-1m",
    dataset: str = DEFAULT_DATABENTO_DATASET,
    stype_in: str = DEFAULT_DATABENTO_STYPE_IN,
    api_key: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Returns Databento's billable size, record count, and credit cost estimate."""
    client = get_historical_client(api_key=api_key)
    parent_symbol = normalize_option_parent_symbol(symbol) if stype_in == "parent" else symbol
    request = {
        "dataset": dataset,
        "start": start,
        "end": end,
        "symbols": parent_symbol,
        "schema": schema,
        "stype_in": stype_in,
    }
    if limit is not None:
        request["limit"] = int(limit)

    return {
        "dataset": dataset,
        "schema": schema,
        "symbol": parent_symbol,
        "stype_in": stype_in,
        "start": start,
        "end": end,
        "record_count": int(client.metadata.get_record_count(**request)),
        "billable_bytes": int(client.metadata.get_billable_size(**request)),
        "estimated_cost_usd": float(client.metadata.get_cost(**request)),
    }


def estimate_request_bundle(
    symbol: str,
    start: str,
    end: str,
    schemas: tuple[str, ...] = DEFAULT_DATABENTO_SCHEMAS,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    stype_in: str = DEFAULT_DATABENTO_STYPE_IN,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Estimates multiple schemas for the same parent symbol and time range."""
    return [
        estimate_request(
            symbol=symbol,
            start=start,
            end=end,
            schema=schema,
            dataset=dataset,
            stype_in=stype_in,
            api_key=api_key,
        )
        for schema in schemas
    ]


def fetch_range_df(
    symbol: str,
    start: str,
    end: str,
    schema: str = "cbbo-1m",
    dataset: str = DEFAULT_DATABENTO_DATASET,
    stype_in: str = DEFAULT_DATABENTO_STYPE_IN,
    api_key: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetches a historical range and converts it to a pandas DataFrame."""
    client = get_historical_client(api_key=api_key)
    parent_symbol = normalize_option_parent_symbol(symbol) if stype_in == "parent" else symbol
    result = client.timeseries.get_range(
        dataset=dataset,
        start=start,
        end=end,
        symbols=parent_symbol,
        schema=schema,
        stype_in=stype_in,
        limit=limit,
    )
    return result.to_df()


def fetch_and_save_range(
    symbol: str,
    start: str,
    end: str,
    output_path: str,
    schema: str = "cbbo-1m",
    dataset: str = DEFAULT_DATABENTO_DATASET,
    stype_in: str = DEFAULT_DATABENTO_STYPE_IN,
    api_key: str | None = None,
    limit: int | None = None,
) -> Path:
    """Fetches a range and saves it as parquet or CSV based on suffix."""
    df = fetch_range_df(
        symbol=symbol,
        start=start,
        end=end,
        schema=schema,
        dataset=dataset,
        stype_in=stype_in,
        api_key=api_key,
        limit=limit,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=True)
    else:
        if out.suffix.lower() != ".parquet":
            out = out.with_suffix(".parquet")
        df.to_parquet(out, index=True)
    return out


def prepare_definition_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes Databento definition rows for option contract selection."""
    if df.empty:
        return df.copy()

    prepared = df.copy()
    if "expiration" in prepared.columns:
        prepared["expiration"] = pd.to_datetime(prepared["expiration"], utc=True, errors="coerce")
    if "strike_price" in prepared.columns:
        prepared["strike_price"] = pd.to_numeric(prepared["strike_price"], errors="coerce")
    if "raw_symbol" in prepared.columns:
        prepared["raw_symbol"] = prepared["raw_symbol"].astype(str)

    if "instrument_class" in prepared.columns:
        prepared["option_type"] = prepared["instrument_class"].map({"C": "call", "P": "put"}).fillna("unknown")
    else:
        prepared["option_type"] = "unknown"

    return prepared


def select_near_atm_contracts(
    definitions_df: pd.DataFrame,
    target_spot: float,
    as_of: str,
    min_dte_days: int = 7,
    max_dte_days: int = 45,
    expiries_count: int = 1,
    strikes_per_type: int = 2,
) -> pd.DataFrame:
    """
    Selects a narrow near-ATM universe from Databento definition rows.

    The output is intended for budget-safe follow-on requests against quote
    schemas such as `cbbo-1m`.
    """
    if definitions_df.empty:
        return pd.DataFrame()
    if target_spot <= 0:
        raise ValueError("target_spot must be positive.")

    prepared = prepare_definition_df(definitions_df)
    as_of_ts = pd.Timestamp(as_of, tz="UTC")

    keep_cols = [
        col
        for col in [
            "raw_symbol",
            "symbol",
            "expiration",
            "strike_price",
            "option_type",
            "underlying",
        ]
        if col in prepared.columns
    ]
    filtered = prepared[keep_cols].copy()
    filtered = filtered.dropna(subset=["expiration", "strike_price", "raw_symbol"])
    filtered["dte_days"] = (filtered["expiration"] - as_of_ts).dt.total_seconds() / 86400.0
    filtered = filtered[(filtered["dte_days"] >= min_dte_days) & (filtered["dte_days"] <= max_dte_days)]
    if filtered.empty:
        return pd.DataFrame()

    filtered["moneyness_gap"] = (filtered["strike_price"] - float(target_spot)).abs()

    selected_frames: list[pd.DataFrame] = []
    expiries = (
        filtered[["expiration", "dte_days"]]
        .drop_duplicates()
        .sort_values(["dte_days", "expiration"])
        .head(expiries_count)
    )
    for expiry in expiries["expiration"]:
        expiry_df = filtered[filtered["expiration"] == expiry].copy()
        for option_type in ("call", "put"):
            side_df = expiry_df[expiry_df["option_type"] == option_type].copy()
            if side_df.empty:
                continue
            side_df = side_df.sort_values(["moneyness_gap", "strike_price"])
            selected_frames.append(side_df.head(strikes_per_type))

    if not selected_frames:
        return pd.DataFrame()

    selected = pd.concat(selected_frames, ignore_index=True)
    selected = selected.drop_duplicates(subset=["raw_symbol"]).sort_values(
        ["expiration", "option_type", "moneyness_gap", "strike_price"]
    )
    selected["atm_distance_pct"] = np.where(
        target_spot > 0,
        (selected["strike_price"] - float(target_spot)).abs() / float(target_spot) * 100.0,
        np.nan,
    )
    return selected.reset_index(drop=True)


def fetch_definitions_for_parent(
    symbol: str,
    start: str,
    end: str,
    api_key: str | None = None,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    limit: int | None = None,
) -> pd.DataFrame:
    """Convenience wrapper for fetching option definitions by parent symbol."""
    return fetch_range_df(
        symbol=symbol,
        start=start,
        end=end,
        schema="definition",
        dataset=dataset,
        stype_in="parent",
        api_key=api_key,
        limit=limit,
    )


def select_contracts_from_parent(
    symbol: str,
    start: str,
    end: str,
    target_spot: float,
    as_of: str,
    api_key: str | None = None,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    definition_limit: int | None = None,
    min_dte_days: int = 7,
    max_dte_days: int = 45,
    expiries_count: int = 1,
    strikes_per_type: int = 2,
) -> pd.DataFrame:
    """Fetches definitions for a parent and returns a narrowed near-ATM universe."""
    definitions_df = fetch_definitions_for_parent(
        symbol=symbol,
        start=start,
        end=end,
        api_key=api_key,
        dataset=dataset,
        limit=definition_limit,
    )
    return select_near_atm_contracts(
        definitions_df=definitions_df,
        target_spot=target_spot,
        as_of=as_of,
        min_dte_days=min_dte_days,
        max_dte_days=max_dte_days,
        expiries_count=expiries_count,
        strikes_per_type=strikes_per_type,
    )


def build_backtest_download_plan(
    symbol: str,
    rebalance_start: str,
    rebalance_end: str,
    frequency: str = "BMS",
    spot_ticker: str | None = None,
    min_dte_days: int = 7,
    max_dte_days: int = 45,
    expiries_count: int = 1,
    strikes_per_type: int = 2,
    quote_schema: str = "cbbo-1m",
    dataset: str = DEFAULT_DATABENTO_DATASET,
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Builds a costed Databento quote-download plan across rebalance dates.

    For each rebalance date:
    - fetch underlying close history
    - select near-ATM contracts from OPRA definitions
    - estimate quote cost from rebalance date through contract expiry
    """
    parent_symbol = normalize_option_parent_symbol(symbol)
    underlying_ticker = spot_ticker or default_spot_symbol_for_option_root(symbol)
    rebalances = build_rebalance_dates(rebalance_start, rebalance_end, frequency=frequency)
    if not rebalances:
        return pd.DataFrame()

    spot_history = fetch_spot_history(
        ticker=underlying_ticker,
        start=(min(rebalances) - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        end=(max(rebalances) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
    )
    if spot_history.empty:
        raise ValueError(f"No spot history returned for {underlying_ticker}.")

    plan_rows: list[dict[str, Any]] = []
    for rebalance_ts in rebalances:
        available_spots = spot_history.loc[:rebalance_ts]
        if available_spots.empty:
            continue

        spot = float(available_spots.iloc[-1])
        def_start = rebalance_ts.strftime("%Y-%m-%d")
        def_end = (rebalance_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        selected = select_contracts_from_parent(
            symbol=parent_symbol,
            start=def_start,
            end=def_end,
            target_spot=spot,
            as_of=rebalance_ts.strftime("%Y-%m-%d"),
            api_key=api_key,
            dataset=dataset,
            min_dte_days=min_dte_days,
            max_dte_days=max_dte_days,
            expiries_count=expiries_count,
            strikes_per_type=strikes_per_type,
        )
        if selected.empty:
            continue

        for row in selected.itertuples(index=False):
            quote_start = rebalance_ts.strftime("%Y-%m-%d")
            quote_end = row.expiration.strftime("%Y-%m-%d")
            estimate = estimate_request(
                symbol=row.raw_symbol,
                start=quote_start,
                end=quote_end,
                schema=quote_schema,
                dataset=dataset,
                stype_in="raw_symbol",
                api_key=api_key,
            )
            plan_rows.append(
                {
                    "rebalance_date": rebalance_ts.strftime("%Y-%m-%d"),
                    "spot_ticker": underlying_ticker,
                    "spot_price": spot,
                    "parent_symbol": parent_symbol,
                    "raw_symbol": row.raw_symbol,
                    "option_type": row.option_type,
                    "expiration": row.expiration.strftime("%Y-%m-%d"),
                    "strike_price": float(row.strike_price),
                    "dte_days": float(row.dte_days),
                    "atm_distance_pct": float(row.atm_distance_pct),
                    "quote_schema": quote_schema,
                    "quote_start": quote_start,
                    "quote_end": quote_end,
                    "record_count": estimate["record_count"],
                    "billable_bytes": estimate["billable_bytes"],
                    "estimated_cost_usd": estimate["estimated_cost_usd"],
                }
            )

    return pd.DataFrame(plan_rows)


def default_output_dir(symbol: str, schema: str = "cbbo-1m") -> Path:
    root = normalize_option_parent_symbol(symbol).replace(".OPT", "").lower()
    return DEFAULT_DATABENTO_OUTPUT_ROOT / "opra" / root / schema


def execute_download_plan(
    plan_df: pd.DataFrame,
    output_dir: str | Path,
    dataset: str = DEFAULT_DATABENTO_DATASET,
    api_key: str | None = None,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """Downloads all quote windows in a plan and returns an execution manifest."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for row in plan_df.itertuples(index=False):
        safe_symbol = str(row.raw_symbol).replace(" ", "_")
        target = output_root / str(row.rebalance_date) / f"{safe_symbol}.parquet"
        status = "downloaded"
        if skip_existing and target.exists():
            status = "skipped_existing"
        else:
            fetch_and_save_range(
                symbol=row.raw_symbol,
                start=row.quote_start,
                end=row.quote_end,
                output_path=str(target),
                schema=row.quote_schema,
                dataset=dataset,
                stype_in="raw_symbol",
                api_key=api_key,
            )

        manifest_rows.append(
            {
                **row._asdict(),
                "output_path": str(target),
                "status": status,
            }
        )

    return pd.DataFrame(manifest_rows)
