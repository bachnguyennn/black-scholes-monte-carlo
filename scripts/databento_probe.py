#!/usr/bin/env python3
"""
Budget-safe Databento probe for OPRA historical options.

Examples:
  python3 scripts/databento_probe.py --symbol SPX --start 2026-03-03 --end 2026-03-04 --estimate-only
  python3 scripts/databento_probe.py --symbol SPX --start 2026-03-03 --end 2026-03-03T15:00:00Z --schema definition --limit 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.databento_provider import (
    DEFAULT_DATABENTO_DATASET,
    DEFAULT_DATABENTO_STYPE_IN,
    estimate_request,
    estimate_request_bundle,
    fetch_and_save_range,
    fetch_range_df,
    normalize_option_parent_symbol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Databento OPRA historical options with a cost-first workflow.")
    parser.add_argument("--symbol", required=True, help="Underlying root, e.g. SPX, XSP, or SPY.")
    parser.add_argument("--start", required=True, help="Inclusive start timestamp/date.")
    parser.add_argument("--end", required=True, help="Exclusive end timestamp/date.")
    parser.add_argument("--schema", default="cbbo-1m", help="Databento schema, e.g. definition or cbbo-1m.")
    parser.add_argument("--dataset", default=DEFAULT_DATABENTO_DATASET, help="Databento dataset code.")
    parser.add_argument("--stype-in", default=DEFAULT_DATABENTO_STYPE_IN, help="Databento stype_in, default parent.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for the fetch request.")
    parser.add_argument("--estimate-only", action="store_true", help="Only print cost metadata, do not download data.")
    parser.add_argument("--bundle", action="store_true", help="Estimate definition and cbbo-1m together.")
    parser.add_argument("--output", default=None, help="Optional output path for parquet/csv download.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    normalized_symbol = normalize_option_parent_symbol(args.symbol) if args.stype_in == "parent" else args.symbol

    if args.bundle:
        estimates = estimate_request_bundle(
            symbol=normalized_symbol,
            start=args.start,
            end=args.end,
            dataset=args.dataset,
            stype_in=args.stype_in,
        )
        print(json.dumps(estimates, indent=2))
        return 0

    estimate = estimate_request(
        symbol=normalized_symbol,
        start=args.start,
        end=args.end,
        schema=args.schema,
        dataset=args.dataset,
        stype_in=args.stype_in,
        limit=args.limit,
    )
    print(json.dumps(estimate, indent=2))

    if args.estimate_only:
        return 0

    if args.output:
        saved = fetch_and_save_range(
            symbol=normalized_symbol,
            start=args.start,
            end=args.end,
            output_path=args.output,
            schema=args.schema,
            dataset=args.dataset,
            stype_in=args.stype_in,
            limit=args.limit,
        )
        print(f"saved_to={saved}")
        return 0

    df = fetch_range_df(
        symbol=normalized_symbol,
        start=args.start,
        end=args.end,
        schema=args.schema,
        dataset=args.dataset,
        stype_in=args.stype_in,
        limit=args.limit,
    )
    print(df.head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
