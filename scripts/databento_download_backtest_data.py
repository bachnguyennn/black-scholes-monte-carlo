#!/usr/bin/env python3
"""
Build and optionally execute a Databento SPX/XSP quote-download plan for the
historical backtester.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.core.databento_provider import (
    build_backtest_download_plan,
    default_output_dir,
    execute_download_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally execute a Databento backtest data download plan.")
    parser.add_argument("--symbol", required=True, help="Underlying option root, e.g. SPX or XSP.")
    parser.add_argument("--rebalance-start", required=True, help="Inclusive rebalance schedule start date.")
    parser.add_argument("--rebalance-end", required=True, help="Inclusive rebalance schedule end date.")
    parser.add_argument("--frequency", default="BMS", help="Pandas date_range frequency, default business month start.")
    parser.add_argument("--spot-ticker", default=None, help="Override underlying spot ticker, default inferred from symbol.")
    parser.add_argument("--min-dte", type=int, default=7, help="Minimum DTE for selected contracts.")
    parser.add_argument("--max-dte", type=int, default=45, help="Maximum DTE for selected contracts.")
    parser.add_argument("--expiries", type=int, default=1, help="Number of nearest expiries to keep per rebalance.")
    parser.add_argument("--strikes-per-type", type=int, default=2, help="Near-ATM contracts per call/put side.")
    parser.add_argument("--schema", default="cbbo-1m", help="Quote schema to estimate/download.")
    parser.add_argument("--output-dir", default=None, help="Output directory for parquet files and manifest.")
    parser.add_argument("--download", action="store_true", help="Execute the quote downloads after building the plan.")
    parser.add_argument("--manifest-name", default="download_manifest.csv", help="Manifest filename written under output-dir.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan_df = build_backtest_download_plan(
        symbol=args.symbol,
        rebalance_start=args.rebalance_start,
        rebalance_end=args.rebalance_end,
        frequency=args.frequency,
        spot_ticker=args.spot_ticker,
        min_dte_days=args.min_dte,
        max_dte_days=args.max_dte,
        expiries_count=args.expiries,
        strikes_per_type=args.strikes_per_type,
        quote_schema=args.schema,
    )

    if plan_df.empty:
        print("No contracts were selected for the requested schedule.")
        return 0

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.symbol, args.schema)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    if not args.download:
        plan_df.to_csv(manifest_path, index=False)
        print(plan_df.to_string(index=False))
        print(f"\nrows={len(plan_df)}")
        print(f"total_estimated_cost_usd={plan_df['estimated_cost_usd'].sum():.8f}")
        print(f"manifest_path={manifest_path}")
        return 0

    executed_df = execute_download_plan(plan_df, output_dir=output_dir)
    executed_df.to_csv(manifest_path, index=False)
    print(executed_df.to_string(index=False))
    print(f"\nrows={len(executed_df)}")
    print(f"total_estimated_cost_usd={executed_df['estimated_cost_usd'].sum():.8f}")
    print(f"manifest_path={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
