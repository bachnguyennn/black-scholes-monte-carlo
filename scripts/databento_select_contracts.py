#!/usr/bin/env python3
"""
Select a narrowed near-ATM SPX/XSP/SPY option universe from Databento
definitions, then estimate quote-download cost for each raw contract.
"""

from __future__ import annotations

import argparse
import json

from src.core.databento_provider import estimate_request, select_contracts_from_parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select near-ATM Databento option contracts and estimate cost.")
    parser.add_argument("--symbol", required=True, help="Underlying root, e.g. SPX or XSP.")
    parser.add_argument("--spot", required=True, type=float, help="Reference spot level used for ATM selection.")
    parser.add_argument("--start", required=True, help="Definition fetch start timestamp/date.")
    parser.add_argument("--end", required=True, help="Definition fetch end timestamp/date.")
    parser.add_argument("--as-of", required=True, help="Reference timestamp/date for DTE filtering.")
    parser.add_argument("--min-dte", type=int, default=7, help="Minimum DTE for selected contracts.")
    parser.add_argument("--max-dte", type=int, default=45, help="Maximum DTE for selected contracts.")
    parser.add_argument("--expiries", type=int, default=1, help="Number of nearest expiries to keep.")
    parser.add_argument("--strikes-per-type", type=int, default=2, help="Contracts per call/put side per expiry.")
    parser.add_argument("--schema", default="cbbo-1m", help="Quote schema for cost estimation.")
    parser.add_argument("--quote-start", default=None, help="Quote-cost start; defaults to --start.")
    parser.add_argument("--quote-end", default=None, help="Quote-cost end; defaults to --end.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = select_contracts_from_parent(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        target_spot=args.spot,
        as_of=args.as_of,
        min_dte_days=args.min_dte,
        max_dte_days=args.max_dte,
        expiries_count=args.expiries,
        strikes_per_type=args.strikes_per_type,
    )
    if selected.empty:
        print("No contracts matched the selection criteria.")
        return 0

    quote_start = args.quote_start or args.start
    quote_end = args.quote_end or args.end

    rows = []
    for row in selected.itertuples(index=False):
        estimate = estimate_request(
            symbol=row.raw_symbol,
            start=quote_start,
            end=quote_end,
            schema=args.schema,
            stype_in="raw_symbol",
        )
        rows.append(
            {
                "raw_symbol": row.raw_symbol,
                "expiration": row.expiration.isoformat(),
                "option_type": row.option_type,
                "strike_price": float(row.strike_price),
                "dte_days": float(row.dte_days),
                "atm_distance_pct": float(row.atm_distance_pct),
                "schema": args.schema,
                "estimated_cost_usd": estimate["estimated_cost_usd"],
                "record_count": estimate["record_count"],
            }
        )

    print(json.dumps(rows, indent=2))
    print(f"total_estimated_cost_usd={sum(row['estimated_cost_usd'] for row in rows):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
