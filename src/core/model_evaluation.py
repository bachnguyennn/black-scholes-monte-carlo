"""
model_evaluation.py

Surface-fit diagnostics for comparing model prices against observed option
quotes. These metrics are intended for live/current option-chain evaluation,
not for the synthetic historical backtester.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.core.black_scholes import black_scholes_price


def _implied_vol_from_price(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Back-solves Black-Scholes implied vol from a model price."""
    intrinsic = max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)
    if price <= intrinsic + 1e-6 or T <= 0:
        return float("nan")

    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        bs_price = black_scholes_price(S0, K, T, r, mid, option_type, q=q)
        if bs_price < price:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break

    iv = (lo + hi) / 2.0
    return iv if 0.001 < iv < 4.9 else float("nan")


def build_live_surface_evaluation(
    priced_df: pd.DataFrame,
    S0: float,
    r: float,
    q: float = 0.0,
) -> dict[str, Any]:
    """
    Computes live-surface fit metrics from the scanner output.

    Metrics are quote-based and execution-aware:
    - price MAE / RMSE versus mid
    - IV MAE versus quoted market IV when available
    - mean absolute error measured in quoted spreads
    - percent of model prices landing inside the quoted NBBO
    """
    required = {"strike", "type", "bid", "ask", "mid", "mc_price"}
    missing = required.difference(priced_df.columns)
    if priced_df.empty or missing:
        return {
            "success": False,
            "message": "Surface evaluation unavailable: priced contracts or required columns are missing.",
            "missing_columns": sorted(missing),
        }

    df = priced_df.copy()
    if "T_days" in df.columns:
        df["T_years"] = pd.to_numeric(df["T_days"], errors="coerce") / 365.0
    elif "T" in df.columns:
        df["T_years"] = pd.to_numeric(df["T"], errors="coerce")
    else:
        df["T_years"] = np.nan

    for col in ("strike", "bid", "ask", "mid", "mc_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    clean = df.dropna(subset=["strike", "bid", "ask", "mid", "mc_price"])
    clean = clean[(clean["mid"] > 0) & (clean["ask"] > clean["bid"])]
    if clean.empty:
        return {
            "success": False,
            "message": "Surface evaluation unavailable after quote cleaning.",
            "missing_columns": [],
        }

    clean["price_error"] = clean["mc_price"] - clean["mid"]
    clean["abs_price_error"] = clean["price_error"].abs()
    clean["spread"] = clean["ask"] - clean["bid"]
    clean["spread_units"] = clean["abs_price_error"] / clean["spread"].replace(0.0, np.nan)
    clean["within_nbbo"] = (clean["mc_price"] >= clean["bid"]) & (clean["mc_price"] <= clean["ask"])

    iv_errors: list[float] = []
    if "market_iv" in clean.columns:
        market_iv = pd.to_numeric(clean["market_iv"], errors="coerce")
        # Scanner output stores IV in percent units; normalize to decimals when needed.
        if float(market_iv.dropna().median()) > 3.0:
            market_iv = market_iv / 100.0
        clean["market_iv_decimal"] = market_iv

        for row in clean.itertuples(index=False):
            market_iv_value = getattr(row, "market_iv_decimal", float("nan"))
            if not math.isfinite(market_iv_value) or market_iv_value <= 0:
                continue

            T = getattr(row, "T_years", float("nan"))
            if not math.isfinite(T) or T <= 0:
                continue

            model_iv = _implied_vol_from_price(
                price=float(row.mc_price),
                S0=float(S0),
                K=float(row.strike),
                T=float(T),
                r=float(r),
                option_type=str(row.type).lower(),
                q=float(q),
            )
            if math.isfinite(model_iv):
                iv_errors.append(abs(model_iv - market_iv_value))

    return {
        "success": True,
        "message": "Live-surface fit metrics compare model values against current observed quotes.",
        "contracts_evaluated": int(len(clean)),
        "price_mae": float(clean["abs_price_error"].mean()),
        "price_rmse": float(np.sqrt(np.mean(np.square(clean["price_error"])))),
        "mean_abs_error_in_spreads": float(clean["spread_units"].dropna().mean()) if clean["spread_units"].notna().any() else float("nan"),
        "within_nbbo_pct": float(clean["within_nbbo"].mean() * 100.0),
        "iv_mae_pct_pts": float(np.mean(iv_errors) * 100.0) if iv_errors else float("nan"),
        "iv_contracts_evaluated": int(len(iv_errors)),
    }
