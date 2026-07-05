"""
common.py — Shared helpers for the Streamlit web app.

Single source of truth for the cached data fetchers, model-name mapping,
formatters, table styling, and small layout helpers that the tabs share, so
each tab file stays thin and the UI stays consistent.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from src.core.data_fetcher import (
    get_available_expirations,
    get_options_chain,
    get_risk_free_rate,
    get_spot_and_vol,
)

# ---------------------------------------------------------------------------
# Model display name -> engine key
# ---------------------------------------------------------------------------
MODEL_MAP = {
    "Standard GBM": "gbm",
    "Jump Diffusion": "jump_diffusion",
    "Heston (Stochastic Vol)": "heston",
    "LSV (Local Stochastic Vol)": "lsv",
}


def model_key(model_type, default="heston"):
    """Map a sidebar model label to its engine key."""
    return MODEL_MAP.get(model_type, default)


# ---------------------------------------------------------------------------
# Cached data fetchers (shared across tabs)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_available_expirations(ticker):
    return get_available_expirations(ticker)


@st.cache_data(ttl=120)
def fetch_spot_data(ticker):
    return get_spot_and_vol(ticker)


@st.cache_data(ttl=300)
def fetch_risk_free_rate():
    return get_risk_free_rate()


@st.cache_data(ttl=300)
def fetch_options_chain(ticker, max_expirations=10):
    return get_options_chain(ticker, max_expirations=max_expirations)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def format_metric(value, fmt):
    """Format a numeric value, returning 'n/a' for missing/invalid input."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return format(numeric, fmt)


def format_as_of_timestamp(value):
    """Render a pandas/py timestamp as a UTC string, tolerant of tz-naive input."""
    if value is None:
        return "n/a"
    try:
        return value.tz_convert("UTC").strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        try:
            return value.tz_localize("UTC").strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            try:
                return value.strftime("%Y-%m-%d %H:%M")
            except Exception:
                return str(value)


def spot_provenance_warning(spot_data):
    """Build a one-line provenance warning from a spot/history payload (or '')."""
    if not spot_data:
        return ""
    parts = []
    if spot_data.get("fallback_from"):
        parts.append(
            f"spot/history fell back from {spot_data['fallback_from']} "
            f"to {spot_data.get('provider', 'yfinance')}"
        )
    if spot_data.get("is_stale"):
        parts.append("spot/history snapshot is flagged stale")
    if spot_data.get("validation_warnings"):
        parts.append("validation: " + " ".join(spot_data["validation_warnings"]))
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Expiration selection
# ---------------------------------------------------------------------------
def build_expiration_choices(available_exps):
    """Turn raw expiration strings into (label, iso_date, days_left) tuples,
    dropping anything already expired."""
    today = datetime.now().date()
    choices = []
    for exp_str in available_exps:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            days_left = (exp_date - today).days
            if days_left < 0:
                continue
            label = f"{exp_date.strftime('%d %b')}  ·  {days_left}d"
            choices.append((label, exp_str, days_left))
        except Exception:
            continue
    return choices


def select_expirations(label, available_exps, key=None, max_default=2):
    """Render a multiselect of expirations and return the chosen ISO dates."""
    choices = build_expiration_choices(available_exps)
    if not choices:
        st.error("Could not fetch active future expiration dates.")
        return []
    exp_labels = {lbl: iso for lbl, iso, _ in choices}
    preferred = [lbl for lbl, _, days_left in choices if days_left >= 1]
    default_labels = (preferred or [lbl for lbl, _, _ in choices])[:max_default]
    chosen = st.multiselect(
        label,
        options=list(exp_labels.keys()),
        default=default_labels,
        key=key,
        help="Labels show Date · Days to Expiry.",
    )
    return [exp_labels[lbl] for lbl in chosen]


# ---------------------------------------------------------------------------
# Table styling
# ---------------------------------------------------------------------------
_GREEN = "color: #16A34A; font-weight: 600"
_RED = "color: #DC2626; font-weight: 600"


def style_signal(val):
    text = str(val)
    if "BUY" in text:
        return _GREEN
    if "SELL" in text:
        return _RED
    return "color: #6B7280"


def style_edge(val):
    try:
        v = float(val)
    except (TypeError, ValueError):
        return ""
    if v > 0:
        return "color: #16A34A"
    if v < 0:
        return "color: #DC2626"
    return ""


def style_result(val):
    if val == "WIN":
        return _GREEN
    if val == "LOSS":
        return _RED
    return ""


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def section(title, caption=None):
    """Consistent section header: a subheader with an optional muted caption."""
    st.subheader(title)
    if caption:
        st.caption(caption)
