"""
pricing_app.py

Standalone Streamlit app for single-option pricing only.
This keeps the full multi-tab terminal intact while providing
a lightweight pricing-focused entrypoint.
"""

import os
import sys
from datetime import timezone

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.core.data_fetcher import get_spot_and_vol
from src.core.heston_model import feller_condition
from src.web.tabs import tab_option_analysis


def _format_as_of_timestamp(value):
    if value is None:
        return "n/a"
    try:
        timestamp = value
        if getattr(timestamp, "tzinfo", None) is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        else:
            timestamp = timestamp.tz_convert(timezone.utc)
        return timestamp.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return str(value)


st.set_page_config(
    page_title="Option Pricing (Standalone)",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Fira Code', 'Roboto Mono', 'Courier New', monospace !important;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 98% !important;
    }
    .stMetric {
        background-color: #1e1e1e;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<h1 style='margin-bottom:0'>Option Pricing (Standalone)</h1>
<p style='color:#A0A0A0; margin-top:2px; font-size:0.9rem'>
Black-Scholes &nbsp;|&nbsp; GBM Monte Carlo &nbsp;|&nbsp; Jump Diffusion &nbsp;|&nbsp; Heston
</p>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("1) Asset")
ticker = st.sidebar.text_input("Ticker", value="^SPX").strip()


@st.cache_data(ttl=600)
def fetch_asset_data(symbol):
    spot_data = get_spot_and_vol(symbol)
    if not spot_data:
        return None
    return {
        "price": float(spot_data["spot"]),
        "vol": min(max(float(spot_data["historical_vol"]), 0.05), 1.0),
        "history": spot_data["history"],
        "name": spot_data["name"],
        "provider": spot_data.get("provider", "yfinance"),
        "as_of": spot_data.get("as_of"),
    }


asset_data = fetch_asset_data(ticker)

if asset_data:
    st.sidebar.success(f"Loaded: {asset_data['name']} ({asset_data.get('provider', 'yfinance')})")
    st.sidebar.caption(f"as-of: `{_format_as_of_timestamp(asset_data.get('as_of'))}`")
    default_spot = asset_data["price"]
    default_vol = asset_data["vol"]
else:
    st.sidebar.warning("Data unavailable; using manual defaults.")
    default_spot = 100.0
    default_vol = 0.2

st.sidebar.markdown("---")
st.sidebar.header("2) Simulation Setup")

preset_configs = {
    "GBM (Baseline)": {
        "model": "Standard GBM",
        "heston_V0": 0.04,
        "heston_kappa": 2.0,
        "heston_theta": 0.04,
        "heston_xi": 0.3,
        "heston_rho": -0.7,
        "jump_intensity": 0.0,
        "jump_mean": 0.0,
        "jump_std": 0.0,
        "n_sims": 10000,
    },
    "Jump Diffusion (2008 Crisis)": {
        "model": "Jump Diffusion",
        "heston_V0": 0.04,
        "heston_kappa": 2.0,
        "heston_theta": 0.04,
        "heston_xi": 0.3,
        "heston_rho": -0.7,
        "jump_intensity": 0.25,
        "jump_mean": -0.10,
        "jump_std": 0.05,
        "n_sims": 25000,
    },
    "Heston (Quarterly Earnings)": {
        "model": "Heston (Stochastic Vol)",
        "heston_V0": 0.04,
        "heston_kappa": 5.0,
        "heston_theta": 0.04,
        "heston_xi": 0.5,
        "heston_rho": -0.8,
        "jump_intensity": 0.0,
        "jump_mean": 0.0,
        "jump_std": 0.0,
        "n_sims": 15000,
    },
    "LSV (Smile Capture)": {
        "model": "Heston (Stochastic Vol)",
        "heston_V0": 0.04,
        "heston_kappa": 3.0,
        "heston_theta": 0.04,
        "heston_xi": 0.4,
        "heston_rho": -0.65,
        "jump_intensity": 0.0,
        "jump_mean": 0.0,
        "jump_std": 0.0,
        "n_sims": 20000,
    },
}

st.sidebar.markdown("**Quick Presets**")
preset = st.sidebar.selectbox(
    "Preset",
    ["GBM (Baseline)", "Jump Diffusion (2008 Crisis)", "Heston (Quarterly Earnings)", "LSV (Smile Capture)"],
)
config = preset_configs[preset]
model_type = config["model"]
st.sidebar.caption(f"Active model: `{model_type}`")

if st.session_state.get("pricing_only_active_preset") != preset:
    st.session_state["pricing_only_active_preset"] = preset
    st.session_state["pricing_only_heston_V0"] = config["heston_V0"]
    st.session_state["pricing_only_heston_kappa"] = config["heston_kappa"]
    st.session_state["pricing_only_heston_theta"] = config["heston_theta"]
    st.session_state["pricing_only_heston_xi"] = config["heston_xi"]
    st.session_state["pricing_only_heston_rho"] = config["heston_rho"]
    st.session_state["pricing_only_jump_intensity"] = config["jump_intensity"]
    st.session_state["pricing_only_jump_mean_pct"] = int(round(config["jump_mean"] * 100))
    st.session_state["pricing_only_jump_std_pct"] = int(round(config["jump_std"] * 100))
    st.session_state["pricing_only_n_sims"] = int(config["n_sims"])

if model_type == "Jump Diffusion":
    with st.sidebar.expander("Crash Parameters", expanded=True):
        jump_intensity = st.number_input(
            "Crash Intensity (lambda)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("pricing_only_jump_intensity", config["jump_intensity"])),
            step=0.05,
            key="pricing_only_jump_intensity",
        )
        jump_mean_pct = st.number_input(
            "Avg Crash Size (%)",
            min_value=-20,
            max_value=0,
            value=int(st.session_state.get("pricing_only_jump_mean_pct", int(round(config["jump_mean"] * 100)))),
            step=1,
            key="pricing_only_jump_mean_pct",
        )
        jump_std_pct = st.number_input(
            "Crash Vol (%)",
            min_value=1,
            max_value=10,
            value=int(st.session_state.get("pricing_only_jump_std_pct", int(round(config["jump_std"] * 100)))),
            step=1,
            key="pricing_only_jump_std_pct",
        )
        jump_mean = jump_mean_pct / 100
        jump_std = jump_std_pct / 100
else:
    jump_intensity, jump_mean, jump_std = 0.0, 0.0, 0.0

if model_type == "Heston (Stochastic Vol)":
    with st.sidebar.expander("Heston Parameters", expanded=True):
        heston_V0 = st.number_input(
            "Initial Variance (V0)",
            min_value=0.01,
            max_value=1.0,
            value=float(st.session_state.get("pricing_only_heston_V0", round(default_vol**2, 4))),
            step=0.01,
            format="%.4f",
            key="pricing_only_heston_V0",
        )
        heston_kappa = st.number_input(
            "Mean Reversion (kappa)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("pricing_only_heston_kappa", 2.0)),
            step=0.1,
            key="pricing_only_heston_kappa",
        )
        heston_theta = st.number_input(
            "Long-Run Var (theta)",
            min_value=0.01,
            max_value=1.0,
            value=float(st.session_state.get("pricing_only_heston_theta", round(default_vol**2, 4))),
            step=0.01,
            format="%.4f",
            key="pricing_only_heston_theta",
        )
        heston_xi = st.number_input(
            "Vol of Vol (xi)",
            min_value=0.1,
            max_value=2.0,
            value=float(st.session_state.get("pricing_only_heston_xi", 0.3)),
            step=0.1,
            key="pricing_only_heston_xi",
        )
        heston_rho = st.number_input(
            "Correlation (rho)",
            min_value=-1.0,
            max_value=0.0,
            value=float(st.session_state.get("pricing_only_heston_rho", -0.7)),
            step=0.05,
            key="pricing_only_heston_rho",
        )
        fc = feller_condition(heston_kappa, heston_theta, heston_xi)
        if fc["satisfied"]:
            st.sidebar.success(f"Feller OK (D={fc['discriminant']:.4f})")
        else:
            st.sidebar.error(fc["message"])
else:
    heston_V0 = default_vol**2
    heston_kappa, heston_theta, heston_xi, heston_rho = 2.0, default_vol**2, 0.3, -0.7

st.sidebar.markdown("---")
n_sims = st.sidebar.number_input(
    "Monte Carlo Paths",
    min_value=1000,
    max_value=100000,
    value=int(st.session_state.get("pricing_only_n_sims", config["n_sims"])),
    step=1000,
    key="pricing_only_n_sims",
)

tab_option_analysis.render(
    ticker=ticker,
    asset_data=asset_data,
    model_type=model_type,
    n_sims=n_sims,
    jump_intensity=jump_intensity,
    jump_mean=jump_mean,
    jump_std=jump_std,
    heston_V0=heston_V0,
    heston_kappa=heston_kappa,
    heston_theta=heston_theta,
    heston_xi=heston_xi,
    heston_rho=heston_rho,
    default_spot=default_spot,
    default_vol=default_vol,
)
