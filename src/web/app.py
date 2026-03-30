"""
app.py 

Architecture:
    src/web/tabs/
        tab_option_analysis.py  — Tab 1: Single option pricing & path simulation
        tab_scanner.py          — Tab 2: Live valuation gap scanner (async FastAPI)
        tab_model_validation.py — Tab 3: Quote-based live model validation
        tab_backtester.py       — Tab 4: Historical backtest (Heston / Jump Diffusion)
        tab_portfolio_risk.py   — Tab 5: 3D Gamma/Vanna/Vega vectorized surfaces
"""

import streamlit as st
import sys
import os
from datetime import timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.data_fetcher import get_market_data_runtime_summary, get_spot_and_vol
from src.core.heston_model import feller_condition
from src.web.tabs import tab_option_analysis, tab_scanner, tab_model_validation, tab_backtester, tab_portfolio_risk


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

# ============================================================================
# PAGE CONFIG & CSS
# ============================================================================

st.set_page_config(
    page_title="MC Options Pricing Engine",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
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
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #A0A0A0 !important; font-size: 0.8rem !important; text-transform: uppercase; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { padding-top: 10px; padding-bottom: 10px; font-weight: bold; }
    .dataframe { font-size: 0.85rem !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='margin-bottom:0'>Monte Carlo Options Pricing Engine</h1>
<p style='color:#A0A0A0; margin-top:2px; font-size:0.9rem'>
Black-Scholes &nbsp;|&nbsp; GBM Monte Carlo &nbsp;|&nbsp; Heston &nbsp;|&nbsp; Jump Diffusion &nbsp;|&nbsp; LSV
</p>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR — Shared State
# ============================================================================

st.sidebar.header("Asset Selection")
ticker = st.sidebar.text_input("Ticker Symbol", value="^SPX",
    help="European-style S&P 500 workflow: prefer ^SPX. SPY, IVV, and VOO are ETF proxies with American-style exercise.").strip()

if ticker.upper() in {"SPY", "IVV", "VOO"}:
    st.sidebar.warning("This ticker is an ETF proxy. Its listed options are typically American-style, not European-style index options.")
elif ticker.upper() in {"^SPX", "SPX"}:
    st.sidebar.success("Using S&P 500 index exposure. This is the best fit for a European-style workflow in the current app.")

market_data_summary = get_market_data_runtime_summary()
st.sidebar.caption(f"Market data preference: `{market_data_summary['provider_preference']}`")
st.sidebar.caption(market_data_summary["note"])
if market_data_summary["provider_preference"] != "yfinance":
    st.sidebar.caption(
        "Options chains still use yfinance in this app because Polygon free/basic is only integrated for supported REST workflows."
    )


@st.cache_data(ttl=3600)
def fetch_asset_data(symbol):
    spot_data = get_spot_and_vol(symbol)
    if not spot_data:
        return None
    return {
        'price': float(spot_data['spot']),
        'vol': min(max(float(spot_data['historical_vol']), 0.05), 1.0),
        'history': spot_data['history'],
        'name': spot_data['name'],
        'provider': spot_data.get('provider', 'yfinance'),
        'requested_provider': spot_data.get('requested_provider', spot_data.get('provider', 'yfinance')),
        'fallback_from': spot_data.get('fallback_from'),
        'provider_note': spot_data.get('provider_note', ''),
        'as_of': spot_data.get('as_of'),
        'history_points': int(spot_data.get('history_points', len(spot_data['history']))),
        'is_stale': bool(spot_data.get('is_stale', False)),
        'validation_warnings': list(spot_data.get('validation_warnings', [])),
    }


asset_data = fetch_asset_data(ticker)

if asset_data:
    st.sidebar.success(f"Loaded: {asset_data['name']} ({asset_data.get('provider', 'yfinance')})")
    if asset_data.get('provider_note'):
        st.sidebar.caption(asset_data['provider_note'])
    st.sidebar.markdown("**Data Source Status**")
    st.sidebar.caption(
        "  \n".join([
            f"requested provider: `{asset_data.get('requested_provider', 'n/a')}`",
            f"resolved provider: `{asset_data.get('provider', 'n/a')}`",
            f"fallback active: `{'yes' if asset_data.get('fallback_from') else 'no'}`",
            f"as-of timestamp: `{_format_as_of_timestamp(asset_data.get('as_of'))}`",
            f"history points: `{asset_data.get('history_points', 0)}`",
            f"stale flag: `{'yes' if asset_data.get('is_stale') else 'no'}`",
        ])
    )
    if asset_data.get('validation_warnings'):
        st.sidebar.warning("Validation warnings: " + " ".join(asset_data['validation_warnings']))
    default_spot = asset_data['price']
    default_vol = asset_data['vol']
else:
    st.sidebar.warning("Using manual defaults.")
    st.sidebar.markdown("**Data Source Status**")
    st.sidebar.caption(
        "  \n".join([
            "requested provider: `n/a`",
            "resolved provider: `manual`",
            "fallback active: `n/a`",
            "as-of timestamp: `n/a`",
            "history points: `n/a`",
            "stale flag: `n/a`",
        ])
    )
    default_spot = 100.0
    default_vol = 0.2

st.sidebar.markdown("---")
st.sidebar.header("Model Parameters")

model_type = st.sidebar.radio(
    "Simulation Model",
    ["Standard GBM", "Jump Diffusion", "Heston (Stochastic Vol)"],
    help="GBM: Constant vol. Jump Diffusion: Crash events. Heston: Vol is random."
)

# --- Preset Model Configurations ---
st.sidebar.markdown("**Quick Presets:**")
preset = st.sidebar.selectbox(
    "Load Configuration",
    ["Custom", "GBM (Baseline)", "Jump Diffusion (2008 Crisis)", "Heston (Quarterly Earnings)", "LSV (Smile Capture)"],
    help="Load predefined parameter sets for common scenarios"
)

# Apply preset if selected (updates session state for parameter inputs)
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
        "n_sims": 10000
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
        "n_sims": 25000
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
        "n_sims": 15000
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
        "n_sims": 20000
    }
}

if preset != "Custom" and preset in preset_configs:
    config = preset_configs[preset]
    st.session_state['heston_V0'] = config['heston_V0']
    st.session_state['heston_kappa'] = config['heston_kappa']
    st.session_state['heston_theta'] = config['heston_theta']
    st.session_state['heston_xi'] = config['heston_xi']
    st.session_state['heston_rho'] = config['heston_rho']
    st.sidebar.success(f"Preset Loaded: {preset}")


if model_type == "Jump Diffusion":
    with st.sidebar.expander("Crash Parameters", expanded=True):
        jump_intensity = st.number_input("Crash Intensity (lambda)", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                                        help="Probability of a crash event occurring per year")
        jump_mean = st.number_input("Avg Crash Size (%)", min_value=-20, max_value=0, value=-5, step=1,
                                   help="Average magnitude of crash (negative %)") / 100
        jump_std = st.number_input("Crash Vol (%)", min_value=1, max_value=10, value=3, step=1,
                                  help="Volatility of crash size (%)") / 100
else:
    jump_intensity, jump_mean, jump_std = 0.0, 0.0, 0.0

if model_type == "Heston (Stochastic Vol)":
    with st.sidebar.expander("Heston Parameters", expanded=True):
        h_v0    = st.session_state.get('heston_V0',    round(default_vol**2, 4))
        h_kappa = st.session_state.get('heston_kappa', 2.0)
        h_theta = st.session_state.get('heston_theta', round(default_vol**2, 4))
        h_xi    = st.session_state.get('heston_xi',    0.3)
        h_rho   = st.session_state.get('heston_rho',   -0.7)

        heston_V0    = st.number_input("Initial Variance (V0)", min_value=0.01, max_value=1.0,  value=h_v0,    step=0.01,  format="%.4f", key="input_v0",
                                      help="Initial volatility squared")
        heston_kappa = st.number_input("Mean Reversion (kappa)", min_value=0.1,  max_value=10.0, value=h_kappa, step=0.1, key="input_kappa",
                                      help="Speed of mean reversion (higher = faster return to theta)")
        heston_theta = st.number_input("Long-Run Var (theta)",   min_value=0.01, max_value=1.0,  value=h_theta, step=0.01,  format="%.4f", key="input_theta",
                                      help="Long-term average volatility squared")
        heston_xi    = st.number_input("Vol of Vol (xi)",        min_value=0.1,  max_value=2.0,  value=h_xi,    step=0.1, key="input_xi",
                                      help="Volatility of volatility (vol clustering)")
        heston_rho   = st.number_input("Correlation (rho)",      min_value=-1.0, max_value=0.0,  value=h_rho,   step=0.05, key="input_rho",
                                      help="Correlation between spot and vol (leverage effect)")

        fc = feller_condition(heston_kappa, heston_theta, heston_xi)
        if not fc['satisfied']:
            st.sidebar.error(f"{fc['message']}")
        else:
            st.sidebar.success(f"Feller OK (D={fc['discriminant']:.4f})")
else:
    heston_V0 = default_vol**2
    heston_kappa, heston_theta, heston_xi, heston_rho = 2.0, default_vol**2, 0.3, -0.7

st.sidebar.markdown("---")
n_sims = st.sidebar.number_input("Simulations (N)", min_value=1000, max_value=100000, value=10000, step=1000)

# ============================================================================
# TABS — delegate to modules
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Option Pricing", "Valuation Scanner", "Model Validation", "Backtester", "Risk Surfaces"
])

with tab1:
    tab_option_analysis.render(
        ticker=ticker, asset_data=asset_data,
        model_type=model_type, n_sims=n_sims,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
        heston_V0=heston_V0, heston_kappa=heston_kappa,
        heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho,
        default_spot=default_spot, default_vol=default_vol
    )

with tab2:
    tab_scanner.render(
        ticker=ticker, model_type=model_type,
        default_vol=default_vol, n_sims=n_sims,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
        heston_V0=heston_V0, heston_kappa=heston_kappa,
        heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho
    )

with tab3:
    tab_model_validation.render(
        ticker=ticker, model_type=model_type,
        default_vol=default_vol, n_sims=n_sims,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
        heston_V0=heston_V0, heston_kappa=heston_kappa,
        heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho
    )

with tab4:
    tab_backtester.render(
        ticker=ticker, option_type="call", n_sims=n_sims,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
        heston_V0=heston_V0, heston_kappa=heston_kappa,
        heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho
    )

with tab5:
    tab_portfolio_risk.render(ticker=ticker, default_spot=default_spot, asset_data=asset_data)
