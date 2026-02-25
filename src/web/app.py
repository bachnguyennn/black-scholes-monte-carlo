
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os
import yfinance as yf
from datetime import timedelta, date

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm
from src.core.jump_diffusion import simulate_jump_diffusion, simulate_jump_diffusion_paths
from src.core.greeks import calculate_all_greeks

# --- Page Config ---
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: rgba(240, 242, 246, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("💰 European Option Pricing: Monte Carlo vs Black-Scholes")

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.header("Asset Selection")

ticker = st.sidebar.text_input(
    "Ticker Symbol", 
    value="SPY",
    help="Enter a Yahoo Finance ticker (e.g., AAPL, SPY, ^TYX for Bonds)."
)

@st.cache_data(ttl=3600)
def fetch_asset_data(symbol):
    try:
        asset = yf.Ticker(symbol)
        info = asset.info
        history = asset.history(period="1y")
        if history.empty:
            return None
        log_returns = np.log(history['Close'] / history['Close'].shift(1))
        vol = log_returns.std() * np.sqrt(252)
        return {
            'price': history['Close'].iloc[-1],
            'vol': vol,
            'history': history['Close'],
            'name': info.get('longName', symbol)
        }
    except:
        return None

asset_data = fetch_asset_data(ticker)

if asset_data:
    st.sidebar.success(f"Loaded: {asset_data['name']}")
    default_spot = float(asset_data['price'])
    default_vol = float(asset_data['vol'])
    default_vol = min(max(default_vol, 0.05), 1.0)
else:
    st.sidebar.warning("Using manual defaults.")
    default_spot = 100.0
    default_vol = 0.2

st.sidebar.markdown("---")
st.sidebar.header("Model Parameters")

model_type = st.sidebar.radio(
    "Choose Simulation Model",
    ["Standard GBM", "Jump Diffusion (Crash Model)"],
    help="GBM: Smooth continuous paths. Jump Diffusion: Includes crash events."
)

if model_type == "Jump Diffusion (Crash Model)":
    st.sidebar.markdown("### Crash Parameters")
    jump_intensity = st.sidebar.slider("Crash Intensity (λ)", 0.0, 1.0, 0.1, 0.05)
    jump_mean = st.sidebar.slider("Average Crash Size (%)", -20, 0, -5, 1) / 100
    jump_std = st.sidebar.slider("Crash Volatility (%)", 1, 10, 3, 1) / 100
else:
    jump_intensity, jump_mean, jump_std = 0.0, 0.0, 0.0

st.sidebar.markdown("---")
option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
current_price = st.sidebar.slider("Spot Price ($S_0$)", 1.0, 2000.0, default_spot, 1.0)
strike_price = st.sidebar.slider("Strike Price ($K$)", 1.0, 2000.0, default_spot, 1.0)
time_to_maturity = st.sidebar.slider("Time to Maturity ($T$ Years)", 0.1, 5.0, 1.0, 0.1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate ($r$)", 0.0, 0.2, 0.05, 0.01)
volatility = st.sidebar.slider("Volatility (σ)", 0.05, 1.5, default_vol, 0.05)

st.sidebar.markdown("---")
market_price = st.sidebar.number_input("Market Price ($)", 0.0, value=0.0, step=0.1)
n_sims = st.sidebar.slider("Simulations ($N$)", 100, 100000, 10000, 100)

# ============================================================================
# CALCULATION ENGINE
# ============================================================================

def _simulate_gbm_paths_for_plot(S0, T, r, sigma, n_paths, n_steps=60):
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    cumulative_log_returns = np.zeros((n_paths, n_steps + 1))
    cumulative_log_returns[:, 1:] = np.cumsum(log_returns, axis=1)
    return S0 * np.exp(cumulative_log_returns)

# 1. Prices
bs_price = black_scholes_price(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())

if model_type == "Jump Diffusion (Crash Model)":
    S_T, crash_mask = simulate_jump_diffusion(current_price, time_to_maturity, risk_free_rate, volatility, n_sims, jump_intensity, jump_mean, jump_std)
else:
    S_T = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
    crash_mask = np.zeros(n_sims, dtype=bool)

# 2. Results
if option_type == "Call":
    payoffs = np.maximum(S_T - strike_price, 0)
else:
    payoffs = np.maximum(strike_price - S_T, 0)

mc_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-risk_free_rate * time_to_maturity)

# ============================================================================
# LAYOUT TOP: METRICS
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Black-Scholes", f"${bs_price:.4f}")
with col2:
    delta = mc_price - bs_price
    st.metric("Monte Carlo", f"${mc_price:.4f}", delta=f"{delta:.4f}", delta_color="inverse")
with col3:
    st.metric("Standard Error", f"±${std_error:.4f}")
with col4:
    if model_type == "Jump Diffusion (Crash Model)":
        cp = np.sum(crash_mask) / n_sims * 100
        st.metric("Crash Prob", f"{cp:.1f}%")
    else:
        st.metric("Risk Model", "GBM")

# ============================================================================
# LAYOUT: CHARTS
# ============================================================================
st.markdown("---")

# 1. HISTORICAL + PROJECTION (Full Width)
st.subheader("Asset Price Projections")
n_plot_paths = 50
if model_type == "Jump Diffusion (Crash Model)":
    sim_paths, path_crash_mask = simulate_jump_diffusion_paths(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths, jump_intensity, jump_mean, jump_std)
else:
    sim_paths = _simulate_gbm_paths_for_plot(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths)
    path_crash_mask = np.zeros(n_plot_paths, dtype=bool)

time_axis = np.linspace(0, time_to_maturity, sim_paths.shape[1])

fig_paths = go.Figure()

# History
if asset_data and 'history' in asset_data:
    history = asset_data['history']
    x_hist = np.linspace(-len(history)/252, 0, len(history))
    fig_paths.add_trace(go.Scatter(x=x_hist, y=history.values, mode='lines', name='Historical', line=dict(color='green', width=2)))

# Paths
for i in range(n_plot_paths):
    color = 'rgba(255, 50, 50, 0.3)' if path_crash_mask[i] else 'rgba(0, 100, 255, 0.2)'
    fig_paths.add_trace(go.Scatter(x=time_axis, y=sim_paths[i], mode='lines', line=dict(width=1, color=color), showlegend=False, hoverinfo='skip'))

fig_paths.add_hline(y=strike_price, line_dash="dash", line_color="red", annotation_text="Strike")
fig_paths.update_layout(xaxis_title="Years from Today", yaxis_title="Price ($)", height=450, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
st.plotly_chart(fig_paths, use_container_width=True)

# 2. DISTRIBUTION + VOL HEATMAP (Columns)
st.markdown("---")
dist_col, heat_col = st.columns(2)

with dist_col:
    st.subheader("Terminal Price Distribution")
    fig_hist = px.histogram(S_T, nbins=50, title=None)
    fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="red")
    fig_hist.update_layout(xaxis_title="Price at Maturity", yaxis_title="Frequency", height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_hist, use_container_width=True)

with heat_col:
    st.subheader("Price Sensitivity (Spot vs Vol)")
    s_range = np.linspace(current_price * 0.8, current_price * 1.2, 10)
    v_range = np.linspace(0.1, 0.6, 10)
    z_prices = np.zeros((10, 10))
    for i, s_tmp in enumerate(s_range):
        for j, v_tmp in enumerate(v_range):
            z_prices[j, i] = black_scholes_price(s_tmp, strike_price, time_to_maturity, risk_free_rate, v_tmp, option_type.lower())
    fig_heat = go.Figure(data=go.Heatmap(z=z_prices, x=s_range, y=v_range, colorscale='Viridis', hovertemplate="Spot: $%{x:.2f}<br>Vol: %{y:.2f}<br>Price: $%{z:.2f}<extra></extra>"))
    fig_heat.update_layout(xaxis_title="Spot Price", yaxis_title="Volatility", height=400, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_heat, use_container_width=True)

# ============================================================================
# ANALYSIS & DETAILS (Bottom)
# ============================================================================
st.markdown("---")
detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    with st.expander("Greeks & Sensitivities"):
        if st.sidebar.button("Calculate Greeks") or True: # Auto-show if possible
            greeks_params = {'jump_intensity': jump_intensity, 'jump_mean': jump_mean, 'jump_std': jump_std} if model_type == "Jump Diffusion (Crash Model)" else {}
            model_name = 'jump_diffusion' if model_type == "Jump Diffusion (Crash Model)" else 'gbm'
            greeks = calculate_all_greeks(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower(), model=model_name, n_sims=5000, **greeks_params)
            
            g_col1, g_col2, g_col3 = st.columns(3)
            g_col1.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
            g_col2.metric("Vega (ν)", f"{greeks['vega']:.4f}")
            g_col3.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}")

with detail_col2:
    if market_price > 0:
        model_p = bs_price
        diff = model_p - market_price
        st.info(f"Signal: **{'BUY (Undervalued)' if diff > 0 else 'SELL (Overvalued)'}** (Diff: ${abs(diff):.2f})")

if model_type == "Jump Diffusion (Crash Model)":
    with st.expander("Model Comparison Details"):
        # Quick GBM recalc for comparison
        S_T_gbm = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
        pay_gbm = np.maximum(S_T_gbm - strike_price, 0) if option_type == "Call" else np.maximum(strike_price - S_T_gbm, 0)
        mc_gbm = np.exp(-risk_free_rate * time_to_maturity) * np.mean(pay_gbm)
        
        comp_df = pd.DataFrame({
            "Metric": ["MC Price", "Crash Premium", "95% VaR"],
            "Standard GBM": [f"${mc_gbm:.4f}", "-", f"${np.percentile(S_T_gbm, 5):.2f}"],
            "Jump Diffusion": [f"${mc_price:.4f}", f"${mc_price - mc_gbm:.4f}", f"${np.percentile(S_T, 5):.2f}"]
        })
        st.table(comp_df)

st.success("Dashboard Ready")
