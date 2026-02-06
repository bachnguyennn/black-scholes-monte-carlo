
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm

# --- Page Config ---
st.set_page_config(
    page_title="Monte Carlo Option Pricing",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS to fix padding ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Function for Visualization ONLY ---
def _simulate_paths_for_plot(S0, T, r, sigma, n_paths, n_steps=100):
    """
    LOCAL HELPER: Generates full price paths for visualization.
    NOTE: The core engine 'simulate_gbm' is optimized for pricing (only returns terminal values).
    This function duplicates the path logic strictly for the UI chart.
    """
    dt = T / n_steps
    # Z ~ N(0, 1)
    Z = np.random.standard_normal((n_paths, n_steps))
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    log_returns = drift + diffusion
    
    # Cumulative sum for paths
    cumulative_log_returns = np.zeros((n_paths, n_steps + 1))
    cumulative_log_returns[:, 1:] = np.cumsum(log_returns, axis=1)
    
    return S0 * np.exp(cumulative_log_returns)

# --- Title ---
st.title("💰 European Option Pricing: Monte Carlo vs Black-Scholes")
st.markdown("---")

import yfinance as yf

# --- Sidebar Inputs ---
st.sidebar.header("Model Parameters")

# Ticker Input for Auto-Fetching
ticker = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)", value="")

# Dynamic Default Values
default_spot = 100.0
default_vol = 0.2

if ticker:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="1d")
        if not history.empty:
            spot = history['Close'].iloc[-1]
            default_spot = float(spot)
            st.sidebar.success(f"Fetched {ticker}: ${default_spot:.2f}")
            
            # Lock Spot Price to Fetched Value
            current_price = default_spot
            st.sidebar.metric("Spot Price ($S_0$)", f"${current_price:.2f}")
        else:
            st.sidebar.error("Ticker not found or no data.")
            current_price = st.sidebar.slider("Spot Price ($S_0$)", min_value=10.0, max_value=2000.0, value=default_spot, step=0.5)
    except Exception as e:
        st.sidebar.error(f"Error fetching data: {e}")
        current_price = st.sidebar.slider("Spot Price ($S_0$)", min_value=10.0, max_value=2000.0, value=default_spot, step=0.5)
else:
    # Manual Input
    current_price = st.sidebar.slider("Spot Price ($S_0$)", min_value=10.0, max_value=2000.0, value=default_spot, step=0.5)

option_type = st.sidebar.radio("Option Type", ["Call", "Put"])
strike_price = st.sidebar.slider("Strike Price ($K$)", min_value=10.0, max_value=2000.0, value=default_spot, step=0.5)
time_to_maturity = st.sidebar.slider("Time to Maturity ($T$ Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
risk_free_rate = st.sidebar.slider("Risk-Free Rate ($r$)", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
volatility = st.sidebar.slider("Volatility ($\sigma$)", min_value=0.05, max_value=1.5, value=default_vol, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Market Data (Optional)")
market_price = st.sidebar.number_input("Current Market Price ($)", min_value=0.0, value=0.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.header("Simulation Settings")
n_sims = st.sidebar.slider("Number of Simulations ($N$)", min_value=100, max_value=100000, value=10000, step=100)

# --- Main Logic ---

# 1. Black-Scholes Benchmark
bs_price = black_scholes_price(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type=option_type.lower())

# 2. Monte Carlo Simulation (Pricing)
# Correctly uses the strict core engine (returns terminal prices S_T)
S_T = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)

# Payoffs
if option_type == "Call":
    payoffs = np.maximum(S_T - strike_price, 0)
else: # Put
    payoffs = np.maximum(strike_price - S_T, 0)

mc_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)

# Standard Error
std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-risk_free_rate * time_to_maturity)

# 3. Path Visualization (UI Only)
n_plot_paths = 50
paths = _simulate_paths_for_plot(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths)
time_axis = np.linspace(0, time_to_maturity, paths.shape[1])

# --- Display Results ---

# Metrics Row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(f"Black-Scholes ({option_type})", f"${bs_price:.4f}")

with col2:
    delta = mc_price - bs_price
    st.metric(f"Monte Carlo ({option_type})", f"${mc_price:.4f}", delta=f"{delta:.4f}", delta_color="inverse")

with col3:
    st.metric("Standard Error", f"±${std_error:.4f}")

# Recommendation Row
if market_price > 0:
    st.markdown("### 🤖 AI Trading Signal")
    rec_col1, rec_col2 = st.columns([1, 3])
    
    model_price = bs_price # Trust BS more for single recommendation, or user could choose
    
    # Logic: 
    # If Model > Market => It's Cheap => BUY
    # If Model < Market => It's Expensive => SELL
    
    diff = model_price - market_price
    pct_diff = (diff / market_price) * 100
    
    with rec_col1:
        if diff > 0:
            st.success("✅ BUY", icon="🤑")
        else:
            st.error("❌ SELL", icon="💸")
            
    with rec_col2:
        if diff > 0:
            st.info(f"The option is **UNDERVALUED** by {pct_diff:.1f}%. Fair Value is ${model_price:.2f}, but Market says ${market_price:.2f}.")
        else:
            st.info(f"The option is **OVERVALUED** by {-pct_diff:.1f}%. Fair Value is ${model_price:.2f}, but Market says ${market_price:.2f}.")

st.markdown("---")

# Charts Row
chart_col, hist_col = st.columns(2)

# --- Simulation & Visualization Logic ---

# 1. Generate Future Paths
# We need full paths now, not just terminal prices for the plot
# Using the local helper _simulate_paths_for_plot
n_plot_paths = 50
sim_paths = _simulate_paths_for_plot(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths)

# 2. Date Handling
import pandas as pd
from datetime import timedelta, date

history_data = None
future_dates = []

if ticker:
    try:
        # Fetch 1 year history
        stock = yf.Ticker(ticker)
        history = stock.history(period="1y")
        if not history.empty:
            history_data = history['Close']
            last_date = history.index[-1].date()
    except:
        last_date = date.today()
else:
    # Synthetic history for manual mode (just a flat line or empty)
    last_date = date.today()

# Generate Future Dates
# Logic: Start from last_date, add days corresponding to simulation steps
n_steps = sim_paths.shape[1]
# Total days ~ T * 365
days_total = int(time_to_maturity * 365)
if days_total == 0: days_total = 1 # Safety
days_step = days_total / n_steps

future_dates = [last_date + timedelta(days=int(i * days_step)) for i in range(n_steps + 1)]


# 3. Plotting Combined Chart
fig_combined = go.Figure()

# Plot History (if exists)
if history_data is not None:
    fig_combined.add_trace(go.Scatter(
        x=history_data.index, 
        y=history_data.values,
        mode='lines',
        name='Historical Price',
        line=dict(color='black', width=2)
    ))

# Plot Future Paths
# Connect the start of simulation to the end of history if possible
# Note: sim_paths[i][0] is S0.
for i in range(n_plot_paths):
    fig_combined.add_trace(go.Scatter(
        x=future_dates, 
        y=sim_paths[i],
        mode='lines',
        line=dict(width=1, color='rgba(0, 100, 255, 0.2)'),
        showlegend=False,
        hoverinfo='skip' 
    ))

# Plot Strike Price Line (Future only)
fig_combined.add_shape(
    type="line",
    x0=future_dates[0], y0=strike_price,
    x1=future_dates[-1], y1=strike_price,
    line=dict(color="red", width=2, dash="dash"),
    name="Strike Price"
)
fig_combined.add_trace(go.Scatter(
    x=[future_dates[-1]], y=[strike_price],
    text=["Strike"], mode="text", showlegend=False
))

fig_combined.update_layout(
    title=f"Historical Price + Monte Carlo Projections ({n_plot_paths} Paths)",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    height=600
)

st.plotly_chart(fig_combined, use_container_width=True)

# Update S_T for histogram (Ensure consistent distribution)
S_T = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)

# Payoffs
if option_type == "Call":
    payoffs = np.maximum(S_T - strike_price, 0)
else: # Put
    payoffs = np.maximum(strike_price - S_T, 0)

mc_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-risk_free_rate * time_to_maturity)


# --- Distribution Histogram (Restored) ---
st.subheader(f"Distribution of Terminal Prices (N={n_sims})")
fig_hist = px.histogram(S_T, nbins=50, title="Probability Density of Future Prices")
fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="red", annotation_text="Strike")
fig_hist.add_vline(x=current_price * np.exp(risk_free_rate * time_to_maturity), line_dash="dot", line_color="green", annotation_text="Exp. Value")

fig_hist.update_layout(
    xaxis_title="Price at Maturity",
    yaxis_title="Frequency",
    showlegend=False,
    height=400
)
st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")

# --- Heatmap & Time Analysis ---
st.header("📅 Time & Volatility Analysis (Heatmap)")

# Range of Spot Prices and Volatilities for Heatmap
spot_range = np.linspace(default_spot * 0.8, default_spot * 1.2, 10)
vol_range = np.linspace(0.1, 0.5, 10)

# Create 2D Matrix of Prices using Current Time T
z_prices = np.zeros((10, 10))
for i, s_tmp in enumerate(spot_range):
    for j, v_tmp in enumerate(vol_range):
        z_prices[j, i] = black_scholes_price(
            s_tmp, strike_price, time_to_maturity, risk_free_rate, v_tmp, option_type.lower()
        )

fig_heatmap = go.Figure(data=go.Heatmap(
    z=z_prices,
    x=spot_range,
    y=vol_range,
    colorscale='Viridis',
    hovertemplate="Spot: %{x:.2f}<br>Vol: %{y:.2f}<br>Price: $%{z:.2f}<extra></extra>"
))

fig_heatmap.update_layout(
    title="Option Price Sensitivity (Spot Price vs. Volatility)",
    xaxis_title="Spot Price ($)",
    yaxis_title="Volatility ($\sigma$)",
    height=500
)

# Recommendation Engine Logic
st.plotly_chart(fig_heatmap, use_container_width=True)

st.markdown("### 🎓 Maturity Recommendation")
rec_col_a, rec_col_b = st.columns(2)

with rec_col_a:
    st.info("**Strategy: Buying Options (Long)**")
    st.markdown("""
    *   **Recommendation:** Prefer **Longer Maturity (> 45 Days)**.
    *   **Reason:** Time decay (Theta) accelerates as you get closer to expiration. Buying more time reduces the daily loss in value, giving your trade more time to work.
    """)

with rec_col_b:
    st.warning("**Strategy: Selling Options (Short)**")
    st.markdown("""
    *   **Recommendation:** Prefer **Shorter Maturity (< 30 Days)**.
    *   **Reason:** You *want* the option to lose value quickly. Time decay is your friend here, accelerating rapidly in the final month.
    """)

# Interactive Decay Plot
st.subheader("📉 Time Decay Visualization (Theta)")
days_to_expiry = np.linspace(0.01, 2.0, 50) # From 2 years down to 0
decay_prices = []

for t_val in days_to_expiry:
    decay_prices.append(
        black_scholes_price(current_price, strike_price, t_val, risk_free_rate, volatility, option_type.lower())
    )

fig_decay = go.Figure()
fig_decay.add_trace(go.Scatter(x=days_to_expiry, y=decay_prices, mode='lines', name='Option Price'))
fig_decay.update_layout(
    title=f"Theoretical Price vs. Time to Maturity (Holding Other Factors Constant)",
    xaxis_title="Time to Maturity (Years)",
    yaxis_title="Theoretical Option Price ($)",
    xaxis=dict(autorange="reversed") # Standard convention: Time moves right to left (2 yrs -> 0)
)
st.plotly_chart(fig_decay, use_container_width=True)
