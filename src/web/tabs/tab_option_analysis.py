"""
tab_option_analysis.py

Tab 1 - Single Option Analysis
Handles pricing, path simulation, distribution, heatmap, Greeks, and model comparison
for a single option contract under GBM, Jump Diffusion, or Heston model.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm
from src.core.jump_diffusion import simulate_jump_diffusion, simulate_jump_diffusion_paths
from src.core.heston_model import simulate_heston, simulate_heston_paths
from src.core.lsv_model import simulate_lsv_paths
from src.core.greeks import calculate_all_greeks


def _simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps=60):
    """GBM path generator for visualization. Vectorized log-return expansion."""
    dt = T / n_steps
    Z = np.random.standard_normal((n_paths, n_steps))
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion
    cum = np.zeros((n_paths, n_steps + 1))
    cum[:, 1:] = np.cumsum(log_returns, axis=1)
    return S0 * np.exp(cum)


def render(ticker, asset_data, model_type, n_sims,
           jump_intensity, jump_mean, jump_std,
           heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
           default_spot, default_vol):
    """Renders the full Option Analysis tab."""

    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    with opt_col2:
        current_price = st.number_input("Spot Price (S0)", value=default_spot, min_value=1.0, step=1.0)
        strike_price = st.number_input("Strike Price (K)", value=default_spot, min_value=1.0, step=1.0)
    with opt_col3:
        time_to_maturity = st.number_input("Time to Maturity (T yrs)", min_value=0.01, max_value=5.0, value=1.0, step=0.01, format="%.2f")
        risk_free_rate = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=0.05, step=0.005, format="%.3f")
        volatility = st.number_input("Volatility (sigma)", min_value=0.01, max_value=2.0, value=default_vol, step=0.01, format="%.4f")

    st.markdown("---")

    # --- Pricing ---
    bs_price = black_scholes_price(current_price, strike_price, time_to_maturity,
                                    risk_free_rate, volatility, option_type.lower())

    lsv_surface_status = None

    if model_type == "Jump Diffusion":
        S_T, crash_mask = simulate_jump_diffusion(
            current_price, time_to_maturity, risk_free_rate, volatility, n_sims,
            jump_intensity, jump_mean, jump_std)
        V_T = None
    elif model_type == "Heston (Stochastic Vol)":
        S_T, V_T = simulate_heston(
            current_price, time_to_maturity, risk_free_rate,
            heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho, n_sims)
        crash_mask = np.zeros(n_sims, dtype=bool)
    elif model_type == "LSV (Local Stochastic Vol)":
        lsv_leverage = st.session_state.get("lsv_leverage_matrix")
        lsv_strikes = st.session_state.get("lsv_strikes")
        lsv_mats = st.session_state.get("lsv_maturities")
        if lsv_leverage is None or lsv_strikes is None or lsv_mats is None:
            lsv_surface_status = "Uncalibrated (L=1 fallback)"
            lsv_leverage = np.ones((100, 100))
            lsv_strikes = np.linspace(current_price * 0.5, current_price * 1.5, 100)
            lsv_mats = np.linspace(0.01, max(time_to_maturity, 0.02), 100)
        else:
            lsv_surface_status = "Calibrated"

        paths, _ = simulate_lsv_paths(
            current_price, time_to_maturity, risk_free_rate,
            heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
            lsv_leverage, lsv_strikes, lsv_mats,
            n_paths=n_sims, n_steps=60
        )
        S_T = paths[:, -1]
        crash_mask = np.zeros(n_sims, dtype=bool)
        V_T = None
    else:
        S_T = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
        crash_mask = np.zeros(n_sims, dtype=bool)
        V_T = None

    if option_type == "Call":
        payoffs = np.maximum(S_T - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - S_T, 0)

    mc_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-risk_free_rate * time_to_maturity)

    # --- Metrics Row ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Black-Scholes", f"${bs_price:.4f}")
    model_metric_label = {
        "Standard GBM": "GBM (MC)",
        "Jump Diffusion": "Jump Diffusion (MC)",
        "Heston (Stochastic Vol)": "Heston (MC)",
        "LSV (Local Stochastic Vol)": "LSV (MC)",
    }.get(model_type, f"{model_type} (MC)")
    m2.metric(model_metric_label, f"${mc_price:.4f}", delta=f"{mc_price - bs_price:.4f}", delta_color="inverse")
    m3.metric("Std Error", f"+/-${std_error:.4f}")
    if model_type == "Jump Diffusion":
        cp = np.sum(crash_mask) / n_sims * 100
        m4.metric("Crash Prob", f"{cp:.1f}%")
    elif model_type == "Heston (Stochastic Vol)" and V_T is not None:
        mean_vol = float(np.mean(np.sqrt(np.maximum(V_T, 0)))) * 100
        m4.metric("Terminal Vol", f"{mean_vol:.1f}%")
    elif model_type == "LSV (Local Stochastic Vol)":
        m4.metric("LSV Surface", "Calibrated" if lsv_surface_status == "Calibrated" else "Fallback")
    else:
        m4.metric("Model", "GBM")

    st.markdown("---")

    # --- Path Projection Chart ---
    st.subheader("Asset Price Projections")
    n_plot_paths = 50

    if model_type == "Jump Diffusion":
        sim_paths, path_crash_mask = simulate_jump_diffusion_paths(
            current_price, time_to_maturity, risk_free_rate, volatility,
            n_plot_paths, jump_intensity, jump_mean, jump_std)
    elif model_type == "Heston (Stochastic Vol)":
        sim_paths, _ = simulate_heston_paths(
            current_price, time_to_maturity, risk_free_rate,
            heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho, n_plot_paths)
        path_crash_mask = np.zeros(n_plot_paths, dtype=bool)
    elif model_type == "LSV (Local Stochastic Vol)":
        lsv_leverage = st.session_state.get("lsv_leverage_matrix")
        lsv_strikes = st.session_state.get("lsv_strikes")
        lsv_mats = st.session_state.get("lsv_maturities")
        if lsv_leverage is None or lsv_strikes is None or lsv_mats is None:
            lsv_leverage = np.ones((100, 100))
            lsv_strikes = np.linspace(current_price * 0.5, current_price * 1.5, 100)
            lsv_mats = np.linspace(0.01, max(time_to_maturity, 0.02), 100)
        sim_paths, _ = simulate_lsv_paths(
            current_price, time_to_maturity, risk_free_rate,
            heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
            lsv_leverage, lsv_strikes, lsv_mats,
            n_paths=n_plot_paths, n_steps=60
        )
        path_crash_mask = np.zeros(n_plot_paths, dtype=bool)
    else:
        sim_paths = _simulate_gbm_paths(
            current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths)
        path_crash_mask = np.zeros(n_plot_paths, dtype=bool)

    time_axis = np.linspace(0, time_to_maturity, sim_paths.shape[1])
    fig_paths = go.Figure()

    if asset_data and 'history' in asset_data:
        history = asset_data['history']
        x_hist = np.linspace(-len(history) / 252, 0, len(history))
        fig_paths.add_trace(go.Scatter(x=x_hist, y=history.values, mode='lines',
            name='Historical', line=dict(color='#00FF00', width=2)))

    for i in range(n_plot_paths):
        if model_type in {"Heston (Stochastic Vol)", "LSV (Local Stochastic Vol)"}:
            color = 'rgba(255, 165, 0, 0.2)'
        elif path_crash_mask[i]:
            color = 'rgba(255, 50, 50, 0.3)'
        else:
            color = 'rgba(0, 100, 255, 0.2)'
        fig_paths.add_trace(go.Scatter(x=time_axis, y=sim_paths[i], mode='lines',
            line=dict(width=1, color=color), showlegend=False, hoverinfo='skip'))

    fig_paths.add_hline(y=strike_price, line_dash="dash", line_color="red", annotation_text="Strike")
    fig_paths.update_layout(
        xaxis_title="Years", yaxis_title="Price ($)", height=400,
        hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_paths, width='stretch')

    st.markdown("---")

    # --- Greeks (ALWAYS VISIBLE) ---
    st.subheader("Option Greeks & Sensitivities")
    greeks_params = {}
    if model_type == "Jump Diffusion":
        greeks_params = {'jump_intensity': jump_intensity, 'jump_mean': jump_mean, 'jump_std': jump_std}
        model_name = 'jump_diffusion'
    else:
        model_name = 'gbm'
    greeks = calculate_all_greeks(current_price, strike_price, time_to_maturity,
        risk_free_rate, volatility, option_type.lower(), model=model_name, n_sims=5000, **greeks_params)

    # Calculate Theta (rate of time decay)
    theta = greeks.get('theta', 0.0)
    if 'theta' not in greeks:
        # Approximate theta using finite difference if not provided
        T_plus = max(time_to_maturity - 1/365, 0.001)
        price_tomorrow = black_scholes_price(current_price, strike_price, T_plus, risk_free_rate, volatility, option_type.lower())
        price_today = black_scholes_price(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())
        theta = price_tomorrow - price_today

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Delta (Δ)", f"{greeks['delta']:.4f}", help="Rate of change w.r.t. spot price")
    g2.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}", help="Rate of change of delta; convexity exposure")
    g3.metric("Vega (ν)", f"{greeks['vega']:.4f}", help="Rate of change w.r.t. volatility")
    g4.metric("Theta (Θ)", f"{theta:.4f}", help="Rate of time decay per day")

    st.markdown("---")

    # --- Distribution ---
    st.subheader("Terminal Price Distribution")
    fig_hist = px.histogram(S_T, nbins=50)
    fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="red")
    fig_hist.update_layout(xaxis_title="Price at Maturity", yaxis_title="Freq",
        height=350, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_hist, width='stretch')

    st.markdown("---")

    # --- Advanced: Spot vs Vol Sensitivity (Optional) ---
    with st.expander("Advanced: Price Sensitivity (Spot vs Vol)"):
        st.markdown("*Visualizing how option price responds to changes in spot and volatility*")
        s_range = np.linspace(current_price * 0.8, current_price * 1.2, 10)
        v_range = np.linspace(0.1, 0.6, 10)
        # Vectorized heatmap — outer product across spot and vol grids
        S_grid, V_grid = np.meshgrid(s_range, v_range)
        z_prices = np.vectorize(
            lambda s, v: black_scholes_price(s, strike_price, time_to_maturity,
                                              risk_free_rate, v, option_type.lower())
        )(S_grid, V_grid)
        fig_heat = go.Figure(data=go.Heatmap(z=z_prices, x=s_range, y=v_range, colorscale='Viridis',
            hovertemplate="Spot: $%{x:.2f}<br>Vol: %{y:.2f}<br>Price: $%{z:.2f}<extra></extra>"))
        fig_heat.update_layout(xaxis_title="Spot", yaxis_title="Vol",
            height=350, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_heat, width='stretch')

    # --- Model Comparison ---
    if model_type != "Standard GBM":
        with st.expander("Model Comparison: GBM vs Selected Model"):
            S_T_gbm = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
            pay_gbm = np.maximum(S_T_gbm - strike_price, 0) if option_type == "Call" else np.maximum(strike_price - S_T_gbm, 0)
            mc_gbm = np.exp(-risk_free_rate * time_to_maturity) * np.mean(pay_gbm)
            comp_df = pd.DataFrame({
                "Metric": ["MC Price", "Premium vs GBM", "5th Percentile VaR"],
                "Standard GBM": [f"${mc_gbm:.4f}", "--", f"${np.percentile(S_T_gbm, 5):.2f}"],
                model_type: [f"${mc_price:.4f}", f"${mc_price - mc_gbm:.4f}", f"${np.percentile(S_T, 5):.2f}"]
            })
            st.table(comp_df)
