"""
tab_portfolio_risk.py

Tab 4 - Portfolio Risk & 3D Greek Surfaces
Renders vectorized analytical Greek surfaces (Gamma, Vanna, Vega) over a
(Spot x Volatility) meshgrid using exact Black-Scholes closed-form derivatives.

Mathematical formulas:
    d1 = [ln(S/K) + (r + 0.5*sigma^2)*T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    phi(d1) = standard normal PDF evaluated at d1

    Gamma = phi(d1) / (S * sigma * sqrt(T))
    Vanna = -phi(d1) * d2 / sigma           [exact, no finite diff]
    Vega  = S * phi(d1) * sqrt(T)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from src.core.data_fetcher import get_options_chain


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_cached_options_chain(ticker, max_expirations=10):
    """Cached wrapper for options chain fetching with 5-minute TTL."""
    return get_options_chain(ticker, max_expirations=max_expirations)


def render(ticker, default_spot):
    """
    Renders the Portfolio Risk tab with dual modes:
    1. Theoretical Risk Surfaces (Spot x Vol x Greek)
    2. Market IV Surface (Strike x Time x IV)
    """

    st.subheader(f"3D Risk & Volatility Analytics: {ticker}")
    
    view_mode = st.radio("Analytics View", ["Theoretical Risk (Greeks)", "Market IV Surface"], horizontal=True)

    if view_mode == "Theoretical Risk (Greeks)":
        st.markdown("Visualizing option sensitivities across Spot and Volatility levels.")

        risk_col1, risk_col2, risk_col3 = st.columns(3)
        with risk_col1:
            target_greek = st.selectbox(
                "Greek Surface",
                ["Gamma (dDelta/dSpot)", "Vanna (dDelta/dVol)", "Vega (dPrice/dVol)"]
            )
        with risk_col2:
            risk_K = st.number_input("Target Strike (K)", value=default_spot, min_value=1.0)
        with risk_col3:
            risk_T = st.number_input("Time to Expiry (T)", value=0.5, min_value=0.01)

        st.markdown("---")

        # --- Vectorized Greek Computation (single NumPy pass) ---
        s_vals = np.linspace(default_spot * 0.7, default_spot * 1.3, 50)
        v_vals = np.linspace(0.05, 1.0, 50)
        X, Y = np.meshgrid(s_vals, v_vals)   # X = spot grid, Y = vol grid

        _sqrt_T = np.sqrt(risk_T)
        _d1 = (np.log(X / risk_K) + (0.05 + 0.5 * Y**2) * risk_T) / (Y * _sqrt_T)
        _d2 = _d1 - Y * _sqrt_T
        _phi_d1 = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * _d1**2)

        if "Gamma" in target_greek:
            Z = _phi_d1 / (X * Y * _sqrt_T)
            colorscale = 'Plasma'
            z_label = "Gamma"
        elif "Vanna" in target_greek:
            Z = -_phi_d1 * _d2 / Y
            colorscale = 'RdBu'
            z_label = "Vanna"
        else:
            Z = X * _phi_d1 * _sqrt_T
            colorscale = 'Viridis'
            z_label = "Vega"

        fig_risk = go.Figure(data=[go.Surface(
            z=Z, x=X, y=Y,
            colorscale=colorscale,
            hovertemplate=(
                "Spot: $%{x:.2f}<br>"
                "Vol: %{y:.2f}<br>"
                f"{z_label}: %{{z:.6f}}<extra></extra>"
            )
        )])

        fig_risk.update_layout(
            scene=dict(
                xaxis_title="Spot Price ($)",
                yaxis_title="Implied Volatility (sigma)",
                zaxis_title=z_label,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=700,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_risk, width='stretch')
        st.caption(
            f"Surface computed analytically over a 50x50 grid "
            f"(K={risk_K:.2f}, T={risk_T:.2f}y, r=5%). "
            f"Formula: exact Black-Scholes closed-form derivative — zero loops."
        )

    else:
        # --- Market IV Surface Mode ---
        st.markdown("Visualizing the live Market Implied Volatility surface across Strike and Maturity.")

        # Add refresh control
        refresh_col1, refresh_col2 = st.columns([3, 1])
        with refresh_col2:
            if st.button("Refresh Data", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        with st.spinner(f"Fetching market data for {ticker}..."):
            # Use cached fetch with 5-minute TTL
            options_df = fetch_cached_options_chain(ticker, max_expirations=10)

        if options_df.empty:
            st.error(f"Could not fetch options chain for {ticker}.")
            return

        # Filter for better visualization (calls only, reasonably near-the-money)
        plot_df = options_df[options_df['type'] == 'call'].copy()
        plot_df = plot_df[(plot_df['strike'] > default_spot * 0.6) & (plot_df['strike'] < default_spot * 1.4)]

        if len(plot_df) < 15:
            st.warning("Not enough liquid call options to build a reliable surface. Showing all available data.")
            plot_df = options_df.copy()

        # Extract axes
        strikes = plot_df['strike'].values
        maturities = (plot_df['T'] * 365).values # Days to maturity
        ivs = plot_df['market_iv'].values * 100 # In percent

        # Create grid for interpolation
        strike_grid = np.linspace(strikes.min(), strikes.max(), 50)
        time_grid = np.linspace(maturities.min(), maturities.max(), 50)
        X, Y = np.meshgrid(strike_grid, time_grid)

        # Interpolate scattered market data onto the grid with fallback strategy
        Z = None
        for method in ['cubic', 'linear', 'nearest']:
            try:
                Z = griddata((strikes, maturities), ivs, (X, Y), method=method)
                # Fill NaNs from interpolation at edges with nearest neighbor if using cubic/linear
                if method != 'nearest':
                    Z_nearest = griddata((strikes, maturities), ivs, (X, Y), method='nearest')
                    Z = np.where(np.isnan(Z), Z_nearest, Z)
                break  # Success, exit fallback loop
            except Exception as e:
                if method == 'nearest':
                    # Last resort failed; show error to user
                    st.error(f"Could not interpolate IV surface: {str(e)[:100]}. Showing raw data only.")
                    # Use raw scatter plot instead
                    fig_iv = go.Figure(data=[go.Scatter3d(
                        x=strikes, y=maturities, z=ivs,
                        mode='markers',
                        marker=dict(size=4, color=ivs, colorscale='Viridis', showscale=True),
                        hovertemplate="Strike: $%{x:.2f}<br>Days: %{y:.1f}<br>IV: %{z:.1f}%<extra></extra>"
                    )])
                    fig_iv.update_layout(
                        scene=dict(
                            xaxis_title="Strike Price ($)",
                            yaxis_title="Time to Maturity (Days)",
                            zaxis_title="Implied Volatility (%)"
                        ),
                        height=700,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_iv, width='stretch')

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Data Points", len(plot_df))
                    c2.metric("Median IV", f"{np.median(ivs):.1f}%")
                    c3.metric("Strike Range", f"${strikes.min():.0f} - ${strikes.max():.0f}")
                    return
                continue

        if Z is not None:
            fig_iv = go.Figure(data=[go.Surface(
                z=Z, x=X, y=Y,
                colorscale='Viridis',
                hovertemplate=(
                    "Strike: $%{x:.2f}<br>"
                    "Days: %{y:.1f}d<br>"
                    "IV: %{z:.1f}%<extra></extra>"
                )
            )])

            fig_iv.update_layout(
                scene=dict(
                    xaxis_title="Strike Price ($)",
                    yaxis_title="Time to Maturity (Days)",
                    zaxis_title="Implied Volatility (%)",
                    camera=dict(eye=dict(x=1.8, y=1.2, z=1.2))
                ),
                height=700,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_iv, width='stretch')

            c1, c2, c3 = st.columns(3)
            c1.metric("Data Points", len(plot_df))
            c2.metric("Median IV", f"{np.median(ivs):.1f}%")
            c3.metric("Strike Range", f"${strikes.min():.0f} - ${strikes.max():.0f}")

            st.info("This surface is derived from live market prices. The 'Smile' is visible across Strikes, while 'Term Structure' is visible across Time. Data is cached for 5 minutes — use the Refresh Data button for the latest prices.")

