
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
import os
import yfinance as yf

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm
from src.core.jump_diffusion import simulate_jump_diffusion, simulate_jump_diffusion_paths
from src.core.greeks import calculate_all_greeks
from src.core.data_fetcher import get_options_chain, get_spot_and_vol, get_risk_free_rate
from src.core.scanner_engine import scan_for_arbitrage
from src.core.backtester import run_synthetic_backtest

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
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.title("Monte Carlo Option Pricing Engine")

# ============================================================================
# SIDEBAR (Shared)
# ============================================================================

st.sidebar.header("Asset Selection")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY",
    help="Enter a Yahoo Finance ticker (e.g., AAPL, SPY, MSFT)."
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
    "Simulation Model",
    ["Standard GBM", "Jump Diffusion (Crash Model)"],
    help="GBM: Smooth paths. Jump Diffusion: Includes crash events."
)

if model_type == "Jump Diffusion (Crash Model)":
    st.sidebar.markdown("### Crash Parameters")
    jump_intensity = st.sidebar.slider("Crash Intensity (λ)", 0.0, 1.0, 0.1, 0.05)
    jump_mean = st.sidebar.slider("Average Crash Size (%)", -20, 0, -5, 1) / 100
    jump_std = st.sidebar.slider("Crash Volatility (%)", 1, 10, 3, 1) / 100
else:
    jump_intensity, jump_mean, jump_std = 0.0, 0.0, 0.0

st.sidebar.markdown("---")
n_sims = st.sidebar.slider("Simulations (N)", 1000, 100000, 10000, 1000)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3 = st.tabs(["Single Option Analysis", "Live Arbitrage Scanner", "Historical Backtester"])

# ============================================================================
# TAB 1: SINGLE OPTION ANALYSIS (Original Dashboard)
# ============================================================================
with tab1:
    # --- Tab 1 Sidebar-like controls (in main area) ---
    opt_col1, opt_col2, opt_col3 = st.columns(3)
    with opt_col1:
        option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    with opt_col2:
        current_price = st.number_input("Spot Price (S0)", value=default_spot, min_value=1.0, step=1.0)
        strike_price = st.number_input("Strike Price (K)", value=default_spot, min_value=1.0, step=1.0)
    with opt_col3:
        time_to_maturity = st.slider("Time to Maturity (T Years)", 0.1, 5.0, 1.0, 0.1)
        risk_free_rate = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.05, 0.01)
        volatility = st.slider("Volatility (σ)", 0.05, 1.5, default_vol, 0.05)

    st.markdown("---")

    # --- Helper ---
    def _simulate_gbm_paths_for_plot(S0, T, r, sigma, n_paths, n_steps=60):
        dt = T / n_steps
        Z = np.random.standard_normal((n_paths, n_steps))
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        log_returns = drift + diffusion
        cumulative_log_returns = np.zeros((n_paths, n_steps + 1))
        cumulative_log_returns[:, 1:] = np.cumsum(log_returns, axis=1)
        return S0 * np.exp(cumulative_log_returns)

    # --- Pricing ---
    bs_price = black_scholes_price(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower())

    if model_type == "Jump Diffusion (Crash Model)":
        S_T, crash_mask = simulate_jump_diffusion(current_price, time_to_maturity, risk_free_rate, volatility, n_sims, jump_intensity, jump_mean, jump_std)
    else:
        S_T = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
        crash_mask = np.zeros(n_sims, dtype=bool)

    if option_type == "Call":
        payoffs = np.maximum(S_T - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - S_T, 0)

    mc_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-risk_free_rate * time_to_maturity)

    # --- Metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Black-Scholes", f"${bs_price:.4f}")
    delta_val = mc_price - bs_price
    m2.metric("Monte Carlo", f"${mc_price:.4f}", delta=f"{delta_val:.4f}", delta_color="inverse")
    m3.metric("Standard Error", f"±${std_error:.4f}")
    if model_type == "Jump Diffusion (Crash Model)":
        cp = np.sum(crash_mask) / n_sims * 100
        m4.metric("Crash Prob", f"{cp:.1f}%")
    else:
        m4.metric("Risk Model", "GBM")

    st.markdown("---")

    # --- Projection Chart (Full Width) ---
    st.subheader("Asset Price Projections")
    n_plot_paths = 50
    if model_type == "Jump Diffusion (Crash Model)":
        sim_paths, path_crash_mask = simulate_jump_diffusion_paths(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths, jump_intensity, jump_mean, jump_std)
    else:
        sim_paths = _simulate_gbm_paths_for_plot(current_price, time_to_maturity, risk_free_rate, volatility, n_plot_paths)
        path_crash_mask = np.zeros(n_plot_paths, dtype=bool)

    time_axis = np.linspace(0, time_to_maturity, sim_paths.shape[1])
    fig_paths = go.Figure()

    if asset_data and 'history' in asset_data:
        history = asset_data['history']
        x_hist = np.linspace(-len(history)/252, 0, len(history))
        fig_paths.add_trace(go.Scatter(x=x_hist, y=history.values, mode='lines', name='Historical', line=dict(color='green', width=2)))

    for i in range(n_plot_paths):
        color = 'rgba(255, 50, 50, 0.3)' if path_crash_mask[i] else 'rgba(0, 100, 255, 0.2)'
        fig_paths.add_trace(go.Scatter(x=time_axis, y=sim_paths[i], mode='lines', line=dict(width=1, color=color), showlegend=False, hoverinfo='skip'))

    fig_paths.add_hline(y=strike_price, line_dash="dash", line_color="red", annotation_text="Strike")
    fig_paths.update_layout(xaxis_title="Years from Today", yaxis_title="Price ($)", height=420, hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_paths, use_container_width=True)

    # --- Distribution + Heatmap ---
    st.markdown("---")
    dist_col, heat_col = st.columns(2)

    with dist_col:
        st.subheader("Terminal Price Distribution")
        fig_hist = px.histogram(S_T, nbins=50)
        fig_hist.add_vline(x=strike_price, line_dash="dash", line_color="red")
        fig_hist.update_layout(xaxis_title="Price at Maturity", yaxis_title="Frequency", height=380, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_hist, use_container_width=True)

    with heat_col:
        st.subheader("Price Sensitivity (Spot vs Vol)")
        s_range = np.linspace(current_price * 0.8, current_price * 1.2, 10)
        v_range = np.linspace(0.1, 0.6, 10)
        z_prices = np.zeros((10, 10))
        for i, s_tmp in enumerate(s_range):
            for j, v_tmp in enumerate(v_range):
                z_prices[j, i] = black_scholes_price(s_tmp, strike_price, time_to_maturity, risk_free_rate, v_tmp, option_type.lower())
        fig_heat = go.Figure(data=go.Heatmap(z=z_prices, x=s_range, y=v_range, colorscale='Viridis',
            hovertemplate="Spot: $%{x:.2f}<br>Vol: %{y:.2f}<br>Price: $%{z:.2f}<extra></extra>"))
        fig_heat.update_layout(xaxis_title="Spot Price", yaxis_title="Volatility", height=380, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Greeks ---
    st.markdown("---")
    with st.expander("Greeks & Sensitivities"):
        greeks_params = {'jump_intensity': jump_intensity, 'jump_mean': jump_mean, 'jump_std': jump_std} if model_type == "Jump Diffusion (Crash Model)" else {}
        model_name = 'jump_diffusion' if model_type == "Jump Diffusion (Crash Model)" else 'gbm'
        greeks = calculate_all_greeks(current_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type.lower(), model=model_name, n_sims=5000, **greeks_params)
        g1, g2, g3 = st.columns(3)
        g1.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
        g2.metric("Vega (ν)", f"{greeks['vega']:.4f}")
        g3.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}")

    # --- Model Comparison (Jump Diffusion only) ---
    if model_type == "Jump Diffusion (Crash Model)":
        with st.expander("Model Comparison: GBM vs Jump Diffusion"):
            S_T_gbm = simulate_gbm(current_price, time_to_maturity, risk_free_rate, volatility, n_sims)
            pay_gbm = np.maximum(S_T_gbm - strike_price, 0) if option_type == "Call" else np.maximum(strike_price - S_T_gbm, 0)
            mc_gbm = np.exp(-risk_free_rate * time_to_maturity) * np.mean(pay_gbm)
            comp_df = pd.DataFrame({
                "Metric": ["MC Price", "Crash Premium", "95% VaR"],
                "Standard GBM": [f"${mc_gbm:.4f}", "-", f"${np.percentile(S_T_gbm, 5):.2f}"],
                "Jump Diffusion": [f"${mc_price:.4f}", f"${mc_price - mc_gbm:.4f}", f"${np.percentile(S_T, 5):.2f}"]
            })
            st.table(comp_df)

# ============================================================================
# TAB 2: LIVE ARBITRAGE SCANNER
# ============================================================================
with tab2:
    st.subheader(f"Live Options Scanner for {ticker}")
    st.markdown("Scan the live options chain to find mispriced options using your Monte Carlo engine.")

    # --- Scanner Controls ---
    scan_col1, scan_col2, scan_col3 = st.columns([2, 1, 1])
    with scan_col1:
        scanner_sims = st.select_slider(
            "MC Simulations per Option",
            options=[1000, 5000, 10000, 25000, 50000],
            value=5000,
            help="Higher = more accurate but slower. 5000 is a good balance."
        )
    with scan_col2:
        scan_mode = st.selectbox("Scan Focus", ["Next 2 Expirations", "Next 5 Expirations", "1-Month (Monthly)"], index=0)
    with scan_col3:
        st.markdown("")  # spacing
        scan_button = st.button("Scan Market", type="primary", use_container_width=True)

    st.markdown("---")

    if scan_button:
        with st.spinner(f"Fetching live options chain for {ticker}..."):
            # Fetch live data
            spot_data = get_spot_and_vol(ticker)
            r_live = get_risk_free_rate()
            # Determine fetching parameters
            if scan_mode == "1-Month (Monthly)":
                options_df = get_options_chain(ticker, target_days=30)
            elif scan_mode == "Next 5 Expirations":
                options_df = get_options_chain(ticker, max_expirations=5)
            else:
                options_df = get_options_chain(ticker, max_expirations=2)

        if spot_data is None:
            st.error(f"Could not fetch spot data for {ticker}. Check the ticker symbol.")
        elif options_df.empty:
            st.error(f"No options data found for {ticker}. This ticker may not have options listed.")
        else:
            S0 = spot_data['spot']
            hist_vol = spot_data['historical_vol']

            # Display market context
            ctx1, ctx2, ctx3, ctx4 = st.columns(4)
            ctx1.metric("Spot Price", f"${S0:.2f}")
            ctx2.metric("Historical Vol", f"{hist_vol*100:.1f}%")
            ctx3.metric("Risk-Free Rate", f"{r_live*100:.2f}%")
            ctx4.metric("Options Found", f"{len(options_df)}")

            st.markdown("---")

            # Run the scanner
            with st.spinner(f"Running Monte Carlo on {len(options_df)} contracts ({scanner_sims} sims each)..."):
                results_df = scan_for_arbitrage(
                    options_df, S0, r_live, hist_vol,
                    n_sims=scanner_sims,
                    jump_intensity=jump_intensity,
                    jump_mean=jump_mean,
                    jump_std=jump_std
                )

            if results_df.empty:
                st.warning("No viable options found after filtering.")
            else:
                # Summary metrics
                buys = len(results_df[results_df['signal'].str.contains('BUY')])
                sells = len(results_df[results_df['signal'].str.contains('SELL')])
                holds = len(results_df[results_df['signal'].str.contains('HOLD')])

                sig1, sig2, sig3, sig4 = st.columns(4)
                sig1.metric("Total Scanned", len(results_df))
                sig2.metric("🟢 BUY Signals", buys)
                sig3.metric("🔴 SELL Signals", sells)
                sig4.metric("⚪ HOLD", holds)

                st.markdown("---")

                # Color-coded results table
                st.subheader("Arbitrage Opportunities (Sorted by Edge)")

                # Prepare display DataFrame
                display_df = results_df.rename(columns={
                    'type': 'Type',
                    'strike': 'Strike',
                    'expiration': 'Expires',
                    'T_days': 'Days',
                    'bid': 'Bid',
                    'ask': 'Ask',
                    'mid': 'Market Mid',
                    'mc_price': 'MC Fair Value',
                    'bs_price': 'BS Price',
                    'edge': 'Edge ($)',
                    'edge_pct': 'Edge (%)',
                    'signal': 'Signal',
                    'volume': 'Volume',
                    'market_iv': 'Market IV (%)',
                })

                # Format columns
                for col in ['Strike', 'Bid', 'Ask', 'Market Mid', 'MC Fair Value', 'BS Price', 'Edge ($)']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")

                display_df['Edge (%)'] = display_df['Edge (%)'].apply(lambda x: f"{x:+.1f}%")
                display_df['Market IV (%)'] = display_df['Market IV (%)'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500,
                    hide_index=True
                )

                # Top opportunities chart
                st.markdown("---")
                st.subheader("Top 10 Opportunities")

                top10 = results_df.head(10).copy()
                top10['label'] = top10['type'] + ' ' + top10['strike'].astype(str) + ' (' + top10['expiration'] + ')'

                fig_bar = go.Figure()
                colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in top10['edge_pct']]
                fig_bar.add_trace(go.Bar(
                    x=top10['label'],
                    y=top10['edge_pct'],
                    marker_color=colors,
                    text=top10['signal'],
                    textposition='outside'
                ))
                fig_bar.update_layout(
                    xaxis_title="Option Contract",
                    yaxis_title="Edge (%)",
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    else:
        # Default state before scanning
        st.info("Click **Scan Market** to analyze the live options chain and find mispriced options using your Monte Carlo engine.")

        st.markdown("""
        **How the Scanner Works:**
        1. Fetches all live option contracts for the selected ticker from Yahoo Finance.
        2. Runs your Jump Diffusion Monte Carlo simulation on each contract.
        3. Compares the MC "Fair Value" against the market's Bid/Ask midpoint.
        4. Flags options where the model price differs significantly from the market price.

        | Signal | Meaning |
        |--------|---------|
        | 🟢 BUY | Model says the option is worth MORE than the market is charging (undervalued) |
        | 🔴 SELL | Model says the option is worth LESS than the market is charging (overvalued) |
        | ⚪ HOLD | Model and market roughly agree (no edge) |
        """)

st.success("Dashboard Ready")

# ============================================================================
# TAB 3: HISTORICAL BACKTESTER
# ============================================================================
with tab3:
    st.subheader(f"Historical Backtester for {ticker}")
    st.markdown("Test how your Jump Diffusion model would have performed trading real historical data.")

    # --- Backtest Controls ---
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    with bt_col1:
        bt_capital = st.number_input("Starting Capital ($)", value=10000.0, min_value=1000.0, step=1000.0)
    with bt_col2:
        bt_period = st.selectbox("Lookback Period", ["1y", "2y", "3y", "5y"], index=1)
    with bt_col3:
        bt_edge = st.slider("Edge Threshold (%)", 1, 30, 10, 1)
    with bt_col4:
        st.markdown("")  # spacing
        bt_button = st.button("Run Backtest", type="primary", use_container_width=True)

    st.markdown("---")

    if bt_button:
        with st.spinner(f"Time-traveling through {bt_period} of {ticker} history..."):
            results = run_synthetic_backtest(
                ticker=ticker,
                period=bt_period,
                initial_capital=bt_capital,
                option_type='call',
                edge_threshold=bt_edge / 100.0,
                risk_free_rate=0.05,
                n_sims=n_sims,
                jump_intensity=jump_intensity,
                jump_mean=jump_mean,
                jump_std=jump_std,
                expiry_days=30
            )

        if results is None:
            st.error(f"Could not fetch enough historical data for {ticker}. Try a shorter period.")
        else:
            # --- Key Metrics ---
            km1, km2, km3, km4 = st.columns(4)
            pnl = results['final_value'] - bt_capital
            km1.metric("Final Portfolio", f"${results['final_value']:,.2f}",
                       delta=f"${pnl:+,.2f}", delta_color="normal")
            km2.metric("Total Return", f"{results['total_return_pct']:+.1f}%")
            km3.metric("Win Rate", f"{results['win_rate']:.0f}%")
            km4.metric("Total Trades", f"{results['total_trades']}")

            st.markdown("---")

            # --- Equity Curve ---
            st.subheader("Portfolio Equity Curve")
            equity_df = results['equity_curve']
            if not equity_df.empty and len(equity_df) > 1:
                fig_equity = go.Figure()
                fig_equity.add_trace(go.Scatter(
                    x=equity_df['date'], y=equity_df['value'],
                    mode='lines+markers',
                    name='Portfolio Value',
                    line=dict(color='#00d4aa', width=3),
                    marker=dict(size=6)
                ))
                fig_equity.add_hline(
                    y=bt_capital, line_dash="dash", line_color="gray",
                    annotation_text=f"Starting Capital: ${bt_capital:,.0f}"
                )
                fig_equity.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=420,
                    hovermode="x unified",
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                st.plotly_chart(fig_equity, use_container_width=True)
            else:
                st.warning("Not enough trades to generate an equity curve.")

            # --- Trade Log ---
            st.markdown("---")
            st.subheader("Trade Log")
            trades_df = results['trades_df']
            if not trades_df.empty:
                def _color_pnl(val):
                    if isinstance(val, str):
                        color = 'color: #00d4aa' if val == 'WIN' else 'color: #ff4b4b'
                        return color
                    return ''

                styled_df = trades_df.style.applymap(
                    _color_pnl, subset=['result']
                )
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            else:
                st.info("No trades were triggered during this period. Try lowering the Edge Threshold.")

    else:
        st.info("Click **Run Backtest** to simulate trading your model over historical data.")
        st.markdown("""
        **How the Backtester Works:**
        1. Downloads historical daily prices for the selected ticker.
        2. On the 1st trading day of every month, it synthesizes an ATM (At-The-Money) call option.
        3. It compares the Black-Scholes "market" price against your Jump Diffusion Monte Carlo "fair value".
        4. If the MC fair value exceeds the BS price by more than the Edge Threshold, it **buys** the option.
        5. It then fast-forwards 30 days and settles the trade using the *actual* stock price on that date.
        6. The result is a full profit/loss history showing how your model would have performed.
        """)
