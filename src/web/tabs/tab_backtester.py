"""
tab_backtester.py

Tab 3 - Research Backtester
Runs a controlled historical simulation using model fair
values against historical option quotes. Outputs equity curve, trade
log, and cost diagnostics with explicit methodology disclosure.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.core.backtester import (
    FULL_HISTORICAL_PERIOD,
    get_historical_option_quote_range,
    has_historical_option_quotes,
    run_historical_quotes_backtest,
)


def render(ticker, option_type, n_sims,
           jump_intensity, jump_mean, jump_std,
           heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho):
    """Renders the full research backtester tab."""

    st.subheader(f"Backtester: {ticker}")

    # --- Methodology Info (BEFORE controls) ---
    with st.expander("How the Backtester Works", expanded=False):
        st.markdown("""
        **Controlled Research Backtest:**

        1. **Daily Delta-Hedging**: Rebalances the underlying daily to isolate volatility-driven PnL from outright directional exposure.
        2. **No Look-Ahead Bias**: Rolling volatility is computed strictly from observations before each trade date.
        3. **Data Source**:
           - **Historical SPX Quotes (CSV)** uses the pasted `combined_options_data.csv` bid/ask snapshots.
        4. **Interpretation**: Returns and Sharpe remain research metrics, especially because hedging is still simulated from daily closes.

        **Key Parameters**:
        - **Edge Threshold**: Minimum % difference between model fair value and market price to trigger trade entry
        - **Historical CSV Mode**: Scans the full pasted SPX options file
        - **Model**: Choose between Heston, Jump Diffusion, or LSV fair-value engines
        """)

    st.markdown("---")

    historical_available = has_historical_option_quotes()
    supported_historical_ticker = str(ticker).upper() in {"SPX", "^SPX"}
    csv_start, csv_end = (None, None)
    if historical_available:
        csv_start, csv_end = get_historical_option_quote_range()

    if historical_available and supported_historical_ticker and csv_start and csv_end:
        st.caption(
            f"Using the full local CSV range: {csv_start.strftime('%Y-%m-%d')} to {csv_end.strftime('%Y-%m-%d')}. "
            "This local file does not currently extend to 2023."
        )
    elif not historical_available:
        st.error("Historical quote mode is unavailable until `combined_options_data.csv` is present in the project root.")
        return
    elif not supported_historical_ticker:
        st.error("Historical quote mode currently supports SPX only (`SPX` / `^SPX`).")
        return

    # --- Backtester Controls ---
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns([1, 1, 1.2, 1.2])
    with bt_col1:
        bt_capital = st.number_input("Capital ($)", value=10000.0, min_value=1000.0, step=1000.0)
    with bt_col2:
        st.text_input("Period", value="Full CSV", disabled=True)
        bt_period = FULL_HISTORICAL_PERIOD
    with bt_col3:
        bt_edge = st.number_input("Edge Threshold (%)", min_value=1, max_value=30, value=10, step=1)
    with bt_col4:
        bt_model = st.selectbox("Model", ["jump_diffusion", "heston", "lsv"])
    st.text_input("Expiration Selection", value="Automatic scan across the full CSV", disabled=True)
    bt_exps = None

    bt_button = st.button("RUN BACKTEST", type="primary", width="stretch")

    st.markdown("---")

    if bt_button:
        spinner_label = f"Backtesting {bt_model} on the full CSV for {ticker}..."
        with st.spinner(spinner_label):
            try:
                lsv_leverage = st.session_state.get("lsv_leverage_matrix")
                lsv_strikes = st.session_state.get("lsv_strikes")
                lsv_maturities = st.session_state.get("lsv_maturities")
                results = run_historical_quotes_backtest(
                    ticker=ticker, period=bt_period, initial_capital=bt_capital,
                    option_type=option_type.lower(), edge_threshold=bt_edge / 100.0,
                    risk_free_rate=0.05, n_sims=n_sims, model=bt_model,
                    jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
                    heston_V0=heston_V0, heston_kappa=heston_kappa,
                    heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho,
                    leverage_matrix=lsv_leverage,
                    leverage_strikes=lsv_strikes,
                    leverage_maturities=lsv_maturities,
                    expiry_days_list=bt_exps
                )
            except Exception as e:
                st.error(f"Backtester Engine Failure: {str(e)}")
                results = None

        if results is None:
            st.error(f"Could not fetch enough data for {ticker}.")
            return

        methodology = results.get('methodology', {})
        disclosure = methodology.get('disclosure')
        if methodology.get('uses_historical_option_quotes', False):
            st.info(disclosure)
            quote_filters = methodology.get("quote_filters", {})
            if quote_filters:
                st.caption(
                    f"Quote filters: bid >= {quote_filters.get('min_bid', 'n/a')}, "
                    f"spread <= {quote_filters.get('max_spread_pct', 0) * 100:.0f}% of mid, "
                    f"positive IV required."
                )

        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Fair Value Model", methodology.get('fair_value_model', bt_model).upper())
        entry_source_label = methodology.get('entry_market_price_source', 'unknown')
        if entry_source_label == "black_scholes_proxy_from_rolling_vol":
            entry_source_label = "BS Proxy"
        elif entry_source_label == "historical_option_bid_ask_mid":
            entry_source_label = "Hist. Bid/Ask"
        meta2.metric("Entry Price Source", entry_source_label)
        meta3.metric("Look-Ahead Guard", "ON" if methodology.get('lookahead_guard') else "OFF")
        if methodology.get('data_start') and methodology.get('data_end'):
            st.caption(f"Historical quote window: {methodology['data_start']} to {methodology['data_end']}")
        if methodology.get('solvency_breach_triggered'):
            st.error("Solvency guard triggered during the backtest. Open positions were force-liquidated and Sharpe is suppressed.")

        km1, km2, km3, km4, km5, km6, km7 = st.columns(7)
        pnl = results['final_value'] - bt_capital
        sharpe_value = results.get('sharpe_ratio')
        km1.metric("Final Value", f"${results['final_value']:,.2f}")
        km2.metric("Net P&L", f"${pnl:+,.2f}")
        km3.metric("Return", f"{results['total_return_pct']:+.1f}%")
        km4.metric("Win Rate", f"{results['win_rate']:.0f}%")
        km5.metric("Trades", f"{results['total_trades']}")
        km6.metric("Sharpe", "N/A" if pd.isna(sharpe_value) else f"{sharpe_value:.2f}")
        km7.metric("Max DD", f"{results['max_drawdown_pct']:.1f}%")

        st.markdown("---")
        st.subheader("Equity Curve")
        equity_df = results['equity_curve']
        if not equity_df.empty and len(equity_df) > 1:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity_df['date'], y=equity_df['net_value'],
                mode='lines+markers', name='Net Portfolio',
                line=dict(color='#00FF00', width=2), marker=dict(size=4)))
            fig_eq.add_hline(y=bt_capital, line_dash="dash", line_color="gray",
                annotation_text=f"Start: ${bt_capital:,.0f}")
            fig_eq.update_layout(xaxis_title="Date", yaxis_title="Value ($)",
                height=380, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_eq, width='stretch')

        st.markdown("---")
        st.subheader("Trade Log")
        trades_df = results['trades_df']
        if not trades_df.empty:
            # Compact view: essential columns only
            price_col = 'market_mid' if 'market_mid' in trades_df.columns else 'market_proxy_price'
            exit_col = 'exit_date' if 'exit_date' in trades_df.columns else 'expiry_date'
            compact_cols = ['entry_date', exit_col, 'contracts', 'strike', 'mc_fair_value', price_col, 'pnl_net', 'result']
            compact_df = trades_df[[col for col in compact_cols if col in trades_df.columns]].copy()
            compact_df = compact_df.rename(columns={
                'entry_date': 'Entry', 'exit_date': 'Exit', 'expiry_date': 'Exit',
                'contracts': 'Contracts',
                'strike': 'Strike', 'mc_fair_value': 'Fair Value', 'market_proxy_price': 'Mkt Proxy',
                'market_mid': 'Mkt Mid',
                'pnl_net': 'P&L ($)', 'result': 'Result'
            })

            def _color_result(val):
                if val == 'WIN':
                    return 'color: #00FF00; font-weight: bold'
                elif val == 'LOSS':
                    return 'color: #FF4444; font-weight: bold'
                return ''

            st.subheader("Compact View", divider=False, anchor=None)
            styled_compact = compact_df.style.map(_color_result, subset=['Result'])
            st.dataframe(styled_compact, width='stretch', hide_index=True, height=300)

            # --- Export trade log ---
            export_col1, export_col2, export_col3 = st.columns([2, 1, 1])
            with export_col1:
                st.markdown("**Export Trade Log:**")
            with export_col2:
                csv_data = trades_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"backtest_trades_{ticker}.csv",
                    mime="text/csv",
                    width="stretch"
                )
            with export_col3:
                json_data = trades_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"backtest_trades_{ticker}.json",
                    mime="application/json",
                    width="stretch"
                )

            # Detailed forensic view
            with st.expander("Detailed Forensic Analysis"):
                st.markdown("*Complete trade record with costs, volatility, and sensitivity analysis*")
                styled_trades = trades_df.style.map(_color_result, subset=['result'])
                st.dataframe(styled_trades, width='stretch', hide_index=True)
        else:
            st.info("No trades triggered. With SPX quote mode, the 100x contract multiplier, quote filters, and hedge reserve can make a small account ineligible even when edge exists.")

        st.markdown("---")
        st.subheader("Cost Sensitivity Analysis")

        # Cost sensitivity: show how returns vary with different transaction cost levels
        if 'cost_sensitivity' in results and results['cost_sensitivity']:
            cost_sens = results['cost_sensitivity']

            # Build sensitivity table
            cost_scenarios = []
            for scenario_key in sorted(cost_sens.keys()):
                scenario_data = cost_sens[scenario_key]
                # Extract multiplier from key (e.g., "cost_x0.5" -> "0.5x")
                mult = scenario_key.split('_')[1]  # e.g., "1.0x"
                cost_scenarios.append({
                    'Cost Level': mult,
                    'Final Value': f"${scenario_data['final_value']:,.2f}",
                    'Return': f"{scenario_data['return_pct']:+.1f}%"
                })

            sensitivity_df = pd.DataFrame(cost_scenarios)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("**50% Costs**")
                if cost_scenarios:
                    st.metric("Final Value", cost_scenarios[0]['Final Value'])
                    st.metric("Return", cost_scenarios[0]['Return'])
            with col2:
                st.markdown("**100% Costs (Baseline)**")
                if len(cost_scenarios) > 1:
                    st.metric("Final Value", cost_scenarios[1]['Final Value'])
                    st.metric("Return", cost_scenarios[1]['Return'])
            with col3:
                st.markdown("**150% Costs**")
                if len(cost_scenarios) > 2:
                    st.metric("Final Value", cost_scenarios[2]['Final Value'])
                    st.metric("Return", cost_scenarios[2]['Return'])

            st.caption("Scenario analysis showing strategy robustness to transaction cost variations")

        # Cost breakdown information
        if 'cost_summary' in results:
            with st.expander("Cost Breakdown & Details"):
                cost_summary = results['cost_summary']
                cb1, cb2, cb3 = st.columns(3)
                with cb1:
                    st.metric("Entry Costs", f"${cost_summary['total_entry_costs']:,.2f}")
                with cb2:
                    st.metric("Hedge Costs", f"${cost_summary['total_hedge_costs']:,.2f}")
                with cb3:
                    st.metric("Total Impact", f"${cost_summary['total_cost_impact']:,.2f}")

                st.write(f"**Avg Entry Edge**: {cost_summary['avg_entry_edge_pct']:.2f}%")
                st.write(f"**Rebalance Threshold**: {cost_summary['hedge_rebalance_delta_threshold']:.4f}")
                if 'option_multiplier' in cost_summary:
                    st.write(f"**Contract Multiplier**: {cost_summary['option_multiplier']}x")
                if 'hedge_margin_ratio' in cost_summary:
                    st.write(f"**Hedge Reserve**: {cost_summary['hedge_margin_ratio'] * 100:.0f}% of initial hedge notional")
                if 'max_open_positions' in cost_summary:
                    st.write(f"**Max Concurrent Positions**: {cost_summary['max_open_positions']}")
                if cost_summary.get('forced_liquidations', 0):
                    st.write(f"**Forced Liquidations**: {cost_summary['forced_liquidations']}")
