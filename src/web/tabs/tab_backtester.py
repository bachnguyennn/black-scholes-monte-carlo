"""
tab_backtester.py

Tab 3 - Research Backtester
Runs a controlled historical simulation using Jump Diffusion or Heston fair
values against a synthetic market-price proxy. Outputs equity curve, trade
log, and cost diagnostics with explicit methodology disclosure.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from src.core.backtester import run_synthetic_backtest


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
        3. **Proxy Market Entry**: The model fair value comes from Heston or Jump Diffusion, but entry uses a Black-Scholes proxy market price because historical option quotes are not replayed.
        4. **Interpretation**: Returns and Sharpe should be treated as controlled research evidence, not proof of executable historical fills.

        **Key Parameters**:
        - **Edge Threshold**: Minimum % difference between model fair value and market price to trigger trade entry
        - **Term Structure**: Target option expirations (Days To Expiry) to scan for candidate valuation gaps
        - **Model**: Choose between Heston (stochastic volatility) or Jump Diffusion (crash events)
        """)

    st.markdown("---")

    # --- Backtester Controls ---
    bt_col1, bt_col2, bt_col3, bt_col4, bt_col5, bt_col6 = st.columns([1, 1, 1.2, 1, 1.5, 1.2])
    with bt_col1:
        bt_capital = st.number_input("Capital ($)", value=10000.0, min_value=1000.0, step=1000.0)
    with bt_col2:
        bt_period = st.selectbox("Period", ["1y", "2y", "3y", "5y"], index=1)
    with bt_col3:
        bt_edge = st.number_input("Edge Threshold (%)", min_value=1, max_value=30, value=10, step=1)
    with bt_col4:
        bt_model = st.selectbox("Model", ["jump_diffusion", "heston"])
    with bt_col5:
        bt_exps = st.multiselect("Term Structure", [7, 14, 21, 30, 60, 90, 120, 150, 180], default=[30, 60, 90])
    with bt_col6:
        st.markdown("")
        bt_button = st.button("RUN BACKTEST", type="primary", use_container_width=True)

    st.markdown("---")

    if bt_button:
        with st.spinner(f"Backtesting {bt_model} over {bt_period} of {ticker}..."):
            try:
                results = run_synthetic_backtest(
                    ticker=ticker, period=bt_period, initial_capital=bt_capital,
                    option_type=option_type.lower(), edge_threshold=bt_edge / 100.0,
                    risk_free_rate=0.05, n_sims=n_sims, model=bt_model,
                    jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
                    heston_V0=heston_V0, heston_kappa=heston_kappa,
                    heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho,
                    expiry_days_list=bt_exps
                )
            except Exception as e:
                st.error(f"Backtester Engine Failure: {str(e)}")
                results = None

        if results is None:
            st.error(f"Could not fetch enough data for {ticker}.")
            return

        methodology = results.get('methodology', {})
        if not methodology.get('uses_historical_option_quotes', True):
            st.warning(methodology.get('disclosure', 'This backtest uses a synthetic market-price proxy rather than historical option quotes.'))

        meta1, meta2, meta3 = st.columns(3)
        meta1.metric("Fair Value Model", methodology.get('fair_value_model', bt_model).upper())
        meta2.metric("Entry Price Source", "BS Proxy" if methodology.get('entry_market_price_source') else "Unknown")
        meta3.metric("Look-Ahead Guard", "ON" if methodology.get('lookahead_guard') else "OFF")

        km1, km2, km3, km4, km5, km6 = st.columns(6)
        pnl = results['final_value'] - bt_capital
        km1.metric("Final Value", f"${results['final_value']:,.2f}", delta=f"${pnl:+,.2f}", delta_color="normal")
        km2.metric("Return", f"{results['total_return_pct']:+.1f}%")
        km3.metric("Win Rate", f"{results['win_rate']:.0f}%")
        km4.metric("Trades", f"{results['total_trades']}")
        km5.metric("Sharpe", f"{results['sharpe_ratio']:.2f}")
        km6.metric("Max DD", f"{results['max_drawdown_pct']:.1f}%")

        st.markdown("---")
        st.subheader("Equity Curve")
        equity_df = results['equity_curve']
        if not equity_df.empty and len(equity_df) > 1:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity_df['date'], y=equity_df['value'],
                mode='lines+markers', name='Portfolio',
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
            compact_cols = ['entry_date', 'expiry_date', 'strike', 'mc_fair_value', 'market_proxy_price', 'pnl_net', 'result']
            compact_df = trades_df[[col for col in compact_cols if col in trades_df.columns]].copy()
            compact_df = compact_df.rename(columns={
                'entry_date': 'Entry', 'expiry_date': 'Exit',
                'strike': 'Strike', 'mc_fair_value': 'Fair Value', 'market_proxy_price': 'Mkt Proxy',
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
                    use_container_width=True
                )
            with export_col3:
                json_data = trades_df.to_json(orient="records", indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"backtest_trades_{ticker}.json",
                    mime="application/json",
                    use_container_width=True
                )

            # Detailed forensic view
            with st.expander("Detailed Forensic Analysis"):
                st.markdown("*Complete trade record with costs, volatility, and sensitivity analysis*")
                styled_trades = trades_df.style.map(_color_result, subset=['result'])
                st.dataframe(styled_trades, width='stretch', hide_index=True)
        else:
            st.info("No trades triggered. Try lowering the Edge Threshold.")

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
