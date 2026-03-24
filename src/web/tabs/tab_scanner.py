"""
tab_scanner.py

Tab 2 - Live Valuation Gap Scanner
Handles the live options chain fetch, async API call to FastAPI backend,
fallback local scan, diagnostics, and results display.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

from src.core.data_fetcher import (
    get_available_expirations,
    get_market_data_runtime_summary,
    get_options_chain,
    get_risk_free_rate,
    get_spot_and_vol,
)
from src.core.scanner_engine import scan_for_valuation_gaps
from src.core.calibration_engine import calibrate_heston, calibrate_lsv


SCAN_API_URL = os.getenv("QUANT_TERMINAL_SCAN_API_URL", "http://127.0.0.1:8000/scan")

@st.cache_data(ttl=300)
def fetch_available_expirations(ticker):
    return get_available_expirations(ticker)


@st.cache_data(ttl=120)
def fetch_spot_data(ticker):
    return get_spot_and_vol(ticker)


@st.cache_data(ttl=300)
def fetch_risk_free_rate():
    return get_risk_free_rate()


def _style_signal(val):
    if 'BUY' in str(val):
        return 'color: #00FF00; font-weight: bold'
    elif 'SELL' in str(val):
        return 'color: #FF4444; font-weight: bold'
    return 'color: #888888'


def _style_edge(val):
    try:
        v = float(val)
        if v > 0:
            return 'color: #00FF00'
        elif v < 0:
            return 'color: #FF4444'
    except Exception:
        pass
    return ''


def _build_expiration_choices(available_exps):
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


def _run_local_scan(options_df, S0, r_live, scan_model, scanner_sims, default_vol,
                    jump_intensity, jump_mean, jump_std,
                    heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
                    lsv_leverage, lsv_strikes, lsv_mats, engine_label):
    results_df, scan_diagnostics = scan_for_valuation_gaps(
        options_df, S0, r_live, model=scan_model,
        leverage_matrix=lsv_leverage,
        leverage_strikes=lsv_strikes,
        leverage_maturities=lsv_mats,
        heston_V0=heston_V0, heston_kappa=heston_kappa,
        heston_theta=heston_theta, heston_xi=heston_xi, heston_rho=heston_rho,
        jump_intensity=jump_intensity, jump_mean=jump_mean, jump_std=jump_std,
        sigma_fallback=default_vol, max_spread_pct=0.25,
        n_sims=scanner_sims,
        return_diagnostics=True
    )
    scan_diagnostics['engine'] = engine_label
    return results_df, scan_diagnostics


def _extract_api_detail(response):
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get('detail', response.text)
    except Exception:
        pass
    return response.text


def render(ticker, model_type, default_vol, n_sims,
           jump_intensity, jump_mean, jump_std,
           heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho):
    """Renders the full valuation-gap scanner tab."""

    st.subheader(f"Live Valuation Gap Scanner: {ticker}")
    st.caption("Signals rank model-versus-market valuation gaps and include data-quality diagnostics so you can explain what was filtered, how volatility was sourced, and which engine produced the prices.")
    market_data_summary = get_market_data_runtime_summary()
    if market_data_summary["provider_preference"] != "yfinance":
        st.info(market_data_summary["options_chain_note"])

    scan_col1, scan_col2, scan_col3 = st.columns([2, 2, 1])
    with scan_col1:
        scanner_sims = st.select_slider("MC Simulations per Contract",
            options=[1000, 5000, 10000, 25000, 50000], value=5000,
            help="Higher = more accurate but slower. 5,000 is a good balance.")
            
    available_exps = fetch_available_expirations(ticker)
    expiration_choices = _build_expiration_choices(available_exps)
    
    with scan_col2:
        if expiration_choices:
            exp_labels = {label: exp_str for label, exp_str, _ in expiration_choices}
            preferred_labels = [label for label, _, days_left in expiration_choices if days_left >= 1]
            default_labels = preferred_labels[:min(2, len(preferred_labels))]
            if not default_labels:
                default_labels = [label for label, _, _ in expiration_choices[:min(2, len(expiration_choices))]]

            chosen_labels = st.multiselect(
                "Expiration Dates",
                options=list(exp_labels.keys()),
                default=default_labels,
                help="Select which option expiry dates to scan. Labels show: Date · Days to Expiry"
            )
            selected_exps = [exp_labels[l] for l in chosen_labels]
        else:
            selected_exps = []
            st.error("Could not fetch active future expiration dates.")

    with scan_col3:
        st.markdown("**Scan Actions**")
        scan_button = st.button("Run Scan", type="primary", use_container_width=True,
            help="Run the pricing engine to rank valuation gaps across the selected expirations.")

    if not selected_exps:
        st.warning(f"Please select at least one expiration date to scan {ticker}.")
        return

    # --- Calibration Section ---
    with st.expander("Advanced: Calibrate Models to Market (Optional)", expanded=False):
        st.markdown("""
        Calibration fits model parameters to today's live market prices, improving valuation-gap estimates.
        Run these steps in order before clicking **Run Scan**.
        """)
        cal_col1, cal_col2 = st.columns(2)
        with cal_col1:
            st.markdown("**Heston Calibration**")
            st.caption("Fits κ, θ, ξ, ρ to the live implied volatility surface. Updates the sidebar parameters automatically.")
            calibrate_button = st.button("Calibrate Heston to Market", use_container_width=True,
                help="Fits the Heston stochastic-vol parameters to today's market implied volatility surface.")
        with cal_col2:
            st.markdown("**LSV Calibration**")
            st.caption("Builds a leverage function on top of Heston to better match the observed market surface.")
            lsv_calibrate_button = st.button("Calibrate LSV Leverage Function", use_container_width=True,
                help="Derives the non-parametric leverage function L(S,t) from the market surface. Requires Heston calibration first.")

    # --- Execute Calibration ---
    if calibrate_button:
        with st.spinner(f"Calibrating Heston to market surface for {ticker}..."):
            r_live = fetch_risk_free_rate()
            spot_data = fetch_spot_data(ticker)
            options_df = get_options_chain(ticker, specific_expirations=selected_exps)
            if spot_data is None:
                st.error(f"Could not fetch spot data for {ticker}.")
            elif not options_df.empty:
                calib_res = calibrate_heston(options_df, spot_data['spot'], r_live)
                if calib_res['success']:
                    st.success(f"Heston calibrated. {calib_res['message']}")
                    st.session_state['heston_kappa'] = calib_res['kappa']
                    st.session_state['heston_theta'] = calib_res['theta']
                    st.session_state['heston_xi'] = calib_res['xi']
                    st.session_state['heston_rho'] = calib_res['rho']
                    st.session_state['heston_V0'] = calib_res['V0']
                    st.info("Sidebar parameters updated. You can run LSV calibration next or scan directly.")
                else:
                    st.error(calib_res['message'])
            else:
                st.error("No options data for calibration.")

    # --- LSV Calibration ---
    if lsv_calibrate_button:
        with st.spinner(f"Calibrating LSV model for {ticker}..."):
            r_live = fetch_risk_free_rate()
            spot_data = fetch_spot_data(ticker)
            options_df = get_options_chain(ticker, specific_expirations=selected_exps)

            if spot_data is None:
                st.error(f"Could not fetch spot data for {ticker}.")
            elif not options_df.empty:
                # Get current Heston params or use defaults
                heston_params = {
                    'kappa': st.session_state.get('heston_kappa', heston_kappa),
                    'theta': st.session_state.get('heston_theta', heston_theta),
                    'xi': st.session_state.get('heston_xi', heston_xi),
                    'rho': st.session_state.get('heston_rho', heston_rho),
                    'V0': st.session_state.get('heston_V0', heston_V0)
                }

                # Calibrate LSV
                lsv_res = calibrate_lsv(
                    options_df, spot_data['spot'], r_live,
                    heston_params=heston_params,
                    num_strikes=None,  # Auto-detect
                    num_mats=None      # Auto-detect
                )

                if lsv_res.get('success', False):
                    st.success(f"{lsv_res['message']}")
                    st.session_state['lsv_leverage_matrix'] = lsv_res['leverage_matrix']
                    st.session_state['lsv_strikes'] = lsv_res['strikes_grid']
                    st.session_state['lsv_maturities'] = lsv_res['maturities_grid']
                    st.info("LSV parameters loaded. Select model 'Heston' and click 'Run Scan' to use the calibrated leverage function.")
                else:
                    st.error(f"{lsv_res.get('message', 'Unknown error')}")
            else:
                st.error("No options data for LSV calibration.")

    st.markdown("---")

    # --- Scan ---
    if scan_button:
        with st.spinner(f"Fetching options chain for {ticker}..."):
            spot_data = fetch_spot_data(ticker)
            r_live = fetch_risk_free_rate()
            options_df = get_options_chain(ticker, specific_expirations=selected_exps)

        if spot_data is None:
            st.error(f"Could not fetch spot data for {ticker}.")
            return
        if options_df.empty:
            st.warning(
                f"No active options quotes were returned for the selected {ticker} expirations. "
                "Try deselecting 0DTE dates or choosing the next available expiry."
            )
            return

        S0 = spot_data['spot']
        hist_vol = spot_data['historical_vol']

        ctx1, ctx2, ctx3, ctx4 = st.columns(4)
        ctx1.metric("Spot", f"${S0:.2f}")
        ctx2.metric("Hist Vol", f"{hist_vol*100:.1f}%")
        ctx3.metric("Risk-Free", f"{r_live*100:.2f}%")
        ctx4.metric("Contracts", f"{len(options_df)}")
        st.markdown("---")

        model_map = {
            "Standard GBM": "gbm",
            "Jump Diffusion": "jump_diffusion",
            "Heston (Stochastic Vol)": "heston"
        }
        scan_model = model_map.get(model_type, "heston")

        # Prepare LSV leverage if calibrated and model=='heston' (use LSV if calibrated)
        lsv_leverage = None
        lsv_strikes = None
        lsv_mats = None
        if 'lsv_leverage_matrix' in st.session_state:
            lsv_leverage = st.session_state['lsv_leverage_matrix']
            lsv_strikes = st.session_state['lsv_strikes']
            lsv_mats = st.session_state['lsv_maturities']
            # If LSV is calibrated and user is running Heston model, we'll use LSV leverage
            if scan_model == 'heston':
                scan_model = 'lsv'  # Use LSV with calibrated leverage

        payload = {
            "ticker": ticker, "r": r_live, "q": 0.0, "model": scan_model,
            "n_sims": scanner_sims,
            "jump_intensity": jump_intensity, "jump_mean": jump_mean, "jump_std": jump_std,
            "heston_V0": heston_V0, "heston_kappa": heston_kappa,
            "heston_theta": heston_theta, "heston_xi": heston_xi, "heston_rho": heston_rho,
            "max_spread_pct": 0.25,
            "options_data": options_df.to_dict(orient="records"),
            "spot": S0, "historical_vol": hist_vol
        }

        results_df = pd.DataFrame()
        scan_diagnostics = {}
        execution_mode = "UNKNOWN"
        with st.spinner("Connecting to pricing engine..."):
            try:
                response = requests.post(SCAN_API_URL, json=payload, timeout=30)
                if response.status_code == 200:
                    response_payload = response.json()
                    results_df = pd.DataFrame(response_payload.get("data", []))
                    scan_diagnostics = response_payload.get("diagnostics", {})
                    execution_mode = "ASYNC API (FastAPI Backend)"
                elif response.status_code in (400, 404):
                    detail = _extract_api_detail(response)
                    st.info(
                        f"Pricing API returned no active contracts ({response.status_code}). "
                        "Retrying locally with the same payload."
                    )
                    results_df, scan_diagnostics = _run_local_scan(
                        options_df, S0, r_live, scan_model, scanner_sims, default_vol,
                        jump_intensity, jump_mean, jump_std,
                        heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
                        lsv_leverage, lsv_strikes, lsv_mats,
                        engine_label='local_after_api_no_data'
                    )
                    scan_diagnostics['api_detail'] = detail
                    execution_mode = "LOCAL FALLBACK (API no-data response)"
                else:
                    st.error(f"API Error {response.status_code}: {response.text}")
                    execution_mode = "API ERROR"
            except requests.exceptions.ConnectionError:
                # Graceful degradation: API offline, run locally
                try:
                    results_df, scan_diagnostics = _run_local_scan(
                        options_df, S0, r_live, scan_model, scanner_sims, default_vol,
                        jump_intensity, jump_mean, jump_std,
                        heston_V0, heston_kappa, heston_theta, heston_xi, heston_rho,
                        lsv_leverage, lsv_strikes, lsv_mats,
                        engine_label='local_fallback'
                    )
                    execution_mode = "LOCAL FALLBACK (synchronous)"
                except Exception as e:
                    st.error(f"Scanner Engine Failure: {str(e)}")
                    execution_mode = "SCANNER ERROR"
            except Exception as e:
                st.error(f"API Request Failure: {str(e)}")
                execution_mode = "REQUEST ERROR"

        # Persist execution mode so diagnostics panel can display it
        st.session_state['last_execution_mode'] = execution_mode

        if results_df.empty:
            st.warning("No viable options found after filtering.")
            return

        buys = len(results_df[results_df['signal'].str.contains('BUY')])
        sells = len(results_df[results_df['signal'].str.contains('SELL')])
        holds = len(results_df[results_df['signal'].str.contains('HOLD')])

        sig1, sig2, sig3, sig4 = st.columns(4)
        sig1.metric("Scanned", len(results_df))
        sig2.metric("[ BUY ]", buys)
        sig3.metric("[ SELL ]", sells)
        sig4.metric("[ HOLD ]", holds)

        with st.expander("Scanner Diagnostics & Data Quality"):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Data Integrity**")
                total_contracts = scan_diagnostics.get('total_contracts', len(options_df))
                contracts_priced = scan_diagnostics.get('contracts_priced', len(results_df))
                contracts_filtered = scan_diagnostics.get('contracts_filtered', total_contracts - contracts_priced)
                st.write(f"- Contracts Fetched: {total_contracts}")
                st.write(f"- Contracts Priced: {contracts_priced}")
                st.write(f"- Filtered Out: {contracts_filtered}")
                reason_counts = scan_diagnostics.get('reason_counts', {})
                for reason, count in reason_counts.items():
                    if count > 0:
                        label = reason.replace('_', ' ').title()
                        st.write(f"- {label}: {count}")
                st.caption("Filters enforce quote validity, moneyness, DTE, spread controls, ghost-contract removal, and pricing-error quarantine.")
            with d2:
                st.markdown("**Volatility Sourcing**")
                sigma_source_counts = scan_diagnostics.get('sigma_source_counts', {})
                if sigma_source_counts:
                    for src, count in sigma_source_counts.items():
                        st.write(f"- {src}: {count} contracts")
                elif 'sigma_source' in results_df.columns:
                    for src, count in results_df['sigma_source'].value_counts().items():
                        st.write(f"- {src}: {count} contracts")
                else:
                    st.write("- No sigma source data")
            with d3:
                st.markdown("**Execution Mode**")
                mode = st.session_state.get('last_execution_mode', 'UNKNOWN')
                st.write(f"- Engine: {mode}")
                st.write(f"- Model: {scan_diagnostics.get('model_used', scan_model).upper()}")
                st.write(f"- Sims/contract: {scanner_sims:,}")
                signal_counts = scan_diagnostics.get('signal_counts', {})
                if signal_counts:
                    st.write(f"- BUY/SELL/HOLD: {signal_counts.get('BUY', 0)}/{signal_counts.get('SELL', 0)}/{signal_counts.get('HOLD', 0)}")

        st.markdown("---")
        st.subheader("Scan Results")

        # --- Filtering Controls ---
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        with filter_col1:
            min_edge = st.number_input("Min Edge (%)", value=0.0, step=0.1, help="Filter for edges above this threshold")
        with filter_col2:
            max_spread = st.number_input("Max Spread (%)", value=100.0, step=1.0, help="Filter for spreads below this threshold")
        with filter_col3:
            option_type_filter = st.selectbox("Type", ["All", "Call", "Put"], help="Filter by option type")
        with filter_col4:
            min_dte = st.number_input("Min DTE (days)", value=0, step=1, help="Filter for options with at least this many days")

        # Apply filters
        filtered_df = results_df.copy()

        # Apply edge filter
        if min_edge > 0:
            filtered_df = filtered_df[filtered_df['edge_pct'] >= min_edge]

        # Apply spread filter
        if max_spread < 100:
            filtered_df = filtered_df[filtered_df['spread_pct'] * 100 <= max_spread]

        # Apply type filter
        if option_type_filter != "All":
            filtered_df = filtered_df[filtered_df['type'].str.upper() == option_type_filter.upper()]

        # Apply DTE filter
        if min_dte > 0:
            filtered_df = filtered_df[filtered_df['T_days'] >= min_dte]

        # Show filter summary
        st.caption(f"Showing {len(filtered_df)} of {len(results_df)} valuation gaps | Filtered: {len(results_df) - len(filtered_df)}")

        st.markdown("---")

        # Essential columns for quick scanning
        essential_cols = ['type', 'strike', 'expiration', 'T_days', 'bid', 'ask', 'mc_price', 'edge', 'edge_pct', 'sigma_source', 'signal']
        display_df = filtered_df[[col for col in essential_cols if col in filtered_df.columns]].copy()

        display_df = display_df.rename(columns={
            'type': 'Type', 'strike': 'Strike', 'expiration': 'Expires',
            'T_days': 'DTE', 'bid': 'Bid', 'ask': 'Ask',
            'mc_price': 'MC Price', 'edge': 'Edge $', 'edge_pct': 'Edge %',
            'sigma_source': 'Sigma Source',
            'signal': 'Signal'
        })

        # Apply styling
        styled = display_df.style.map(_style_signal, subset=['Signal'])
        if 'Edge %' in display_df.columns:
            styled = styled.map(_style_edge, subset=['Edge %'])

        st.dataframe(styled, width='stretch', height=500, hide_index=True)

        # --- Export Options ---
        st.markdown("---")
        export_col1, export_col2, export_col3 = st.columns([2, 1, 1])
        with export_col1:
            st.markdown("**Export Results:**")
        with export_col2:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"scan_results_{ticker}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with export_col3:
            json_data = filtered_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"scan_results_{ticker}.json",
                mime="application/json",
                use_container_width=True
            )

        st.markdown("---")
        with st.expander("Advanced: Show All Scanner Details"):
            st.markdown("*Contains technical columns: spreads, volatility source, liquidity adjustments, etc.*")
            detailed_df = filtered_df.rename(columns={
                'type': 'Type', 'strike': 'Strike', 'expiration': 'Expires',
                'T_days': 'Days', 'bid': 'Bid', 'ask': 'Ask', 'mid': 'Mid',
                'spread': 'Spread$', 'spread_pct': 'Spread%',
                'mc_price': 'MC Value', 'bs_price': 'BS Price',
                'edge': 'Edge$', 'edge_pct': 'Edge%',
                'signal': 'Signal', 'volume': 'Vol', 'openInterest': 'OI',
                'market_iv': 'IV%', 'sigma_source': 'IV Source',
            })
            detailed_styled = detailed_df.style.map(_style_signal, subset=['Signal'])
            st.dataframe(detailed_styled, width='stretch', hide_index=True)

        st.markdown("---")
        st.subheader("Top Valuation Gaps")
        top_n = min(10, len(filtered_df))
        if top_n > 0:
            topN = filtered_df.head(top_n).copy()
            topN['label'] = topN['type'] + ' ' + topN['strike'].astype(str) + ' (' + topN['expiration'] + ')'
            colors = ['#00FF00' if e > 0 else '#FF4444' for e in topN['edge_pct']]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=topN['label'], y=topN['edge_pct'],
                marker_color=colors, text=topN['signal'], textposition='outside'))
            fig_bar.update_layout(xaxis_title="Contract", yaxis_title="Edge (%)",
                height=380, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig_bar, width='stretch')

    else:
        st.info("Click **RUN SCAN** to rank model-versus-market valuation gaps.")
        st.markdown("""
        | Signal | Meaning |
        |--------|---------|
        | [ BUY ] | MC Fair Value > Market Ask (positive buy-side valuation gap after crossing spread) |
        | [ SELL ] | MC Fair Value < Market Bid (positive sell-side valuation gap after crossing spread) |
        | [ HOLD ] | No actionable edge after accounting for bid-ask spread |
        """)
