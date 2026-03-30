"""
tab_model_validation.py

Dedicated model-validation view for quote-based surface diagnostics.
This tab separates live quote fit assessment from the valuation-gap scanner
and the synthetic historical backtester.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.data_fetcher import (
    get_available_expirations,
    get_options_chain,
    get_risk_free_rate,
    get_spot_and_vol,
)
from src.core.model_evaluation import build_live_surface_evaluation
from src.core.scanner_engine import scan_for_valuation_gaps


@st.cache_data(ttl=300)
def fetch_available_expirations(ticker):
    return get_available_expirations(ticker)


@st.cache_data(ttl=120)
def fetch_spot_data(ticker):
    return get_spot_and_vol(ticker)


@st.cache_data(ttl=300)
def fetch_risk_free_rate():
    return get_risk_free_rate()


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


def _format_metric(value, fmt):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if pd.isna(numeric):
        return "n/a"
    return format(numeric, fmt)


def render(
    ticker,
    model_type,
    default_vol,
    n_sims,
    jump_intensity,
    jump_mean,
    jump_std,
    heston_V0,
    heston_kappa,
    heston_theta,
    heston_xi,
    heston_rho,
):
    """Render quote-based model validation metrics on the live option surface."""
    st.subheader(f"Model Validation: {ticker}")
    st.caption(
        "Use this tab to evaluate how well the selected model explains current option quotes. "
        "These diagnostics are quote-based and should be treated as more important than the synthetic backtest."
    )

    with st.expander("What this tab measures", expanded=False):
        st.markdown(
            """
            - **Primary:** out-of-sample style price and IV fit on the observed option surface
            - **Secondary:** error measured relative to quoted spread and percent of model prices inside NBBO
            - **Not available yet:** true walk-forward historical surface testing and delta-hedged residual PnL volatility, which require archived option quotes
            """
        )

    ctl1, ctl2, ctl3 = st.columns([2, 2, 1])
    available_exps = fetch_available_expirations(ticker)
    expiration_choices = _build_expiration_choices(available_exps)

    with ctl1:
        validator_sims = st.select_slider(
            "MC Simulations per Contract",
            options=[1000, 5000, 10000, 25000, 50000],
            value=min(max(int(n_sims), 1000), 50000) if int(n_sims) in [1000, 5000, 10000, 25000, 50000] else 5000,
            help="Used for simulation-based models when valuing the live chain.",
        )

    with ctl2:
        if expiration_choices:
            exp_labels = {label: exp_str for label, exp_str, _ in expiration_choices}
            default_labels = [label for label, _, _ in expiration_choices[: min(2, len(expiration_choices))]]
            chosen_labels = st.multiselect(
                "Validation Expirations",
                options=list(exp_labels.keys()),
                default=default_labels,
                help="Choose live expirations to evaluate against current quotes.",
            )
            selected_exps = [exp_labels[label] for label in chosen_labels]
        else:
            selected_exps = []
            st.error("Could not fetch active future expiration dates.")

    with ctl3:
        st.markdown("**Validation Run**")
        run_button = st.button("Run Validation", type="primary", use_container_width=True)

    if not selected_exps:
        st.info("Select at least one live expiration to evaluate the model surface.")
        return

    if not run_button:
        st.info("Click **Run Validation** to compute live-surface fit metrics.")
        return

    with st.spinner(f"Evaluating {model_type} against live quotes for {ticker}..."):
        spot_data = fetch_spot_data(ticker)
        r_live = fetch_risk_free_rate()
        options_df = get_options_chain(ticker, specific_expirations=selected_exps)

    if spot_data is None:
        st.error(f"Could not fetch spot data for {ticker}.")
        return
    if options_df.empty:
        st.warning("No options data returned for the selected expirations.")
        return

    S0 = float(spot_data["spot"])

    model_map = {
        "Standard GBM": "gbm",
        "Jump Diffusion": "jump_diffusion",
        "Heston (Stochastic Vol)": "heston",
    }
    eval_model = model_map.get(model_type, "heston")

    lsv_leverage = None
    lsv_strikes = None
    lsv_mats = None
    if "lsv_leverage_matrix" in st.session_state and eval_model == "heston":
        lsv_leverage = st.session_state["lsv_leverage_matrix"]
        lsv_strikes = st.session_state["lsv_strikes"]
        lsv_mats = st.session_state["lsv_maturities"]
        eval_model = "lsv"

    priced_df, diagnostics = scan_for_valuation_gaps(
        options_df,
        S0,
        r_live,
        model=eval_model,
        leverage_matrix=lsv_leverage,
        leverage_strikes=lsv_strikes,
        leverage_maturities=lsv_mats,
        heston_V0=heston_V0,
        heston_kappa=heston_kappa,
        heston_theta=heston_theta,
        heston_xi=heston_xi,
        heston_rho=heston_rho,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        sigma_fallback=default_vol,
        max_spread_pct=0.25,
        n_sims=validator_sims,
        return_diagnostics=True,
    )

    if priced_df.empty:
        st.warning("No contracts remained after quote-quality filtering.")
        return

    surface_eval = build_live_surface_evaluation(priced_df, S0, r_live)
    if not surface_eval.get("success"):
        st.warning(surface_eval.get("message", "Validation metrics unavailable."))
        return

    ctx1, ctx2, ctx3, ctx4 = st.columns(4)
    ctx1.metric("Contracts Evaluated", surface_eval["contracts_evaluated"])
    ctx2.metric("Model", eval_model.upper())
    ctx3.metric("Spot", f"${S0:.2f}")
    ctx4.metric("Risk-Free", f"{r_live*100:.2f}%")

    met1, met2, met3, met4, met5 = st.columns(5)
    met1.metric("Price MAE", f"${_format_metric(surface_eval['price_mae'], '.4f')}")
    met2.metric("Price RMSE", f"${_format_metric(surface_eval['price_rmse'], '.4f')}")
    met3.metric("IV MAE", f"{_format_metric(surface_eval['iv_mae_pct_pts'], '.2f')} pts")
    met4.metric("Within NBBO", f"{_format_metric(surface_eval['within_nbbo_pct'], '.1f')}%")
    met5.metric("Abs Error / Spread", f"{_format_metric(surface_eval['mean_abs_error_in_spreads'], '.2f')}x")

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Error vs Spread")
        spread_plot_df = priced_df.copy()
        spread_plot_df["abs_price_error"] = (spread_plot_df["mc_price"] - spread_plot_df["mid"]).abs()
        spread_plot_df["label"] = (
            spread_plot_df["type"].astype(str)
            + " "
            + spread_plot_df["strike"].astype(str)
            + " · "
            + spread_plot_df["expiration"].astype(str)
        )

        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=spread_plot_df["spread_pct"],
                y=spread_plot_df["abs_price_error"],
                mode="markers",
                marker=dict(
                    size=9,
                    color=spread_plot_df["T_days"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="DTE"),
                ),
                text=spread_plot_df["label"],
                hovertemplate="%{text}<br>Spread %: %{x:.2f}<br>Abs Error: %{y:.4f}<extra></extra>",
            )
        )
        fig_scatter.update_layout(
            xaxis_title="Quoted Spread (%)",
            yaxis_title="Absolute Price Error ($)",
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    with chart_col2:
        st.subheader("Error by Contract")
        ranked_df = priced_df.copy()
        ranked_df["price_error"] = ranked_df["mc_price"] - ranked_df["mid"]
        ranked_df["abs_price_error"] = ranked_df["price_error"].abs()
        ranked_df = ranked_df.sort_values("abs_price_error", ascending=False).head(12)
        ranked_df["label"] = (
            ranked_df["type"].astype(str)
            + " "
            + ranked_df["strike"].astype(str)
            + " ("
            + ranked_df["expiration"].astype(str)
            + ")"
        )

        colors = ["#FF6B6B" if x > 0 else "#4D96FF" for x in ranked_df["price_error"]]
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=ranked_df["label"],
                y=ranked_df["price_error"],
                marker_color=colors,
                hovertemplate="%{x}<br>Price Error: %{y:.4f}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            xaxis_title="Contract",
            yaxis_title="Model Price - Mid ($)",
            height=380,
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig_bar, width="stretch")

    st.markdown("---")
    st.subheader("Validation Table")
    validation_df = priced_df.copy()
    validation_df["price_error"] = validation_df["mc_price"] - validation_df["mid"]
    validation_df["abs_price_error"] = validation_df["price_error"].abs()
    validation_df["error_in_spreads"] = validation_df["abs_price_error"] / validation_df["spread"].replace(0.0, pd.NA)
    validation_df["inside_nbbo"] = (
        (validation_df["mc_price"] >= validation_df["bid"]) & (validation_df["mc_price"] <= validation_df["ask"])
    ).map({True: "YES", False: "NO"})
    validation_df = validation_df.sort_values("abs_price_error", ascending=False)

    display_cols = [
        "type",
        "strike",
        "expiration",
        "T_days",
        "bid",
        "ask",
        "mid",
        "mc_price",
        "price_error",
        "abs_price_error",
        "error_in_spreads",
        "inside_nbbo",
        "market_iv",
        "sigma_source",
    ]
    st.dataframe(
        validation_df[[col for col in display_cols if col in validation_df.columns]].rename(
            columns={
                "type": "Type",
                "strike": "Strike",
                "expiration": "Expires",
                "T_days": "DTE",
                "bid": "Bid",
                "ask": "Ask",
                "mid": "Mid",
                "mc_price": "Model Price",
                "price_error": "Error $",
                "abs_price_error": "Abs Error $",
                "error_in_spreads": "Error / Spread",
                "inside_nbbo": "Inside NBBO",
                "market_iv": "Market IV %",
                "sigma_source": "Sigma Source",
            }
        ),
        width="stretch",
        height=420,
        hide_index=True,
    )

    with st.expander("Validation Notes", expanded=False):
        st.markdown(
            f"""
            - Quote-quality filters removed {diagnostics.get('contracts_filtered', 0)} contracts before evaluation.
            - These metrics are based on current observed quotes, not on the synthetic backtester.
            - `Within NBBO` and `Error / Spread` are the most useful execution-aware diagnostics available under free-data constraints.
            - Delta-hedged residual PnL volatility is not shown because the repo does not store historical option quote paths.
            """
        )
