import numpy as np
import pandas as pd

from src.core.scanner_engine import scan_for_arbitrage, scan_for_valuation_gaps
from src.core.calibration_engine import _implied_vol_from_price, calibrate_heston
from src.core.heston_model import price_option_heston_fourier


def _sample_options_df():
    return pd.DataFrame(
        [
            {
                "type": "call",
                "strike": 100.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 4.8,
                "ask": 5.2,
                "mid": 5.0,
                "volume": 100,
                "openInterest": 200,
                "market_iv": 0.20,
            }
        ]
    )


def test_scanner_gbm_path_returns_rows():
    df = _sample_options_df()
    out = scan_for_arbitrage(df, S0=100.0, r=0.03, q=0.01, model="gbm")
    assert not out.empty
    assert out.iloc[0]["type"] == "CALL"


def test_scanner_heston_path_returns_rows():
    df = _sample_options_df()
    out = scan_for_arbitrage(df, S0=100.0, r=0.03, q=0.01, model="heston")
    assert not out.empty
    assert out.iloc[0]["type"] == "CALL"


def test_calibrated_scan_beats_arbitrary_parameters():
    """A market generated from a known Heston surface contains no real
    mispricings, so a correctly-parameterised scan should report only small
    residual edges. Scanning with arbitrary (uncalibrated) parameters instead
    produces large, systematic phantom gaps. This guards the scanner's core
    premise: calibrate first, then flag the residuals."""
    S0, r, q = 100.0, 0.03, 0.0
    true_params = dict(V0=0.05, kappa=3.0, theta=0.06, xi=0.5, rho=-0.75)

    rows = []
    for T in [0.1, 0.25, 0.5, 1.0]:
        for K in np.linspace(85, 115, 9):
            price = price_option_heston_fourier(
                S0, K, T, r, true_params['V0'], true_params['kappa'],
                true_params['theta'], true_params['xi'], true_params['rho'],
                'call', q,
            )
            iv = _implied_vol_from_price(price, S0, K, T, r, 'call', q)
            rows.append(dict(strike=K, T=T, type='call', bid=price * 0.99,
                             ask=price * 1.01, mid=price, market_iv=iv,
                             volume=500, openInterest=2000,
                             expiration='2026-12-19'))
    df = pd.DataFrame(rows)

    uncalibrated = scan_for_valuation_gaps(
        df, S0, r, q, model='heston',
        heston_V0=0.04, heston_kappa=2.0, heston_theta=0.04,
        heston_xi=0.3, heston_rho=-0.7,
    )
    calib = calibrate_heston(df, S0, r, q=q)
    assert calib['success']
    calibrated = scan_for_valuation_gaps(
        df, S0, r, q, model='heston',
        heston_V0=calib['V0'], heston_kappa=calib['kappa'],
        heston_theta=calib['theta'], heston_xi=calib['xi'],
        heston_rho=calib['rho'],
    )

    # The calibrated scan's residual edges are far smaller than the phantom
    # gaps produced by arbitrary parameters.
    assert calibrated['edge_pct'].abs().mean() < uncalibrated['edge_pct'].abs().mean() / 3.0


def test_scanner_returns_diagnostics_for_filtered_contracts():
    df = pd.DataFrame(
        [
            {
                "type": "call",
                "strike": 100.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 4.8,
                "ask": 5.2,
                "mid": 5.0,
                "volume": 100,
                "openInterest": 200,
                "market_iv": 0.20,
            },
            {
                "type": "call",
                "strike": 140.0,
                "expiration": "2026-04-17",
                "T": 0.25,
                "bid": 0.4,
                "ask": 0.6,
                "mid": 0.5,
                "volume": 10,
                "openInterest": 20,
                "market_iv": 0.22,
            },
        ]
    )

    out, diagnostics = scan_for_arbitrage(
        df,
        S0=100.0,
        r=0.03,
        q=0.01,
        model="gbm",
        return_diagnostics=True,
    )

    assert not out.empty
    assert diagnostics["total_contracts"] == 2
    assert diagnostics["contracts_priced"] == 1
    assert diagnostics["contracts_filtered"] == 1
    assert diagnostics["reason_counts"]["moneyness"] == 1
    assert sum(diagnostics["signal_counts"].values()) == 1

