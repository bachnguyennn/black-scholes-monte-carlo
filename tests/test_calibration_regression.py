import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.calibration_engine import _implied_vol_from_price, calibrate_heston
from src.core.heston_model import price_option_heston_fourier


def _synthetic_heston_surface():
    S0 = 100.0
    r = 0.03
    q = 0.0
    true_params = {
        'kappa': 2.2,
        'theta': 0.045,
        'xi': 0.28,
        'rho': -0.65,
        'V0': 0.04,
    }

    rows = []
    for maturity in [0.25, 0.75]:
        for strike in [90.0, 100.0, 110.0]:
            price = price_option_heston_fourier(
                S0,
                strike,
                maturity,
                r,
                true_params['V0'],
                true_params['kappa'],
                true_params['theta'],
                true_params['xi'],
                true_params['rho'],
                option_type='call',
                q=q,
            )
            market_iv = _implied_vol_from_price(price, S0, strike, maturity, r, option_type='call', q=q)
            rows.append(
                {
                    'strike': strike,
                    'T': maturity,
                    'type': 'call',
                    'bid': price * 0.99,
                    'ask': price * 1.01,
                    'mid': price,
                    'market_iv': market_iv,
                    'volume': 250,
                    'openInterest': 1200,
                    'expiration': '2026-12-19',
                }
            )

    return pd.DataFrame(rows), S0, r, q, true_params


def test_calibrate_heston_fits_synthetic_surface():
    options_df, S0, r, q, true_params = _synthetic_heston_surface()

    result = calibrate_heston(
        options_df,
        S0,
        r,
        q=q,
        initial_params={
            'kappa': 1.9,
            'theta': 0.04,
            'xi': 0.24,
            'rho': -0.55,
            'V0': 0.035,
        },
    )

    assert result['success'] is True
    assert result['n_contracts'] == len(options_df)
    assert result['sse'] < 5e-4
    assert bool(result['feller']['satisfied'])
    assert abs(result['rho'] - true_params['rho']) < 0.2


def test_calibrate_heston_rejects_too_few_contracts():
    options_df, S0, r, q, _ = _synthetic_heston_surface()
    insufficient_df = options_df.iloc[:4].copy()

    result = calibrate_heston(insufficient_df, S0, r, q=q)

    assert result['success'] is False
    assert result['n_contracts'] == 4
    assert 'Insufficient calibration data' in result['message']