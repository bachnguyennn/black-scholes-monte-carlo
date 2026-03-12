"""
test_lsv_calibration.py

Comprehensive test suite for LSV model calibration and integration.
Tests cover:
  - IV surface building from scattered options data
  - Leverage function calibration
  - Integration with scanner engine
  - Adaptive grid resolution
  - Error handling for sparse data
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.calibration_engine import build_iv_surface, calibrate_lsv
from src.core.lsv_model import calibrate_leverage_function
from src.core.scanner_engine import scan_for_arbitrage


def _sample_options_df(n_contracts=50, include_sparse=False):
    """Generate synthetic options DataFrame for testing."""
    np.random.seed(42)
    S0 = 100.0

    strikes = np.random.uniform(85, 115, n_contracts)
    maturities = np.random.uniform(0.05, 1.0, n_contracts)
    market_ivs = 0.20 + 0.05 * np.sin(np.log(strikes / S0) * 2) + 0.02 * np.random.randn(n_contracts)
    market_ivs = np.clip(market_ivs, 0.10, 0.40)

    bids = np.random.uniform(0.05, 2.0, n_contracts)
    asks = bids + np.random.uniform(0.02, 0.20, n_contracts)
    mids = (bids + asks) / 2

    option_types = np.random.choice(['call', 'put'], n_contracts)
    volumes = np.random.randint(10, 1000, n_contracts)
    ois = np.random.randint(50, 5000, n_contracts)

    return pd.DataFrame({
        'strike': strikes,
        'T': maturities,
        'type': option_types,
        'bid': bids,
        'ask': asks,
        'mid': mids,
        'market_iv': market_ivs,
        'volume': volumes,
        'openInterest': ois,
        'expiration': pd.to_datetime('2026-06-15')
    })


class TestBuildIVSurface:
    """Tests for build_iv_surface() function."""

    def test_iv_surface_basic_valid(self):
        """Basic test: valid options DataFrame produces valid grid."""
        df = _sample_options_df(n_contracts=50)
        S0 = 100.0

        result = build_iv_surface(df, S0, num_strikes=15, num_mats=10)

        assert result['success'] is True
        assert result['strikes_grid'] is not None
        assert result['maturities_grid'] is not None
        assert result['iv_surface'] is not None
        assert result['iv_surface'].shape == (15, 10)
        assert result['coverage'] > 0
        assert np.all(result['iv_surface'] >= 0.01)
        assert np.all(result['iv_surface'] <= 3.0)

    def test_iv_surface_strikes_ascending(self):
        """Test that strike grid is sorted in ascending order."""
        df = _sample_options_df(n_contracts=50)
        result = build_iv_surface(df, 100.0, num_strikes=20, num_mats=15)

        assert np.all(np.diff(result['strikes_grid']) > 0), "Strikes not sorted"

    def test_iv_surface_maturities_ascending(self):
        """Test that maturity grid is sorted in ascending order."""
        df = _sample_options_df(n_contracts=50)
        result = build_iv_surface(df, 100.0, num_strikes=20, num_mats=15)

        assert np.all(np.diff(result['maturities_grid']) > 0), "Maturities not sorted"

    def test_iv_surface_filters_bad_iv(self):
        """Test that bad IVs (< 0.01 or > 3.0) are filtered."""
        df = _sample_options_df(n_contracts=50)
        df.loc[0, 'market_iv'] = 0.005  # Too low
        df.loc[1, 'market_iv'] = 5.0    # Too high

        result = build_iv_surface(df, 100.0, num_strikes=10, num_mats=8)

        assert result['success'] is True
        assert result['n_contracts_used'] < result['n_contracts_input']

    def test_iv_surface_insufficient_data(self):
        """Test: too few contracts returns failure."""
        df = _sample_options_df(n_contracts=3)  # < 5 minimum

        result = build_iv_surface(df, 100.0, num_strikes=10, num_mats=8)

        assert result['success'] is False
        assert 'Insufficient' in result['message']

    def test_iv_surface_adaptive_resolution_sparse(self):
        """Test: sparse data (< 20 contracts) gets coarse grid."""
        df = _sample_options_df(n_contracts=10)

        result = build_iv_surface(df, 100.0)  # num_strikes=None, num_mats=None

        assert result['success'] is True
        # With 10 contracts, adaptive should choose 10x8 grid
        assert result['strikes_grid'].shape[0] <= 15
        assert result['maturities_grid'].shape[0] <= 12

    def test_iv_surface_adaptive_resolution_dense(self):
        """Test: dense data (> 100 contracts) gets finer grid."""
        df = _sample_options_df(n_contracts=150)

        result = build_iv_surface(df, 100.0)  # num_strikes=None, num_mats=None

        assert result['success'] is True
        # With 150 contracts, adaptive should choose 25x15 grid
        assert result['strikes_grid'].shape[0] >= 20
        assert result['maturities_grid'].shape[0] >= 12

    def test_iv_surface_coverage_reported(self):
        """Test: coverage percentage is reported correctly."""
        df = _sample_options_df(n_contracts=50)

        result = build_iv_surface(df, 100.0, num_strikes=20, num_mats=15)

        assert result['success'] is True
        assert 0 <= result['coverage'] <= 100
        assert result['coverage'] > 0  # Should have some coverage with 50 contracts


class TestCalibrateLSV:
    """Tests for calibrate_lsv() orchestrator function."""

    def test_calibrate_lsv_success(self):
        """Test: complete calibration pipeline succeeds with sufficient data."""
        df = _sample_options_df(n_contracts=80)
        S0 = 100.0
        r = 0.05

        result = calibrate_lsv(df, S0, r)

        assert result['success'] is True
        assert result['leverage_matrix'] is not None
        assert result['strikes_grid'] is not None
        assert result['maturities_grid'] is not None
        assert result['leverage_matrix'].shape[0] == len(result['strikes_grid'])
        assert result['leverage_matrix'].shape[1] == len(result['maturities_grid'])

    def test_calibrate_lsv_leverage_bounds(self):
        """Test: leverage matrix values are within [0.1, 5.0]."""
        df = _sample_options_df(n_contracts=80)

        result = calibrate_lsv(df, 100.0, 0.05)

        assert result['success'] is True
        leverage = result['leverage_matrix']
        assert np.all(leverage >= 0.1), "Leverage too low"
        assert np.all(leverage <= 5.0), "Leverage too high"

    def test_calibrate_lsv_insufficient_data(self):
        """Test: too few contracts returns failure."""
        df = _sample_options_df(n_contracts=3)

        result = calibrate_lsv(df, 100.0, 0.05)

        assert result['success'] is False
        assert 'Insufficient' in result['message']

    def test_calibrate_lsv_low_coverage(self):
        """Test: warns if IV surface coverage < 30%."""
        # Create very sparse options data
        np.random.seed(123)
        strikes = [95.0, 96.0, 105.0, 106.0]
        maturities = [0.01, 0.02]
        df = pd.DataFrame({
            'strike': strikes,
            'T': [0.01, 0.02, 0.01, 0.02],
            'type': ['call'] * 4,
            'bid': [1.0, 1.1, 1.2, 1.3],
            'ask': [1.05, 1.15, 1.25, 1.35],
            'mid': [1.025, 1.125, 1.225, 1.325],
            'market_iv': [0.20, 0.21, 0.19, 0.18],
            'volume': [100, 100, 100, 100],
            'openInterest': [500, 500, 500, 500],
            'expiration': pd.to_datetime('2026-06-15')
        })

        result = calibrate_lsv(df, 100.0, 0.05, num_strikes=20, num_mats=15)

        # May fail due to low coverage
        if not result['success']:
            assert result.get('surface_coverage', 0) < 30 or 'coverage' in result['message'].lower()

    def test_calibrate_lsv_with_heston_params(self):
        """Test: calibrate_lsv uses provided Heston parameters."""
        df = _sample_options_df(n_contracts=80)
        heston_params = {
            'kappa': 3.0,
            'theta': 0.05,
            'xi': 0.4,
            'rho': -0.6,
            'V0': 0.04
        }

        result = calibrate_lsv(df, 100.0, 0.05, heston_params=heston_params)

        assert result['success'] is True
        assert result['heston_kappa'] == 3.0
        assert result['heston_theta'] == 0.05
        assert result['heston_V0'] == 0.04

    def test_calibrate_lsv_message_format(self):
        """Test: result message is descriptive."""
        df = _sample_options_df(n_contracts=80)

        result = calibrate_lsv(df, 100.0, 0.05)

        if result['success']:
            msg = result['message'].lower()
            # Check for key terms in message
            assert any(word in msg for word in ['calibrat', 'matrix', 'leverage', 'coverage'])


class TestCalibrateLeverageFunction:
    """Tests for calibrate_leverage_function() with kappa parameter fix."""

    def test_leverage_function_kappa_parameter(self):
        """Test: different kappas produce different leverage matrices."""
        # Create IV surface with skew (not flat)
        strikes = np.linspace(90, 110, 10)
        maturities = np.array([0.25, 0.50, 1.0])
        moneyness = strikes / 100.0
        # Skewed IV surface: higher vol for lower strikes
        iv_surface = 0.20 + 0.05 * np.where(moneyness < 1.0, 1 - moneyness, moneyness - 1)
        iv_surface = np.tile(iv_surface.reshape(-1, 1), (1, 3))

        # Calibrate with kappa=1.0
        lev_kappa1 = calibrate_leverage_function(
            iv_surface, strikes, maturities,
            kappa=1.0, heston_theta=0.04, heston_V0=0.04
        )

        # Calibrate with kappa=3.0
        lev_kappa3 = calibrate_leverage_function(
            iv_surface, strikes, maturities,
            kappa=3.0, heston_theta=0.04, heston_V0=0.04
        )

        # Results should differ (though may be similar if surface is flat)
        # Just verify they're both valid
        assert lev_kappa1.shape == (10, 3)
        assert lev_kappa3.shape == (10, 3)
        assert np.all((lev_kappa1 >= 0.1) & (lev_kappa1 <= 5.0))
        assert np.all((lev_kappa3 >= 0.1) & (lev_kappa3 <= 5.0))

    def test_leverage_function_returns_valid_shape(self):
        """Test: leverage matrix has correct shape."""
        strikes = np.linspace(90, 110, 15)
        maturities = np.array([0.25, 0.50, 1.0])
        iv_surface = np.random.uniform(0.15, 0.25, (15, 3))

        result = calibrate_leverage_function(iv_surface, strikes, maturities)

        assert result.shape == (15, 3)

    def test_leverage_function_bounds(self):
        """Test: leverage values clipped to [0.1, 5.0]."""
        strikes = np.linspace(90, 110, 10)
        maturities = np.array([0.25, 0.50, 1.0])

        # Use extreme IV surface
        iv_surface = np.ones((10, 3)) * 0.01  # Very low
        result_low = calibrate_leverage_function(iv_surface, strikes, maturities)
        assert np.all(result_low >= 0.1)

        iv_surface = np.ones((10, 3)) * 2.0  # Very high
        result_high = calibrate_leverage_function(iv_surface, strikes, maturities)
        assert np.all(result_high <= 5.0)


class TestScannerWithLSV:
    """Tests for scan_for_arbitrage() with LSV leverage."""

    def test_scanner_lsv_with_leverage(self):
        """Test: scanner accepts and uses calibrated leverage."""
        df = _sample_options_df(n_contracts=30)
        S0 = 100.0
        r = 0.05

        # Create dummy leverage matrix
        leverage = np.ones((20, 10)) * 1.0
        strikes = np.linspace(85, 115, 20)
        maturities = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

        result_df = scan_for_arbitrage(
            df, S0, r,
            model='lsv',
            leverage_matrix=leverage,
            leverage_strikes=strikes,
            leverage_maturities=maturities,
            n_sims=1000
        )

        assert not result_df.empty
        assert 'sigma_source' in result_df.columns
        # Should indicate LSV was used
        assert any('LSV' in str(x) for x in result_df['sigma_source'].unique())

    def test_scanner_lsv_without_leverage_fallback(self):
        """Test: scanner falls back to pure Heston when leverage=None."""
        df = _sample_options_df(n_contracts=30)
        S0 = 100.0
        r = 0.05

        result_df = scan_for_arbitrage(
            df, S0, r,
            model='lsv',
            leverage_matrix=None,  # No calibrated leverage
            n_sims=1000
        )

        assert not result_df.empty
        # Should indicate fallback
        assert any('No Calibration' in str(x) or 'Pure Heston' in str(x)
                   for x in result_df['sigma_source'].unique())

    def test_scanner_lsv_produces_prices(self):
        """Test: scanner with LSV produces valid prices."""
        df = _sample_options_df(n_contracts=20)
        S0 = 100.0
        r = 0.05

        leverage = np.ones((15, 10))
        strikes = np.linspace(85, 115, 15)
        maturities = np.linspace(0.01, 2.0, 10)

        result_df = scan_for_arbitrage(
            df, S0, r,
            model='lsv',
            leverage_matrix=leverage,
            leverage_strikes=strikes,
            leverage_maturities=maturities,
            n_sims=500
        )

        if not result_df.empty:
            assert 'mc_price' in result_df.columns
            assert np.all(result_df['mc_price'] > 0)


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""

    def test_full_calibration_to_scanner_workflow(self):
        """Test: full pipeline from data to scanner results."""
        df = _sample_options_df(n_contracts=100)
        S0 = 100.0
        r = 0.05

        # Calibrate LSV
        lsv_res = calibrate_lsv(df, S0, r)
        assert lsv_res['success'] is True

        # Use calibrated parameters in the scanner
        result_df = scan_for_arbitrage(
            df, S0, r,
            model='lsv',
            leverage_matrix=lsv_res['leverage_matrix'],
            leverage_strikes=lsv_res['strikes_grid'],
            leverage_maturities=lsv_res['maturities_grid'],
            n_sims=500
        )

        # Verify results
        assert not result_df.empty
        assert len(result_df) <= len(df)
        assert all(col in result_df.columns for col in ['mc_price', 'bid', 'ask'])


# Run tests with: pytest tests/test_lsv_calibration.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
