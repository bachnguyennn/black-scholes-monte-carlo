"""
config.py — Quant Research Terminal V2: Centralized Configuration

Single source of truth for all tuneable constants, risk thresholds, and
model assumptions. No magic numbers should appear inline in any module.
To change system behaviour, edit only this file.

Sections:
    SCANNER_*   -- Arbitrage scanner data quality and execution assumptions
    HESTON_*    -- Default Heston model parameter assumptions
    BACKTEST_*  -- Backtester defaults
    GREEK_*     -- Greek surface grid resolution
"""

# ============================================================================
# SCANNER: Data quality rules
# Reference: https://quant.stackexchange.com/questions/7935 for spread heuristics
# ============================================================================

# Moneyness bounds: only scan options within ±15% of spot.
# Deep OTM/ITM options break numerical integration and generate phantom signals.
SCANNER_MONEYNESS_LOWER = 0.85   # Strike must be >= spot * 0.85
SCANNER_MONEYNESS_UPPER = 1.15   # Strike must be <= spot * 1.15

# Minimum time-to-expiry to price. Options expiring today have undefined drift.
SCANNER_MIN_T_DAYS = 1           # Expressed in days; converted to T = N/365 in scanner

# Maximum allowable bid-ask spread as a fraction of mid price.
# Contracts wider than this are treated as illiquid -- not actionable.
SCANNER_MAX_SPREAD_PCT = 0.25

# Minimum valid market implied volatility. Below this, IV data is likely missing
# or broken and we fall back to the historical volatility.
SCANNER_MIN_MARKET_IV = 0.01     # 1%

# ============================================================================
# SCANNER: Order-Book Slippage Model
# Proxy for market impact using volume/open-interest as a liquidity score.
# These tiers are based on typical equity options market depth conventions:
#   Tier 1 (<50 contracts):    Illiquid — wide effective spread expected (~5%)
#   Tier 2 (50–249 contracts): Light liquidity — moderate slippage (~2%)
#   Tier 3 (250–999):          Adequate liquidity — minor slippage (~0.5%)
#   Tier 4 (>=1000):           Liquid market — no additional slippage penalty
# ============================================================================

SLIPPAGE_TIER_1_THRESHOLD = 50       # Max liquidity score for Tier 1 (illiquid)
SLIPPAGE_TIER_2_THRESHOLD = 250      # Max liquidity score for Tier 2
SLIPPAGE_TIER_3_THRESHOLD = 1000     # Max liquidity score for Tier 3
SLIPPAGE_TIER_1_PENALTY   = 0.05    # 5% price penalty for illiquid contracts
SLIPPAGE_TIER_2_PENALTY   = 0.02    # 2% price penalty for lightly liquid
SLIPPAGE_TIER_3_PENALTY   = 0.005   # 0.5% price penalty for moderate liquidity
SLIPPAGE_TIER_4_PENALTY   = 0.0     # No penalty for deep liquid contracts

# ============================================================================
# HESTON MODEL: Default Parameter Assumptions
# Based on Gatheral (2006) calibration to equity index surfaces.
# These are ONLY used when no calibration has been performed.
# ============================================================================

HESTON_DEFAULT_KAPPA = 2.0    # Mean reversion speed (half-life ~4 months)
HESTON_DEFAULT_XI    = 0.3    # Vol of vol (moderate, typical for SPY)
HESTON_DEFAULT_RHO   = -0.7   # Correlation (leverage effect, negative for equities)

# ============================================================================
# GREEK SURFACE: Grid Resolution & Bounds
# ============================================================================

GREEK_GRID_POINTS   = 50     # N x N grid for 3D surface (higher = smoother, slower)
GREEK_VOL_MIN       = 0.05   # Minimum sigma on the vol axis (5%)
GREEK_VOL_MAX       = 1.0    # Maximum sigma on the vol axis (100%)
GREEK_SPOT_LOWER    = 0.70   # Spot grid lower bound as fraction of default_spot
GREEK_SPOT_UPPER    = 1.30   # Spot grid upper bound as fraction of default_spot
GREEK_RISK_FREE     = 0.05   # Constant risk-free rate used in Greek computations

# ============================================================================
# BACKTESTER: Execution Defaults
# ============================================================================

BACKTEST_RISK_FREE_RATE = 0.05   # Annual risk-free rate used in Sharpe calculation
BACKTEST_EXPIRY_DAYS    = 30     # Synthetic option contract length (days)
BACKTEST_DEFAULT_EDGE   = 0.10   # Minimum edge to enter a position (10%)
