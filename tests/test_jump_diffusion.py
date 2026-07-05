"""
test_jump_diffusion.py

Comprehensive test suite for Jump Diffusion engine optimization.
Tests performance, accuracy, and mathematical correctness.
"""

import numpy as np
import pytest
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.jump_diffusion import simulate_jump_diffusion, simulate_jump_diffusion_paths
from src.core.gbm_engine import simulate_gbm
from src.core.scanner_engine import price_single_option_mc
from src.core.black_scholes import black_scholes_price


class TestPerformance:
    """Test A: Performance Benchmark - Verify vectorization speedup"""
    
    def test_performance_100k_paths(self):
        """Verify 100k simulations complete in < 0.1 seconds"""
        # Warm up the Numba JIT first: the initial call pays a one-off
        # compilation cost that would otherwise dominate this timing and make
        # the benchmark flaky. We measure steady-state runtime, not compile time.
        simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.2, n_sims=1000,
            jump_intensity=0.1, jump_mean=-0.05, jump_std=0.03
        )

        start = time.time()
        S_T, crash_mask = simulate_jump_diffusion(
            S0=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            n_sims=100000,
            jump_intensity=0.1,
            jump_mean=-0.05,
            jump_std=0.03
        )
        elapsed = time.time() - start
        
        # Verify output shapes
        assert S_T.shape == (100000,), "Terminal prices shape mismatch"
        assert crash_mask.shape == (100000,), "Crash mask shape mismatch"
        
        # Performance requirement
        assert elapsed < 0.1, f"Too slow: {elapsed:.3f}s (target: <0.1s)"
        
        print(f"✓ Performance test passed: {elapsed:.4f}s for 100k paths")
    
    def test_performance_paths_function(self):
        """Verify paths function is also optimized"""
        start = time.time()
        paths, crash_mask = simulate_jump_diffusion_paths(
            S0=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            n_paths=10000,  # Smaller for paths (more memory intensive)
            jump_intensity=0.1,
            n_steps=100
        )
        elapsed = time.time() - start
        
        assert paths.shape == (10000, 101), "Paths shape mismatch"
        assert elapsed < 1.0, f"Paths function too slow: {elapsed:.3f}s"
        
        print(f"✓ Paths performance: {elapsed:.4f}s for 10k paths × 100 steps")


class TestConvergence:
    """Test B: Convergence to GBM - When λ=0, Jump Diffusion should match GBM"""
    
    def test_convergence_to_gbm_zero_intensity(self):
        """When jump_intensity=0, Jump Diffusion should match GBM"""
        np.random.seed(42)  # For reproducibility
        
        # Jump Diffusion with λ=0 (no jumps)
        S_T_jump, crash_mask = simulate_jump_diffusion(
            S0=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            n_sims=50000,
            jump_intensity=0.0  # No jumps
        )
        
        # Verify no crashes occurred
        assert np.sum(crash_mask) == 0, "Crashes occurred with λ=0"
        
        # Reset seed for fair comparison
        np.random.seed(42)
        S_T_gbm = simulate_gbm(
            S0=100,
            T=1.0,
            r=0.05,
            sigma=0.2,
            n_sims=50000
        )
        
        # Both estimators must converge to the analytic risk-neutral mean
        # E[S_T] = S0 * exp(r*T). Comparing each against this closed-form
        # value is the meaningful convergence check; comparing the two Monte
        # Carlo means directly is not, since with lambda=0 the two functions
        # still consume independent RNG streams (JD interleaves randn/poisson
        # per path, GBM draws one batched randn), so their sampling errors are
        # independent and a sub-0.1% agreement is coincidental, not required.
        mean_jump = np.mean(S_T_jump)
        mean_gbm = np.mean(S_T_gbm)
        analytic_mean = 100 * np.exp(0.05 * 1.0)

        jump_err = abs(mean_jump - analytic_mean) / analytic_mean
        gbm_err = abs(mean_gbm - analytic_mean) / analytic_mean

        # ~4x the Monte Carlo standard error of the mean for these settings.
        assert jump_err < 0.005, f"JD(lambda=0) mean off analytic by {jump_err:.4%}"
        assert gbm_err < 0.005, f"GBM mean off analytic by {gbm_err:.4%}"

        print(f"✓ Convergence: Jump={mean_jump:.4f}, GBM={mean_gbm:.4f}, "
              f"analytic={analytic_mean:.4f}")
    
    def test_convergence_statistics(self):
        """Verify that statistical properties match when λ=0"""
        np.random.seed(123)
        
        S_T_jump, _ = simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            n_sims=50000, jump_intensity=0.0
        )
        
        np.random.seed(123)
        S_T_gbm = simulate_gbm(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            n_sims=50000
        )
        
        # Compare standard deviations
        std_jump = np.std(S_T_jump)
        std_gbm = np.std(S_T_gbm)
        std_diff = abs(std_jump - std_gbm) / std_gbm
        
        assert std_diff < 0.02, f"Std devs differ by {std_diff:.4%}"
        
        print(f"✓ Statistics match: Jump_std={std_jump:.4f}, GBM_std={std_gbm:.4f}")


class TestCrashProbability:
    """Test C: Crash Probability Validation - Verify Poisson process correctness"""
    
    def test_crash_probability_matches_theory(self):
        """Verify crash probability matches theoretical expectation"""
        lambda_val = 0.1
        T = 1.0
        
        _, crash_mask = simulate_jump_diffusion(
            S0=100,
            T=T,
            r=0.05,
            sigma=0.2,
            n_sims=100000,
            jump_intensity=lambda_val
        )
        
        observed_prob = np.sum(crash_mask) / len(crash_mask)
        expected_prob = 1 - np.exp(-lambda_val * T)  # ~9.52%
        
        diff = abs(observed_prob - expected_prob)
        
        # Should be within ±2%
        assert diff < 0.02, f"Crash prob: observed={observed_prob:.4f}, expected={expected_prob:.4f}, diff={diff:.4f}"
        
        print(f"✓ Crash probability: observed={observed_prob:.4%}, expected={expected_prob:.4%}, diff={diff:.4%}")
    
    def test_crash_probability_scaling(self):
        """Verify crash probability scales correctly with λ"""
        test_cases = [
            (0.05, 1.0, 0.04877),  # λ=0.05, T=1 → ~4.88%
            (0.20, 1.0, 0.18127),  # λ=0.20, T=1 → ~18.13%
            (0.10, 2.0, 0.18127),  # λ=0.10, T=2 → ~18.13%
        ]
        
        for lambda_val, T, expected_prob in test_cases:
            _, crash_mask = simulate_jump_diffusion(
                S0=100, T=T, r=0.05, sigma=0.2,
                n_sims=50000,
                jump_intensity=lambda_val
            )
            
            observed_prob = np.sum(crash_mask) / len(crash_mask)
            diff = abs(observed_prob - expected_prob)
            
            assert diff < 0.02, f"λ={lambda_val}, T={T}: diff={diff:.4f}"
            
        print(f"✓ Crash probability scales correctly with λ and T")


class TestMathematicalCorrectness:
    """Additional tests for mathematical correctness"""
    
    def test_terminal_prices_positive(self):
        """Ensure all terminal prices are positive"""
        S_T, _ = simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            n_sims=10000, jump_intensity=0.2,
            jump_mean=-0.10, jump_std=0.05
        )
        
        assert np.all(S_T > 0), "Some terminal prices are non-positive"
        print(f"✓ All terminal prices positive: min={np.min(S_T):.4f}")
    
    def test_crash_mask_boolean(self):
        """Verify crash mask is boolean array"""
        _, crash_mask = simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            n_sims=1000, jump_intensity=0.1
        )
        
        assert crash_mask.dtype == bool, "Crash mask should be boolean"
        assert np.all((crash_mask == 0) | (crash_mask == 1)), "Crash mask contains non-boolean values"
        
        print(f"✓ Crash mask is boolean: {np.sum(crash_mask)} crashes in 1000 paths")
    
    def test_paths_start_at_S0(self):
        """Verify all paths start at initial price"""
        S0 = 100
        paths, _ = simulate_jump_diffusion_paths(
            S0=S0, T=1.0, r=0.05, sigma=0.2,
            n_paths=1000, jump_intensity=0.1, n_steps=50
        )
        
        assert np.allclose(paths[:, 0], S0), "Paths don't start at S0"
        print(f"✓ All paths start at S0={S0}")


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_volatility(self):
        """Test with zero continuous volatility (only jumps)"""
        S_T, _ = simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.0,  # No continuous vol
            n_sims=10000, jump_intensity=0.1
        )
        
        assert len(S_T) == 10000
        print(f"✓ Zero volatility case: mean={np.mean(S_T):.4f}")
    
    def test_high_jump_intensity(self):
        """Test with very high jump intensity"""
        S_T, crash_mask = simulate_jump_diffusion(
            S0=100, T=1.0, r=0.05, sigma=0.2,
            n_sims=10000, jump_intensity=2.0  # Expect ~6.4 jumps per path
        )
        
        crash_prob = np.sum(crash_mask) / len(crash_mask)
        assert crash_prob > 0.8, "High λ should cause most paths to crash"
        
        print(f"✓ High jump intensity: {crash_prob:.2%} of paths crashed")


class TestVarianceReduction:
    """Antithetic + control-variate pricing in price_single_option_mc."""

    def test_matches_black_scholes_when_no_jumps(self):
        # With jumps switched off, the jump-diffusion payoff equals the
        # control payoff, so the estimator must collapse onto the exact
        # Black-Scholes price.
        S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.2
        out = price_single_option_mc(
            S0, K, T, r, sigma, 'call', n_sims=40000,
            jump_intensity=0.0, jump_mean=0.0, jump_std=0.0
        )
        bs = black_scholes_price(S0, K, T, r, sigma, option_type='call')
        assert abs(out['mc_price'] - bs) < 1e-6
        assert out['std_error'] < 1e-6

    def test_variance_is_reduced_versus_naive(self):
        # Same path budget: the variance-reduced estimator's spread across
        # seeds should be several times tighter than naive Monte Carlo.
        S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.2
        lam, jm, js = 0.1, -0.05, 0.03
        n = 8000

        def naive(seed):
            np.random.seed(seed)
            S_T, _ = simulate_jump_diffusion(
                S0, T, r, sigma, n,
                jump_intensity=lam, jump_mean=jm, jump_std=js
            )
            return np.exp(-r * T) * np.maximum(S_T - K, 0.0).mean()

        naive_spread = np.std([naive(s) for s in range(15)])
        vr_spread = np.std([
            price_single_option_mc(S0, K, T, r, sigma, 'call', n, lam, jm, js)['mc_price']
            for _ in range(15)
        ])
        assert vr_spread < naive_spread / 3.0

    def test_unbiased_against_high_accuracy_reference(self):
        S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.03, 0.2
        lam, jm, js = 0.1, -0.05, 0.03
        np.random.seed(0)
        S_ref, _ = simulate_jump_diffusion(
            S0, T, r, sigma, 2_000_000,
            jump_intensity=lam, jump_mean=jm, jump_std=js
        )
        ref = np.exp(-r * T) * np.maximum(S_ref - K, 0.0).mean()
        price = price_single_option_mc(S0, K, T, r, sigma, 'call', 40000, lam, jm, js)['mc_price']
        assert abs(price - ref) < 0.1


if __name__ == "__main__":
    # Run all tests with verbose output
    pytest.main([__file__, "-v", "-s"])
