
import unittest
import numpy as np
import sys
import os

# Ensure src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm

class TestConvergenceValidation(unittest.TestCase):
    
    def test_statistical_convergence(self):
        """
        Step 2: Statistical Convergence Test
        Assert that Monte Carlo price is within 3 Standard Errors of Black-Scholes price.
        """
        S0 = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.2
        n_sims = 100000  # High number of paths for accuracy
        
        # 1. Black-Scholes Price
        bs_price = black_scholes_price(S0, K, T, r, sigma, option_type='call')
        
        # 2. Monte Carlo Price
        S_T = simulate_gbm(S0, T, r, sigma, n_sims)
        payoffs = np.maximum(S_T - K, 0)
        mc_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Standard Error
        std_error = np.std(payoffs) / np.sqrt(n_sims) * np.exp(-r * T)
        
        print(f"\n--- Convergence Test ---")
        print(f"BS Price: {bs_price:.4f}")
        print(f"MC Price: {mc_price:.4f}")
        print(f"Standard Error: {std_error:.4f}")
        print(f"Difference: {abs(mc_price - bs_price):.4f}")
        print(f"Z-Score: {abs(mc_price - bs_price) / std_error:.4f}")
        
        # 3. Statistical Assertion (3 Sigma Rule)
        self.assertTrue(
            abs(mc_price - bs_price) < 3 * std_error,
            f"MC Price {mc_price:.4f} is NOT within 3 Standard Errors of BS Price {bs_price:.4f}"
        )

    def test_edge_cases(self):
        """
        Step 4: Edge Case Stress Test
        """
        print(f"\n--- Edge Case Test ---")
        
        # Case 1: Volatility near 0 (Option intrinsic value)
        # Call Option: max(S0 - K*exp(-rT), 0) ?? No, if vol is 0
        # Formula: S_T = S0 * exp(rT). Call Payoff = max(S0*exp(rT) - K, 0). Discounted = max(S0 - K*exp(-rT), 0)
        S0, K, T, r = 100, 100, 1.0, 0.05
        sigma_low = 1e-5
        
        bs_low_vol = black_scholes_price(S0, K, T, r, sigma_low, 'call')
        expected_intrinsic = max(S0 - K * np.exp(-r * T), 0) # For ATM, this is roughly S0 - K*exp(-rT)
        
        print(f"Low Volatility BS: {bs_low_vol:.4f}, Intrinsic: {expected_intrinsic:.4f}")
        self.assertAlmostEqual(bs_low_vol, expected_intrinsic, places=4, msg="Low Volatility failed")

        # Case 2: Deep OTM (Price ~ 0)
        S_otm, K_otm = 50, 100 # Deep OTM Call
        bs_otm = black_scholes_price(S_otm, K_otm, T, r, 0.2, 'call')
        print(f"Deep OTM Price: {bs_otm:.4e}")
        self.assertTrue(bs_otm < 0.01, "Deep OTM option should be near zero")

        # Case 3: Small T (Price converges to payoff)
        # Note: ATM options have max time value. Even at T=1e-5 (approx 5 mins), 
        # there is non-zero value (~0.025). We strictly check it's small.
        T_tiny = 1e-5 
        bs_tiny_t = black_scholes_price(S0, K, T_tiny, r, 0.2, 'call')
        payoff = max(S0 - K, 0)
        print(f"Small T Price: {bs_tiny_t:.4f}, Payoff: {payoff:.4f}")
        self.assertAlmostEqual(bs_tiny_t, payoff, delta=0.05, msg="Small T failed") 

if __name__ == '__main__':
    unittest.main()
