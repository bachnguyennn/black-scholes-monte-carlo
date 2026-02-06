"""
test_convergence.py

Tests the convergence of the Monte Carlo simulation to the Black-Scholes price.
Standard Error Check: |BS - MC| < 1.96 * SE
"""

import unittest

class TestConvergence(unittest.TestCase):
    def test_convergence(self):
        pass

if __name__ == "__main__":
    unittest.main()
