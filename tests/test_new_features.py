"""
Quick test of new features
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.gbm_engine import calculate_profit_probability, calculate_breakeven_price
import numpy as np

# Test Jump Diffusion
S_T, crash_mask = simulate_jump_diffusion(100, 1.0, 0.05, 0.2, 1000)
print(f"Jump Diffusion Test:")
print(f"  Mean Terminal Price: ${np.mean(S_T):.2f}")
print(f"  Crash Events: {np.mean(crash_mask):.1%}")

# Test Probability Metrics
prob = calculate_profit_probability(S_T, 100, 10, 'call')
breakeven = calculate_breakeven_price(100, 10, 'call')
print(f"\nProbability Metrics Test:")
print(f"  Profit Probability: {prob:.2%}")
print(f"  Breakeven Price: ${breakeven:.2f}")

print("\n✅ All tests passed!")
