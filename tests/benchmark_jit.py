import time
import numpy as np

from src.core.gbm_engine import simulate_gbm
from src.core.jump_diffusion import simulate_jump_diffusion
from src.core.heston_model import simulate_heston

def run_benchmarks():
    print("Running JIT Benchmarks for Monte Carlo Engines...")
    n_sims = 100000
    S0, T, r, sigma = 100.0, 1.0, 0.05, 0.2
    
    # 1. GBM Benchmark
    print("\n--- GBM Benchmark ---")
    # Warmup
    simulate_gbm(S0, T, r, sigma, 10)
    
    start = time.perf_counter()
    S_T_gbm = simulate_gbm(S0, T, r, sigma, n_sims)
    end = time.perf_counter()
    gbm_time = end - start
    print(f"GBM {n_sims} paths generated in: {gbm_time:.5f} seconds")
    
    # 2. Jump Diffusion Benchmark
    print("\n--- Jump Diffusion Benchmark ---")
    # Warmup
    simulate_jump_diffusion(S0, T, r, sigma, 10, jump_intensity=0.1, n_steps=100)
    
    start = time.perf_counter()
    S_T_jd, crashes = simulate_jump_diffusion(S0, T, r, sigma, n_sims, jump_intensity=0.1, n_steps=100)
    end = time.perf_counter()
    jd_time = end - start
    print(f"Jump Diffusion {n_sims} paths (100 steps) generated in: {jd_time:.5f} seconds")
    
    # 3. Heston Benchmark
    print("\n--- Heston Benchmark ---")
    V0, kappa, theta, xi, rho = 0.04, 2.0, 0.04, 0.1, -0.7
    # Warmup
    simulate_heston(S0, T, r, V0, kappa, theta, xi, rho, 10, n_steps=100)
    
    start = time.perf_counter()
    S_T_h, V_T_h = simulate_heston(S0, T, r, V0, kappa, theta, xi, rho, n_sims, n_steps=100)
    end = time.perf_counter()
    h_time = end - start
    print(f"Heston {n_sims} paths (100 steps) generated in: {h_time:.5f} seconds")

    print("\nBenchmark Complete.")
    
    if gbm_time < 0.05 and jd_time < 0.5 and h_time < 0.5:
        print("Performance targets met!")
    else:
        print("Some performance targets not met. Please review JIT optimizations.")

if __name__ == "__main__":
    run_benchmarks()
