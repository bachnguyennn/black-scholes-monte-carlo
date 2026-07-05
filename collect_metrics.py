#!/usr/bin/env python3
"""
Collect model metrics for the Final Report.
Run from project root:  PYTHONPATH=. python3 collect_metrics.py
"""
import sys, time, os
sys.path.insert(0, os.getcwd())

import numpy as np
from src.core.black_scholes import black_scholes_price
from src.core.gbm_engine import simulate_gbm
from src.core.heston_model import price_option_heston_fourier, simulate_heston
from src.core.jump_diffusion import simulate_jump_diffusion, merton_jump_price
from src.core.greeks import calculate_delta, calculate_gamma, calculate_vega, calculate_all_greeks
from src.core.scanner_engine import price_single_option_mc

print("=" * 70)
print("MODEL METRICS FOR FINAL REPORT")
print("=" * 70)

# ── Common parameters ──
S0, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.0, 0.20

# ══════════════════════════════════════════════════════════════════════
# 1. BSM ANALYTICAL BENCHMARK
# ══════════════════════════════════════════════════════════════════════
print("\n■ 1. BLACK-SCHOLES ANALYTICAL PRICES")
bs_call = black_scholes_price(S0, K, T, r, sigma, q=q, option_type='call')
bs_put  = black_scholes_price(S0, K, T, r, sigma, q=q, option_type='put')
print(f"  Call Price = {bs_call:.6f}")
print(f"  Put  Price = {bs_put:.6f}")
print(f"  Put-Call Parity: C - P = {bs_call - bs_put:.6f},  S0 - Ke^(-rT) = {S0 - K*np.exp(-r*T):.6f}")

# ══════════════════════════════════════════════════════════════════════
# 2. GBM MONTE CARLO CONVERGENCE
# ══════════════════════════════════════════════════════════════════════
print("\n■ 2. GBM MONTE CARLO CONVERGENCE")
print(f"  {'N paths':>12}  {'MC Price':>10}  {'Std Error':>10}  {'Abs Error':>10}  {'Time (s)':>10}")
print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

for n in [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]:
    np.random.seed(42)
    t0 = time.perf_counter()
    S_T = simulate_gbm(S0, T, r, sigma, n)
    elapsed = time.perf_counter() - t0
    payoffs = np.maximum(S_T - K, 0) * np.exp(-r * T)
    mc_price = payoffs.mean()
    se = payoffs.std() / np.sqrt(n)
    abs_err = abs(mc_price - bs_call)
    print(f"  {n:>12,}  {mc_price:>10.4f}  {se:>10.6f}  {abs_err:>10.6f}  {elapsed:>10.5f}")

# ══════════════════════════════════════════════════════════════════════
# 3. HESTON MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n■ 3. HESTON MODEL")
kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04
feller = 2 * kappa * theta > xi**2
print(f"  Parameters: κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}, V₀={v0}")
print(f"  Feller condition (2κθ > ξ²): {2*kappa*theta:.4f} > {xi**2:.4f} → {'SATISFIED' if feller else 'VIOLATED'}")

# Fourier pricing
t0 = time.perf_counter()
heston_call = price_option_heston_fourier(S0, K, T, r, v0, kappa, theta, xi, rho, q=q)
fourier_time = time.perf_counter() - t0
print(f"\n  Fourier price = {heston_call:.6f}  (time: {fourier_time*1000:.2f} ms)")

# MC pricing
for n_mc in [10_000, 50_000, 100_000]:
    np.random.seed(42)
    t0 = time.perf_counter()
    try:
        S_hest, V_hest = simulate_heston(S0, T, r, v0, kappa, theta, xi, rho, n_mc, n_steps=252, q=q)
        elapsed = time.perf_counter() - t0
        payoffs_h = np.maximum(S_hest - K, 0) * np.exp(-r * T)
        mc_h = payoffs_h.mean()
        se_h = payoffs_h.std() / np.sqrt(n_mc)
        print(f"  MC({n_mc:>7,}) = {mc_h:.4f} ± {se_h:.4f}  |err vs Fourier| = {abs(mc_h - heston_call):.4f}  (time: {elapsed:.3f}s)")
    except Exception as e:
        print(f"  MC({n_mc:>7,}) error: {e}")

print(f"\n  Heston vs BSM difference = {heston_call - bs_call:+.6f}")
print(f"  (Heston {'higher' if heston_call > bs_call else 'lower'} due to stochastic vol + negative ρ)")

# ══════════════════════════════════════════════════════════════════════
# 4. JUMP DIFFUSION
# ══════════════════════════════════════════════════════════════════════
print("\n■ 4. MERTON JUMP DIFFUSION")
lam, mu_j, sigma_j = 0.1, -0.05, 0.10
print(f"  Parameters: λ={lam}, μ_J={mu_j}, σ_J={sigma_j}")
kappa_j = np.exp(mu_j + 0.5 * sigma_j**2) - 1
print(f"  Jump compensator κ_J = {kappa_j:.6f}")

for n_jd in [10_000, 50_000, 100_000]:
    np.random.seed(42)
    t0 = time.perf_counter()
    try:
        result = simulate_jump_diffusion(S0, T, r, sigma, n_jd, jump_intensity=lam,
                                          jump_mean=mu_j, jump_std=sigma_j, n_steps=252)
        elapsed = time.perf_counter() - t0
        if isinstance(result, tuple):
            S_jd = result[0]
        else:
            S_jd = result
        payoffs_jd = np.maximum(S_jd - K, 0) * np.exp(-r * T)
        mc_jd = payoffs_jd.mean()
        se_jd = payoffs_jd.std() / np.sqrt(n_jd)
        print(f"  MC({n_jd:>7,}) = {mc_jd:.4f} ± {se_jd:.4f}  |diff vs BSM| = {abs(mc_jd - bs_call):.4f}  (time: {elapsed:.3f}s)")
    except Exception as e:
        print(f"  MC({n_jd:>7,}) error: {e}")

# Analytic Merton price (Poisson-weighted Black-Scholes sum) — exact ground
# truth the Monte Carlo above converges to.
mert = merton_jump_price(S0, K, T, r, sigma, 'call', jump_intensity=lam,
                         jump_mean=mu_j, jump_std=sigma_j, q=q)
mert0 = merton_jump_price(S0, K, T, r, sigma, 'call', jump_intensity=0.0,
                          jump_mean=0.0, jump_std=0.0, q=q)
print(f"\n  Analytic Merton price = {mert:.6f}  (exact; the MC above converges to this)")
print(f"  λ=0 limit = {mert0:.6f}  vs BSM {bs_call:.6f}  (diff {abs(mert0 - bs_call):.2e})")

# ══════════════════════════════════════════════════════════════════════
# 5. GREEKS (BSM Analytical)
# ══════════════════════════════════════════════════════════════════════
print("\n■ 5. BSM GREEKS (S₀=100, K=100, T=1, r=0.05, σ=0.20)")
try:
    greeks = calculate_all_greeks(S0, K, T, r, sigma, option_type='call')
    for name, val in greeks.items():
        if isinstance(val, (int, float)):
            print(f"  {name:>8s} = {val:+.6f}")
        else:
            print(f"  {name:>8s} = {val}")
except Exception as e:
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    delta = np.exp(-q*T) * norm.cdf(d1)
    gamma = np.exp(-q*T) * norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega  = S0 * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
    theta = (-(S0 * np.exp(-q*T) * norm.pdf(d1) * sigma) / (2*np.sqrt(T))
             - r*K*np.exp(-r*T)*norm.cdf(d2) + q*S0*np.exp(-q*T)*norm.cdf(d1))
    print(f"    Delta = {delta:+.6f}")
    print(f"    Gamma = {gamma:+.6f}")
    print(f"     Vega = {vega:+.6f}")
    print(f"    Theta = {theta:+.6f}")
    print(f"  (fallback scipy calc, greeks module err: {e})")

# ══════════════════════════════════════════════════════════════════════
# 6. JIT BENCHMARK COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n■ 6. PERFORMANCE BENCHMARKS (100,000 paths)")
n_bench = 100_000

_ = simulate_gbm(S0, T, r, sigma, 10)
t0 = time.perf_counter()
simulate_gbm(S0, T, r, sigma, n_bench)
gbm_t = time.perf_counter() - t0

try:
    _ = simulate_heston(S0, T, r, v0, kappa, theta, xi, rho, 10, n_steps=252, q=q)
    t0 = time.perf_counter()
    simulate_heston(S0, T, r, v0, kappa, theta, xi, rho, n_bench, n_steps=252, q=q)
    heston_mc_t = time.perf_counter() - t0
except:
    heston_mc_t = float('nan')

t0 = time.perf_counter()
for _ in range(100):
    price_option_heston_fourier(S0, K, T, r, v0, kappa, theta, xi, rho, q=q)
fourier_t = (time.perf_counter() - t0) / 100

try:
    _ = simulate_jump_diffusion(S0, T, r, sigma, 10, jump_intensity=lam, n_steps=252)
    t0 = time.perf_counter()
    simulate_jump_diffusion(S0, T, r, sigma, n_bench, jump_intensity=lam, n_steps=252)
    jd_t = time.perf_counter() - t0
except:
    jd_t = float('nan')

print(f"  {'Engine':<25s}  {'Time':>10s}  {'Speedup vs Heston MC':>22s}")
print(f"  {'-'*25}  {'-'*10}  {'-'*22}")
print(f"  {'GBM (vectorized)':<25s}  {gbm_t*1000:>8.2f}ms  {heston_mc_t/gbm_t:>20.1f}x")
print(f"  {'Heston Fourier (1 opt)':<25s}  {fourier_t*1000:>8.2f}ms  {heston_mc_t/fourier_t:>20.1f}x")
print(f"  {'Heston MC (252 steps)':<25s}  {heston_mc_t*1000:>8.2f}ms  {'1.0x':>22s}")
print(f"  {'Jump Diffusion MC':<25s}  {jd_t*1000:>8.2f}ms  {heston_mc_t/jd_t:>20.1f}x")

# ══════════════════════════════════════════════════════════════════════
# 7. DISTRIBUTION MOMENTS
# ══════════════════════════════════════════════════════════════════════
print("\n■ 7. TERMINAL DISTRIBUTION MOMENTS (100,000 paths)")
from scipy.stats import skew, kurtosis
np.random.seed(42)

S_gbm = simulate_gbm(S0, T, r, sigma, 100_000)
log_ret_gbm = np.log(S_gbm / S0)

try:
    np.random.seed(42)
    result = simulate_jump_diffusion(S0, T, r, sigma, 100_000, jump_intensity=lam,
                                      jump_mean=mu_j, jump_std=sigma_j, n_steps=252)
    S_jd_full = result[0] if isinstance(result, tuple) else result
    log_ret_jd = np.log(S_jd_full / S0)
    jd_ok = True
except:
    jd_ok = False

theory_mean = (r - q - 0.5*sigma**2)*T
theory_std = sigma * np.sqrt(T)

print(f"  {'Statistic':<20s}  {'GBM':>10s}  {'Jump Diff':>10s}  {'Theory (GBM)':>14s}")
print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*14}")
print(f"  {'Mean(log ret)':<20s}  {log_ret_gbm.mean():>10.4f}  {log_ret_jd.mean() if jd_ok else 'N/A':>10}  {theory_mean:>14.4f}")
print(f"  {'Std(log ret)':<20s}  {log_ret_gbm.std():>10.4f}  {log_ret_jd.std() if jd_ok else 'N/A':>10}  {theory_std:>14.4f}")
print(f"  {'Skewness':<20s}  {skew(log_ret_gbm):>10.4f}  {skew(log_ret_jd) if jd_ok else 'N/A':>10}  {'0.0000':>14s}")
print(f"  {'Excess Kurtosis':<20s}  {kurtosis(log_ret_gbm):>10.4f}  {kurtosis(log_ret_jd) if jd_ok else 'N/A':>10}  {'0.0000':>14s}")

# ══════════════════════════════════════════════════════════════════════
# 8. MONTE CARLO VARIANCE REDUCTION (antithetic + control variate)
# ══════════════════════════════════════════════════════════════════════
print("\n■ 8. MC VARIANCE REDUCTION (Jump Diffusion ATM call, equal path budget)")
n_vr = 20_000

def _naive_jd(seed):
    np.random.seed(seed)
    S_T, _ = simulate_jump_diffusion(S0, T, r, sigma, n_vr, jump_intensity=lam,
                                     jump_mean=mu_j, jump_std=sigma_j)
    return np.exp(-r * T) * np.maximum(S_T - K, 0).mean()

naive = np.array([_naive_jd(s) for s in range(25)])
vr = np.array([price_single_option_mc(S0, K, T, r, sigma, 'call', n_vr, lam, mu_j, sigma_j)['mc_price']
               for _ in range(25)])
factor = naive.std() / vr.std() if vr.std() > 0 else float('nan')
print(f"  naive MC std across seeds        = {naive.std():.4f}")
print(f"  antithetic + control-variate std = {vr.std():.4f}")
print(f"  => variance reduction = {factor:.1f}x std  ({factor**2:.0f}x variance / paths)")

print("\n" + "=" * 70)
print("COPY THESE METRICS INTO THE LATEX REPORT")
print("=" * 70)
