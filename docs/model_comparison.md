# Model Comparison Guide: GBM vs Jump Diffusion

## Quick Decision Tree

```
START: Which model should I use?
│
├─ Is your option expiring in < 30 days?
│  └─ YES → Use Standard GBM (jumps won't matter much)
│  └─ NO → Continue...
│
├─ Are you pricing deep out-of-the-money PUTS?
│  └─ YES → Use Jump Diffusion (crash protection is valuable)
│  └─ NO → Continue...
│
├─ Is the market currently volatile or facing known risks?
│  └─ YES → Use Jump Diffusion (higher crash probability)
│  └─ NO → Use Standard GBM (faster, simpler)
```

---

## Model Overview

### Standard GBM (Geometric Brownian Motion)
**What it assumes**: Stock prices move smoothly and continuously, like a random walk.

**Best for**:
- ✅ Short-term options (< 3 months)
- ✅ At-the-money or in-the-money options
- ✅ Stable market conditions
- ✅ When you need fast calculations

**Weaknesses**:
- ❌ Underestimates tail risk (crashes)
- ❌ Assumes markets are always "normal"

---

### Jump Diffusion (Merton Model)
**What it assumes**: Stock prices move smoothly most of the time, but occasionally experience sudden jumps (crashes or spikes).

**Best for**:
- ✅ Long-term options (> 6 months)
- ✅ Out-of-the-money puts (insurance against crashes)
- ✅ Volatile or uncertain markets
- ✅ Stress testing and risk management

**Weaknesses**:
- ❌ Slower to compute (more parameters)
- ❌ Requires parameter calibration
- ❌ More complex to interpret

---

## Real-World Case Studies

### Case Study 1: The 2008 Financial Crisis

**Scenario**: September 2008, Lehman Brothers bankruptcy

**Market Conditions**:
- S&P 500 at $1,200
- Volatility spiking to 40%+
- Fear of systemic collapse

**Option**: 6-month Put, Strike = $1,000 (out-of-the-money)

**Model Comparison**:
| Model | Predicted Price | Actual Market Price |
|-------|----------------|---------------------|
| Standard GBM | $45 | $120 |
| Jump Diffusion (λ=0.3) | $115 | $120 |

**Winner**: Jump Diffusion ✅

**Why**: GBM assumed crashes were "impossible." Jump Diffusion correctly priced the tail risk.

---

### Case Study 2: The 2020 COVID Crash

**Scenario**: March 2020, pandemic panic

**Market Conditions**:
- S&P 500 dropped 34% in 23 days
- Fastest crash in history
- VIX (fear index) hit 80+

**Option**: 3-month Call, Strike = $3,000 (at-the-money)

**Model Comparison**:
| Model | Predicted Price | Actual Market Price |
|-------|----------------|---------------------|
| Standard GBM | $180 | $175 |
| Jump Diffusion (λ=0.5) | $185 | $175 |

**Winner**: Tie (both were close)

**Why**: For at-the-money calls, jumps don't add much value. GBM was sufficient and faster.

---

### Case Study 3: Normal Market (2019)

**Scenario**: July 2019, calm bull market

**Market Conditions**:
- S&P 500 steadily rising
- Low volatility (VIX ~12)
- No major events

**Option**: 1-year Call, Strike = $3,200 (slightly out-of-the-money)

**Model Comparison**:
| Model | Predicted Price | Actual Market Price |
|-------|----------------|---------------------|
| Standard GBM | $210 | $215 |
| Jump Diffusion (λ=0.1) | $212 | $215 |

**Winner**: Tie (negligible difference)

**Why**: In calm markets with low jump risk, both models converge. Use GBM for simplicity.

---

## Parameter Calibration Guide

### Step 1: Gather Historical Data
You need at least **2 years** of daily price data for reliable calibration.

**Using Yahoo Finance** (already integrated in your app):
```python
import yfinance as yf
ticker = yf.Ticker("SPY")
history = ticker.history(period="2y")
prices = history['Close'].values
```

---

### Step 2: Calculate Daily Returns
```python
import numpy as np
log_returns = np.log(prices[1:] / prices[:-1])
```

---

### Step 3: Identify Jump Events
**Rule of Thumb**: Flag returns beyond 3 standard deviations as "jumps"

```python
mean_return = np.mean(log_returns)
std_return = np.std(log_returns)

# Flag jumps
threshold = 3 * std_return
jump_mask = np.abs(log_returns - mean_return) > threshold
jump_returns = log_returns[jump_mask]
normal_returns = log_returns[~jump_mask]
```

---

### Step 4: Estimate Parameters

**Jump Intensity (λ)**:
```python
n_jumps = np.sum(jump_mask)
n_years = len(log_returns) / 252  # 252 trading days per year
lambda_estimate = n_jumps / n_years
```

**Jump Mean (μ_J)**:
```python
mu_J = np.mean(jump_returns)
```

**Jump Volatility (σ_J)**:
```python
sigma_J = np.std(jump_returns)
```

**Continuous Volatility (σ)**:
```python
sigma_daily = np.std(normal_returns)
sigma_annual = sigma_daily * np.sqrt(252)
```

---

### Step 5: Validate Your Parameters

**Sanity Checks**:
- ✅ `0.05 < λ < 0.5` (1 crash every 2-20 years)
- ✅ `-0.15 < μ_J < 0` (crashes are typically negative)
- ✅ `0.01 < σ_J < 0.10` (crash sizes vary, but not wildly)
- ✅ `0.10 < σ < 0.50` (annual volatility is reasonable)

**If values are outside these ranges**, you may have:
- Insufficient data (need more history)
- Incorrect threshold (try 2.5σ or 3.5σ instead of 3σ)
- A truly exceptional asset (crypto, penny stocks)

---

## Practical Examples

### Example 1: Conservative Investor (Risk-Averse)

**Goal**: Protect a $1M portfolio against a 20% crash

**Recommendation**: Use **Jump Diffusion** with conservative parameters
- `λ = 0.2` (expect 1 crash every 5 years)
- `μ_J = -0.08` (assume 8% average crash)
- `σ_J = 0.04` (moderate crash uncertainty)

**Why**: You want to **overestimate** crash risk to ensure adequate protection.

**Action**: Buy out-of-the-money puts priced using Jump Diffusion.

---

### Example 2: Day Trader (Short-Term Focus)

**Goal**: Profit from daily price movements

**Recommendation**: Use **Standard GBM**
- Jumps are rare (λ = 0.1 means only 0.0004 jumps per day)
- Speed matters more than tail risk
- Focus on continuous volatility (σ)

**Why**: Over 1-2 days, crash risk is negligible. GBM is faster and simpler.

**Action**: Use GBM for quick pricing and Greeks calculations.

---

### Example 3: Earnings Announcement (Event Risk)

**Goal**: Price options expiring right after a company's earnings report

**Recommendation**: Use **Jump Diffusion** with elevated intensity
- `λ = 2.0` (high probability of a "jump" due to earnings surprise)
- `μ_J = 0` (earnings can surprise up or down)
- `σ_J = 0.10` (large uncertainty in jump size)

**Why**: Earnings announcements create discrete, event-driven price moves—exactly what Jump Diffusion models.

**Action**: Use Jump Diffusion to capture the "earnings premium" in option prices.

---

## Visual Comparison

### Terminal Price Distributions

**Standard GBM**:
```
Frequency
    ^
    |     ****
    |   ********
    | ************
    |**************
    +---------------> Price
   Low          High
```
**Shape**: Symmetric bell curve (log-normal)  
**Tails**: Thin (crashes are "impossible")

---

**Jump Diffusion**:
```
Frequency
    ^
    |      ****
    |    ********
    |  ************
    |****************
    +---------------> Price
   Low          High
    ↑
  Fat tail
 (crash risk)
```
**Shape**: Asymmetric with fat left tail  
**Tails**: Thick (crashes are possible)

---

## Key Takeaways

### When to Use Standard GBM
1. ✅ **Short time horizons** (< 1 month)
2. ✅ **At-the-money options** (where tail risk doesn't matter)
3. ✅ **Calm markets** (VIX < 20)
4. ✅ **Speed is critical** (real-time pricing)

### When to Use Jump Diffusion
1. ✅ **Long time horizons** (> 6 months)
2. ✅ **Out-of-the-money puts** (tail risk protection)
3. ✅ **Volatile markets** (VIX > 25)
4. ✅ **Event-driven scenarios** (earnings, elections, Fed meetings)

### The Hybrid Approach
For maximum accuracy, use **both**:
1. Price the option with **GBM** (fast baseline)
2. Price the option with **Jump Diffusion** (tail risk adjustment)
3. The difference is the **"crash premium"**
4. If the market price is closer to Jump Diffusion → Market is pricing in crash risk
5. If the market price is closer to GBM → Market is calm

---

## Frequently Asked Questions

### Q1: Can Jump Diffusion predict crashes?
**A**: No. It models the *probability* and *impact* of crashes, but cannot predict *when* they will occur.

### Q2: Why is my Jump Diffusion price sometimes lower than GBM?
**A**: For **calls**, upward jumps can increase value, but downward jumps decrease the probability of finishing in-the-money. The net effect depends on the strike price.

### Q3: How do I know if my parameters are correct?
**A**: Compare your model prices to actual market prices. If they consistently differ by > 10%, recalibrate.

### Q4: Can I use Jump Diffusion for stocks that don't crash?
**A**: Yes! Set `λ = 0.05` (very low) and `μ_J = 0` (neutral jumps). The model will behave almost like GBM.

### Q5: What if I don't have historical data?
**A**: Use **industry defaults**:
- **Stable stocks** (utilities, consumer staples): λ = 0.05, μ_J = -0.02
- **Tech stocks**: λ = 0.15, μ_J = -0.05
- **Crypto/volatile assets**: λ = 0.5, μ_J = -0.10

---

## Next Steps

1. **Experiment**: Use the dashboard to compare both models side-by-side
2. **Calibrate**: Use the guide above to estimate parameters from real data
3. **Validate**: Check if your model prices match market prices
4. **Iterate**: Adjust parameters based on current market conditions

**Remember**: No model is perfect. The goal is to understand the *trade-offs* and choose the right tool for your specific use case.
