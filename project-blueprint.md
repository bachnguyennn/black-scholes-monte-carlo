# Master Blueprint: European Options Pricing Engine

## 1. API Contract

All functions must use NumPy for vectorization. No raw for-loops are permitted in simulation or math modules.

### `src/core/black_scholes.py`
```python
def black_scholes_price(S0, K, T, r, sigma, option_type='call'):
    """
    Calculates the analytical Black-Scholes price for a European option.
    
    Inputs:
        S0: Current asset price (float or np.ndarray)
        K: Strike price (float or np.ndarray)
        T: Time to maturity (float or np.ndarray)
        r: Risk-free rate (float or np.ndarray)
        sigma: Volatility (float or np.ndarray)
        option_type: 'call' or 'put' (str)
        
    Output:
        price: NumPy array or scalar of option prices
    """
```

### `src/core/gbm_engine.py`
```python
def simulate_gbm(S0, T, r, sigma, n_sims):
    """
    Simulates terminal asset prices using Geometric Brownian Motion.
    
    Inputs:
        S0: Initial price (float)
        T: Time to maturity (float)
        r: Risk-free rate (float)
        sigma: Volatility (float)
        n_sims: Number of simulations (int)
        
    Output:
        S_T: NumPy array of shape (n_sims,) representing terminal prices
    """
```

## 2. Data Flow

1.  **Frontend (`src/web/app.py`)**:
    *   Captures user inputs: $S_0, K, T, r, \sigma, N$ (simulations).
    *   Calls `black_scholes_price` to get the theoretical benchmark.
    *   Calls `simulate_gbm` to get $N$ terminal price paths.
2.  **Logic**:
    *   Calculate payoffs: $P_i = \max(S_T^{(i)} - K, 0)$ for calls.
    *   Calculate MC Price: $\hat{C} = e^{-rT} \times \text{mean}(P)$.
    *   Calculate Standard Error: $SE = \text{std}(P) / \sqrt{N} \times e^{-rT}$.
3.  **Output**:
    *   Display BS Price vs. MC Price.
    *   Show Convergence Plot (Price vs. $N$).

## 3. Validation Plan: `tests/test_convergence.py`

*   **Accuracy Threshold**: The Black-Scholes price must fall within the 95% Confidence Interval ($\hat{C} \pm 1.96 \times SE$).
*   **Convergence Rate**: Standard Error must scale with $1/\sqrt{N}$.
*   **Edge Cases**: Test with varied $K$ (in-the-money, at-the-money, out-of-the-money).
