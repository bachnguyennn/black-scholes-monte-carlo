# Project Scope & Component Architecture

## 1. Project Scope
The goal of this project is to build a robust **Financial Option Pricing Engine** that compares two fundamental methods of calculating the fair value of European Call Options:

1.  **Black-Scholes Model**: The theoretical "exact" price derived from stochastic calculus (analytical solution). This serves as the **benchmark/truth**.
2.  **Monte Carlo Simulation**: A numerical method that simulates thousands of possible future price paths for the underlying asset to estimate the fair value. This demonstrates the **Law of Large Numbers**—as the number of simulations ($N$) increases, the Monte Carlo price converges to the Black-Scholes price.

This project is designed to be modular, separating the mathematical "logic" (Quants) from the "presentation" (Frontend).

---

## 2. Component Architecture

### A. The Core Logic (`src/core/`)
This module contains the pure mathematical functions. It has zero knowledge of the website or user interface.

#### 1. `black_scholes.py` (The Benchmark)
*   **Role**: Calculates the theoretical price.
*   **Math**: Uses the standard Cumulative Distribution Function (CDF) of the Normal distribution.
*   **Key Inputs**: $S_0$ (Price), $K$ (Strike), $T$ (Time), $r$ (Rate), $\sigma$ (Volatility).
*   **Why it's here**: To provide a target value to check the accuracy of our simulation.

#### 2. `gbm_engine.py` (The Engine)
*   **Role**: Simulates the chaotic future of the stock market.
*   **Math**: Uses **Geometric Brownian Motion (GBM)**.
    *   Formula: $S_T = S_0 \cdot \exp((r - 0.5\sigma^2)T + \sigma\sqrt{T}Z)$
    *   Where $Z$ is a random number drawn from a standard normal distribution.
*   **Optimization**: Uses `numpy` vectorization to simulate 100,000+ paths instantly without slow loops.

### B. The User Interface (`src/web/`)
This module connects the math to the user.

#### 3. `app.py` (The Dashboard)
*   **Role**: An interactive web dashboard utilizing **Streamlit**.
*   **Features**:
    *   **Sidebar**: Allows real-time adjustment of market parameters (Volatility, Risk-free rate, etc.).
    *   **Visualization**:
        *   **Path Plot**: Shows 50 random possible future paths for the stock, visualizing market uncertainty.
        *   **Histogram**: Shows the distribution of where the stock price effectively "landed" after all simulations.
    *   **Real-time Pricing**: Automatically re-calculates prices as you slide the widgets.

---

## 3. Data Flow

1.  **User** adjusts a slider (e.g., increases Volatility $\sigma$).
2.  **App** sends new $\sigma$ to:
    *   `black_scholes_price()` $\rightarrow$ Returns **\$10.50** (Example)
    *   `simulate_gbm()` $\rightarrow$ Returns array of 10,000 terminal prices.
3.  **App** calculates average of those 10,000 prices $\rightarrow$ **\$10.52**.
4.  **App** updates the UI to show the prices side-by-side and re-draws the charts.
