# Option Pricing Theory

## 1. The Stochastic Foundation: Geometric Brownian Motion (GBM)
We assume the underlying asset price $S_t$ follows a Geometric Brownian Motion. This means the percentage changes in price are independent and normally distributed. The Stochastic Differential Equation (SDE) is:
$$dS_t = r S_t dt + \sigma S_t dW_t$$
Where:
- $r$ is the risk-free interest rate (drift).
- $\sigma$ is the constant volatility.
- $dW_t$ is a Wiener process (random noise).

By applying Itô’s Lemma, we derive the exact solution used to simulate the price at maturity ($T$):
$$S_T = S_0 \exp\left( \left(r - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z \right)$$
Where $Z$ is a random variable sampled from a Standard Normal Distribution $N(0, 1)$.

## 2. The Analytical Solution: Black-Scholes-Merton
For European options, we have a closed-form solution. The price of a European Call Option ($C$) is calculated as:
$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$
To find $d_1$ and $d_2$, we use:
$$d_1 = \frac{\ln(S_0 / K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}$$
$$d_2 = d_1 - \sigma \sqrt{T}$$
- $N(\cdot)$ is the Cumulative Distribution Function (CDF) of the standard normal distribution.
- $K$ is the strike price.

## 3. The Numerical Approach: Monte Carlo Simulation
While Black-Scholes solves a differential equation, Monte Carlo simply "plays out" the future thousands of times.
1. **Simulate Paths**: Generate $N$ random terminal prices $S_T^{(i)}$ using the GBM formula.
2. **Calculate Payoffs**: For each path, determine the payoff at expiration:
   $$\text{Payoff}_i = \max(S_T^{(i)} - K, 0)$$
3. **Average and Discount**: The present value of the option is the average payoff discounted back to today:
   $$\hat{C} = e^{-rT} \left( \frac{1}{N} \sum_{i=1}^{N} \text{Payoff}_i \right)$$

## 4. Verification & Convergence
As $N$ (the number of simulations) approaches infinity, the Monte Carlo price $\hat{C}$ will converge to the Black-Scholes price $C$. This is guaranteed by the Law of Large Numbers.
In your project, the Standard Error (SE) of your MC estimate will be:
$$SE = \frac{SD(\text{Payoffs})}{\sqrt{N}}$$
A good simulation should result in a price where the Black-Scholes value falls within the 95% Confidence Interval ($MC \pm 1.96 \times SE$).
