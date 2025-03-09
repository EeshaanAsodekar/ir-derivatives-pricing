import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
from pathlib import Path

# Ensure we can import from our project
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cir import simulate_cir

# === Market Data ===
market_bond_prices = {
    "1M": 0.9957,
    "3M": 0.9873,
    "6M": 0.9745,
    "1Y": 0.9562
}

# Convert from e.g. "1M" → 1/12 years
maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0}
market_maturities = np.array([maturity_map[m] for m in market_bond_prices.keys()])
market_prices = np.array(list(market_bond_prices.values()))

# === Initial Guess & Model Params ===
initial_guess = [0.05, 0.04, 0.08]  # (a, b, sigma)
r0 = 0.052  # Short rate at t=0
num_paths = 10000
dt = 0.01   # Use a slightly larger dt for stability

# === Constraints ===
# We'll ensure: a > 0, b > 0, sigma > 0, sigma < 0.3
def constraint_a(params):
    a, _, _ = params
    return a  # Must be > 0

def constraint_b(params):
    _, b, _ = params
    return b  # Must be > 0

def constraint_sigma_positive(params):
    _, _, sigma = params
    return sigma  # Must be > 0

def constraint_sigma_upper(params):
    _, _, sigma = params
    return 0.3 - sigma  # Must be >= 0

def constraint_b_upper(params):
    _, b, _ = params
    return 0.07 - b  # Must be >= 0


constraints = [
    {'type': 'ineq', 'fun': constraint_a},            # a > 0
    {'type': 'ineq', 'fun': constraint_b},            # b > 0
    {'type': 'ineq', 'fun': constraint_sigma_positive}, # sigma > 0
    {'type': 'ineq', 'fun': constraint_sigma_upper},   # sigma < 0.3
    {'type': 'ineq', 'fun': constraint_b_upper}   # b < 0.07
]

# === Objective Function (log-price SSE) ===
def objective_function(params):
    """
    Computes sum of squared errors on log bond prices:
        SSE = sum( [ log(P_sim(t_i)) - log(P_mkt(t_i)) ]^2 ).
    """
    a, b, sigma = params
    T = max(market_maturities)  # 1 year

    # Simulate the short-rate process with Euler-Maruyama
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # Compute the simulated bond price for each market maturity
    simulated_prices = []
    for key in market_bond_prices.keys():
        m = maturity_map[key]
        num_steps = int(m / dt)
        # Approximate integral
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        # Discount factor
        P_t = np.exp(-integral_rt).mean()
        simulated_prices.append(P_t)

    simulated_prices = np.array(simulated_prices)
    
    # Convert to log prices & compute SSE
    # Avoid log(0) by clipping
    eps = 1e-12
    log_sim = np.log(np.clip(simulated_prices, eps, 1.0))
    log_mkt = np.log(np.clip(market_prices, eps, 1.0))
    sse_log = np.sum((log_sim - log_mkt) ** 2)
    return sse_log

# === Perform Optimization (SLSQP) with constraints ===
result = minimize(objective_function, 
                  initial_guess, 
                  method="SLSQP", 
                  constraints=constraints,
                  options={"maxiter": 500, "ftol": 1e-12})

a_opt, b_opt, sigma_opt = result.x
print("Optimal Parameters (SLSQP, log-price SSE):")
print(f"  a={a_opt:.6f}, b={b_opt:.6f}, sigma={sigma_opt:.6f}")
print(f"  Final SSE (log-prices)={result.fun:.6f}")

# === Plot the best-fit vs. market data
def compute_bond_prices_at_market_maturities(a, b, sigma, r0, market_maturities, dt, num_paths):
    # Resimulate the process
    rates_df = simulate_cir(a, b, sigma, r0, max(market_maturities), dt, num_paths)

    bond_prices = {}
    for key in market_bond_prices.keys():
        m = maturity_map[key]
        num_steps = int(m / dt)
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        P_t = np.exp(-integral_rt).mean()
        bond_prices[key] = P_t

    return pd.DataFrame.from_dict(bond_prices, orient='index', columns=['Bond Price'])

optimal_prices_df = compute_bond_prices_at_market_maturities(a_opt, b_opt, sigma_opt, r0, market_maturities, dt, num_paths)

# Reorder rows to match the plotting order of maturities
optimal_prices_df = optimal_prices_df.reindex(market_bond_prices.keys())

plt.figure(figsize=(8, 5))
plt.scatter(market_bond_prices.keys(), market_prices, color='red', label="Market Prices")
plt.plot(optimal_prices_df.index, optimal_prices_df["Bond Price"], marker='o', linestyle='dashed',
         label=f"Best-Fit CIR (SLSQP)")

plt.xlabel("Maturity")
plt.ylabel("Zero-Coupon Bond Price")
plt.title("CIR Model Calibration (Log-Price SSE with Constraints)")
plt.legend()
plt.grid(True)
plt.show()
