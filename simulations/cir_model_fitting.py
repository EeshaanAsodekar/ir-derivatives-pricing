import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
from pathlib import Path
# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.cir import simulate_cir
from simulations.cir_simulation import compute_zero_coupon_bond_prices


# === Market Data ===
market_bond_prices = {
    "1M": 0.9957,
    "3M": 0.9873,
    "6M": 0.9745,
    "1Y": 0.9562
}

maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0}
market_maturities = np.array([maturity_map[m] for m in market_bond_prices.keys()])
market_prices = np.array(list(market_bond_prices.values()))

# === Initial Guess & Model Params ===
initial_guess = [0.05, 0.04, 0.08]  # (a, b, sigma)
r0 = 0.052  # Short-term forward rate
num_paths = 10000
dt = 0.001

# === Objective Function ===
def objective_function(params):
    """
    Computes the SSE between market bond prices and simulated bond prices
    at the exact market maturities. 
    """
    a, b, sigma = params
    T = max(market_maturities)  # 1 year in this case

    # Simulate short-rate process
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # Compute bond price for each market maturity
    simulated_bond_prices = {}
    for m in market_maturities:
        num_steps = int(m / dt)
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        P_t = np.exp(-integral_rt).mean()
        key = f"{int(round(m * 12))}M" if m < 1 else "1Y"
        simulated_bond_prices[key] = P_t

    # Gather simulated prices in the same order as market data
    simulated_prices = np.array([simulated_bond_prices[k] for k in market_bond_prices.keys()])

    # SSE in "per-100" scale (since data is 0.xx = fraction of par)
    sse = np.sum((simulated_prices * 100 - market_prices * 100) ** 2)
    return sse

# === Compare Multiple Optimization Methods ===
methods_to_try = ["Nelder-Mead", "BFGS", "Powell", "SLSQP"]
results = {}

for method in methods_to_try:
    result = minimize(objective_function, initial_guess, method=method)
    sse = result.fun
    a_opt, b_opt, sigma_opt = result.x

    results[method] = {
        "a": a_opt,
        "b": b_opt,
        "sigma": sigma_opt,
        "SSE": sse
    }

# === Print Summary of Each Method ===
for method, vals in results.items():
    print(f"\nMethod: {method}")
    print(f"  SSE: {vals['SSE']:.6f}")
    print(f"  a={vals['a']:.6f}, b={vals['b']:.6f}, sigma={vals['sigma']:.6f}")

# === Pick the Best (lowest SSE) Method & Plot ===
best_method = min(results, key=lambda m: results[m]["SSE"])
best_params = results[best_method]
print(f"\nBest Method: {best_method} with SSE={best_params['SSE']:.6f}")

# === Recompute Bond Prices for Plotting ===
def compute_bond_prices_at_market_maturities(a, b, sigma, r0, market_maturities, dt, num_paths):
    rates_df = simulate_cir(a, b, sigma, r0, max(market_maturities), dt, num_paths)
    bond_prices = {}
    for m in market_maturities:
        num_steps = int(m / dt)
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        P_t = np.exp(-integral_rt).mean()
        key = f"{int(round(m * 12))}M" if m < 1 else "1Y"
        bond_prices[key] = P_t
    return pd.DataFrame.from_dict(bond_prices, orient='index', columns=['Bond Price'])

optimal_prices_df = compute_bond_prices_at_market_maturities(
    best_params["a"], best_params["b"], best_params["sigma"],
    r0, market_maturities, 0.01, 10000
)

# === Plot Market vs. Best-Fit CIR Model ===
plt.figure(figsize=(8, 5))
plt.scatter(list(market_bond_prices.keys()), market_prices, color='red', label="Market Prices", zorder=2)
plt.plot(optimal_prices_df.index, optimal_prices_df["Bond Price"], marker='o', linestyle='dashed',
         label=f"Best-Fit CIR ({best_method})", zorder=1)
plt.xlabel("Maturity")
plt.ylabel("Zero-Coupon Bond Price")
plt.title("CIR Model Calibration: Market vs. Simulated Bond Prices")
plt.legend()
plt.grid(True)
plt.show()