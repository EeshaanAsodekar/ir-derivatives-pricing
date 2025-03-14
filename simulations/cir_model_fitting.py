import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
from pathlib import Path

# 1. Make sure Python can find our 'simulate_cir' function.
#    (Adjust the path if your project structure is different.)
sys.path.append(str(Path(__file__).resolve().parents[1]))
from simulations.cir_simulation import compute_zero_coupon_bond_prices
from src.models.cir import simulate_cir

# ------------------------------------------------------------------------------
# MARKET DATA
# ------------------------------------------------------------------------------
# Here we define the known market zero-coupon bond prices for specific maturities.
market_bond_prices = {
    "1M": 0.9957,  # Price for a bond maturing in 1 month
    "3M": 0.9873,  # Price for a bond maturing in 3 months
    "6M": 0.9745,  # Price for a bond maturing in 6 months
    "1Y": 0.9562   # Price for a bond maturing in 1 year
}

# Convert textual maturities (e.g., "1M") into year-fractions (e.g., 1/12).
maturity_map = {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y": 1.0}
# Extract arrays for easy vector operations:
market_maturities = np.array([maturity_map[m] for m in market_bond_prices.keys()])
market_prices = np.array(list(market_bond_prices.values()))

# ------------------------------------------------------------------------------
# MODEL/OPTIMIZATION SETTINGS
# ------------------------------------------------------------------------------
# (a) Our initial guess for the CIR parameters: a, b, sigma
initial_guess = [0.05, 0.04, 0.08]  # (a, b, sigma)
# (b) The initial short rate r(0).
r0 = 0.052
# (c) Monte Carlo settings.
num_paths = 1000
dt = 0.0001

# ------------------------------------------------------------------------------
# CONSTRAINTS
# ------------------------------------------------------------------------------
# We impose certain constraints to ensure positivity of a, b, sigma,
# and put an upper bound on sigma and b if desired.
def constraint_a(params):
    """ Return 'a'; must be > 0 to satisfy positivity. """
    a, _, _ = params
    return a

def constraint_b(params):
    """ Return 'b'; must be > 0 to satisfy positivity. """
    _, b, _ = params
    return b

def constraint_sigma_positive(params):
    """ Return 'sigma'; must be > 0 for positivity. """
    _, _, sigma = params
    return sigma

def constraint_a_upper(params):
    """ Ensure a < 0.5 by having 0.5 - a >= 0. """
    a, _, _ = params
    return 0.5 - a

def constraint_sigma_upper(params):
    """ Ensure sigma < 0.3 by having 0.3 - sigma >= 0. """
    _, _, sigma = params
    return 0.3 - sigma

def constraint_b_upper(params):
    """ Example: b < 0.07 => 0.07 - b >= 0. """
    _, b, _ = params
    return 0.07 - b

constraints = [
    {'type': 'ineq', 'fun': constraint_a},             # a > 0
    {'type': 'ineq', 'fun': constraint_b},             # b > 0
    {'type': 'ineq', 'fun': constraint_sigma_positive},# sigma > 0
    {'type': 'ineq', 'fun': constraint_sigma_upper},   # sigma < 0.3
    {'type': 'ineq', 'fun': constraint_a_upper},        # a < 0.5
    {'type': 'ineq', 'fun': constraint_b_upper}        # b < 0.07
]

# ------------------------------------------------------------------------------
# OBJECTIVE FUNCTION
# ------------------------------------------------------------------------------
def objective_function(params):
    """
    Computes the sum of squared errors (SSE) on *log bond prices* 
    between the CIR-simulated bond prices and the given market data.

    Parameters:
      params: [a, b, sigma] for the CIR model.

    Steps:
      1) Simulate short-rate paths with the given (a, b, sigma).
      2) For each market maturity, compute the average zero-coupon bond price 
         from the paths.
      3) Compare log(simulated_price) vs. log(market_price) to form an SSE.
    """
    # Unpack the CIR parameters.
    a, b, sigma = params

    # We'll need to simulate up to the max maturity in our market data (1 year).
    T = max(market_maturities)

    # Simulate short-rate paths under the CIR model.
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # Compute bond prices for each market maturity by discounting 
    # each path's short rates and taking the average.
    simulated_prices = []
    for key in market_bond_prices.keys():
        m = maturity_map[key]
        num_steps = int(m / dt)
        # Integrate r(t)*dt along the path for t up to 'm'
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        # Bond price = E[exp(-integral of r)]
        P_t = np.exp(-integral_rt).mean()
        simulated_prices.append(P_t)

    simulated_prices = np.array(simulated_prices)
    
    # Use log prices to reduce scale differences and penalize relative errors.
    eps = 1e-12  # small number to avoid log(0)
    log_sim = np.log(np.clip(simulated_prices, eps, 1.0))
    log_mkt = np.log(np.clip(market_prices, eps, 1.0))
    sse_log = np.sum((log_sim - log_mkt) ** 2)

    return sse_log

# ------------------------------------------------------------------------------
# OPTIMIZATION
# ------------------------------------------------------------------------------
# We'll use SLSQP, which allows us to enforce our inequality constraints easily.
result = minimize(
    objective_function, 
    initial_guess, 
    method="SLSQP", 
    constraints=constraints,
    options={"maxiter": 500, "ftol": 1e-12}
)

# The fitted parameters:
a_opt, b_opt, sigma_opt = result.x

print("Optimal Parameters (SLSQP, log-price SSE):")
print(f"  a={a_opt:.6f}, b={b_opt:.6f}, sigma={sigma_opt:.6f}")
print(f"  Final SSE (log-prices)={result.fun:.6f}")

# ------------------------------------------------------------------------------
# VALIDATION / PLOT
# ------------------------------------------------------------------------------
# We'll compare the best-fit model's bond prices against the market data visually.
def compute_bond_prices_at_market_maturities(a, b, sigma, r0, market_maturities, dt, num_paths):
    """
    Helper function to compute zero-coupon bond prices at all given maturities 
    using the specified CIR parameters, for plot comparison.
    """
    # Simulate up to the max maturity:
    rates_df = simulate_cir(a, b, sigma, r0, max(market_maturities), dt, num_paths)

    bond_prices = {}
    for key in market_bond_prices.keys():
        m = maturity_map[key]
        num_steps = int(m / dt)
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        P_t = np.exp(-integral_rt).mean()
        bond_prices[key] = P_t

    return pd.DataFrame.from_dict(bond_prices, orient='index', columns=['Bond Price'])

# Compute the bond prices for the fitted model:
optimal_prices_df = compute_bond_prices_at_market_maturities(
    a_opt, b_opt, sigma_opt, r0, market_maturities, dt, num_paths
)

# Reorder rows to match the order we used in 'market_bond_prices'
optimal_prices_df = optimal_prices_df.reindex(market_bond_prices.keys())

### plotting
x_market = [maturity_map[k] for k in market_bond_prices.keys()]  # Numeric x for market
x_model = [maturity_map[k] for k in optimal_prices_df.index]     # Numeric x for model

plt.figure(figsize=(8, 5))

# Plot only market points at proportional x-coordinates
plt.scatter(x_market, market_prices, color='red', label="Market Prices")

# Plot only model points (without connecting line)
plt.scatter(x_model, optimal_prices_df["Bond Price"], color='blue', marker='o', label="Best-Fit CIR (SLSQP)")

plt.xlabel("Maturity (Years)")
plt.ylabel("Zero-Coupon Bond Price")
plt.title("CIR Model Calibration (Log-Price SSE with Constraints)")

# Use numeric x-values but label them with the original tenor strings
plt.xticks(x_market, market_bond_prices.keys())

plt.legend()
plt.grid(True)
plt.show()



def plot_zero_coupon_bond_prices_for_params(a, b, sigma,
                                            market_bond_prices,  # e.g., {"1M": 0.9957, "3M": 0.9873, ...}
                                            maturity_map,        # e.g., {"1M": 1/12, "3M": 3/12, ...}
                                            r0=0.05, T=5, dt=0.01, 
                                            num_paths=10000, max_maturity=5):
    """
    Plots zero-coupon bond prices at each monthly maturity (1M, 2M, ..., up to max_maturity*12M)
    for the given (a, b, sigma). Also overlays market bond prices (with thicker markers).

    Args:
        a (float): Speed of mean reversion.
        b (float): Long-run mean interest rate.
        sigma (float): Volatility.
        market_bond_prices (dict): Market bond prices, e.g. {"1M": 0.9957, "3M": 0.9873, ...}.
        maturity_map (dict): Mapping for each market maturity to a year fraction, 
                             e.g. {"1M": 1/12, "3M": 3/12, "6M": 6/12, "1Y":1.0, ...}.
        r0 (float): Initial short rate.
        T (float): Total simulation time in years (must be >= max_maturity).
        dt (float): Time step size.
        num_paths (int): Number of simulation paths.
        max_maturity (int): Maximum bond maturity in years (integer).
    """
    # --------------------------------------------------------------------------
    # 1. Simulate CIR paths once, up to at least `T` = max_maturity (in years).
    # --------------------------------------------------------------------------
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # --------------------------------------------------------------------------
    # 2. Compute a model-based bond price for each monthly maturity
    #    from 1M out to (max_maturity * 12)M.
    # --------------------------------------------------------------------------
    monthly_maturities = np.arange(1, max_maturity * 12 + 1) / 12.0  # [1/12, 2/12, ..., 5 for max_maturity=5]
    bond_prices = {}
    for maturity in monthly_maturities:
        num_steps = int(maturity / dt)
        # Integrate rates up to 'maturity'
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
        # Bond price = E[e^(-∫r_t dt)]
        bond_prices[maturity] = np.exp(-integral_rt).mean()

    # Convert to a DataFrame sorted by maturity
    bond_prices_df = pd.DataFrame.from_dict(bond_prices, orient='index', columns=['Bond Price'])
    bond_prices_df.sort_index(inplace=True)

    # --------------------------------------------------------------------------
    # 3. Extract the market bond prices for plotting (with bigger/thicker markers).
    # --------------------------------------------------------------------------
    # Example usage of the provided 'maturity_map':
    #    "1M" -> 1/12, "3M" -> 3/12, etc.
    x_market = [maturity_map[m] for m in market_bond_prices.keys()]
    y_market = list(market_bond_prices.values())

    # --------------------------------------------------------------------------
    # 4. Plot the results
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 5))

    # (a) Model-based monthly curve (connected line with small markers).
    plt.plot(
        bond_prices_df.index,
        bond_prices_df['Bond Price'],
        linestyle='-',
        linewidth=1.75,
        label=f"Model: a={a:.5f}, b={b:.5f}, σ={sigma:.5f}"
    )

    # (b) Overlay the market bond prices with bigger markers & thicker edges.
    plt.scatter(
        x_market, y_market,
        s=120,            # marker size
        linewidths=2,     # edge/border width
        edgecolors='black',
        color='red',
        marker='o',
        label="Market Prices"
    )

    # --------------------------------------------------------------------------
    # 5. Finalize the plot
    # --------------------------------------------------------------------------
    plt.title("Zero-Coupon Bond Prices: Monthly Model Curve vs. Market")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero-Coupon Bond Price")
    plt.grid(True)
    plt.legend()
    plt.show()

# 5 year curve
plot_zero_coupon_bond_prices_for_params(
    a=a_opt, 
    b=b_opt, 
    sigma=sigma_opt,
    market_bond_prices=market_bond_prices,  # dict like {"1M": 0.9957, "3M": 0.9873, ...}
    maturity_map=maturity_map,              # dict like {"1M": 1/12, "3M": 3/12, ...}
    r0=r0,
    T=5, 
    dt=0.01, 
    num_paths=10000,
    max_maturity=5
)

# 1 year curve
plot_zero_coupon_bond_prices_for_params(
    a=a_opt, 
    b=b_opt, 
    sigma=sigma_opt,
    market_bond_prices=market_bond_prices,  # dict like {"1M": 0.9957, "3M": 0.9873, ...}
    maturity_map=maturity_map,              # dict like {"1M": 1/12, "3M": 3/12, ...}
    r0=r0,
    T=1, 
    dt=0.01, 
    num_paths=10000,
    max_maturity=1
)