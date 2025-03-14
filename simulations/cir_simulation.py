import sys
from pathlib import Path
# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cir import simulate_cir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_zero_coupon_bond_prices(a, b, sigma, r0, T, dt, num_paths, max_maturity=5):
    """
    Computes zero-coupon bond prices using the CIR model.

    Args:
        a (float): Speed of mean reversion.
        b (float): Long-run mean interest rate.
        sigma (float): Volatility.
        r0 (float): Initial short rate.
        T (float): Total simulation time in years.
        dt (float): Time step size.
        num_paths (int): Number of simulation paths.
        max_maturity (int): Maximum bond maturity in years.

    Returns:
        pd.DataFrame: Bond prices for different maturities.
    """
    # Simulate CIR short-term rates
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # Time grid
    delta_t = dt  # Time step

    # Maturities to compute zero-coupon bond prices
    # 10 maturities between 0.5 and 5 years
    maturities = np.linspace(0.5, max_maturity, 10)
    bond_prices = {}

    for maturity in maturities:
        # Number of time steps corresponding to the maturity
        num_steps = int(maturity / delta_t)

        # Compute the integral sum(r_t dt) for each path
        integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * delta_t

        # Compute bond price: P(t) = E[exp(-integral r_t dt)]
        P_t = np.exp(-integral_rt).mean()
        bond_prices[maturity] = P_t

    return pd.DataFrame.from_dict(bond_prices, orient='index', columns=['Bond Price'])


def plot_zero_coupon_bond_prices():
    """
    Generates three separate plots showing how zero-coupon bond prices change with 
    variations in CIR parameters: a (mean reversion speed), b (long-run mean), 
    and sigma (volatility).
    """
    # Define base CIR parameters
    base_params = {
        "a": 0.05, "b": 0.04, "sigma": 0.08, "r0": 0.05, "T": 5, "dt": 0.01, "num_paths": 10000
    }

    # Define 8 values for each parameter
    a_values = np.linspace(0.01, 0.30, 8)  # Speed of mean reversion
    b_values = np.linspace(0.02, 0.1, 8)  # Long-run mean
    sigma_values = np.linspace(0.02, 0.30, 8)  # Volatility

    # === 1. Plot for Varying 'a' (Mean Reversion Speed) ===
    plt.figure(figsize=(8, 5))
    for a in a_values:
        params = base_params.copy()
        params["a"] = a
        bond_prices_df = compute_zero_coupon_bond_prices(**params)
        plt.plot(bond_prices_df.index,
                 bond_prices_df['Bond Price'], marker='o', label=f"a={a:.2f}")

    plt.title("Effect of a (Mean Reversion Speed)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero-Coupon Bond Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === 2. Plot for Varying 'b' (Long-Run Mean) ===
    plt.figure(figsize=(8, 5))
    for b in b_values:
        params = base_params.copy()
        params["b"] = b
        bond_prices_df = compute_zero_coupon_bond_prices(**params)
        plt.plot(bond_prices_df.index,
                 bond_prices_df['Bond Price'], marker='o', label=f"b={b:.2f}")

    plt.title("Effect of b (Long-Run Mean)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero-Coupon Bond Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === 3. Plot for Varying 'sigma' (Volatility) ===
    plt.figure(figsize=(8, 5))
    for sigma in sigma_values:
        params = base_params.copy()
        params["sigma"] = sigma
        bond_prices_df = compute_zero_coupon_bond_prices(**params)
        plt.plot(bond_prices_df.index,
                 bond_prices_df['Bond Price'], marker='o', label=f"σ={sigma:.2f}")

    plt.title("Effect of σ (Volatility)")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero-Coupon Bond Price")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_zero_coupon_bond_prices_for_params(a, b, sigma, r0=0.05, T=5, dt=0.01, 
                                            num_paths=10000, max_maturity=5):
    """
    Plots zero-coupon bond prices for a single (a, b, sigma) parameter set.

    Args:
        a (float): Speed of mean reversion.
        b (float): Long-run mean interest rate.
        sigma (float): Volatility.
        r0 (float): Initial short rate.
        T (float): Total simulation time in years.
        dt (float): Time step size.
        num_paths (int): Number of simulation paths.
        max_maturity (int): Maximum bond maturity in years.
    """
    bond_prices_df = compute_zero_coupon_bond_prices(a, b, sigma, r0, T, dt, 
                                                     num_paths, max_maturity)
    
    plt.figure(figsize=(8, 5))
    plt.plot(bond_prices_df.index, bond_prices_df['Bond Price'], marker='o',
             label=f"a={a:.5f}, b={b:.5f}, σ={sigma:.5f}")
    plt.title("Zero-Coupon Bond Prices for Given a, b, σ")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Zero-Coupon Bond Price")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_zero_coupon_bond_prices_for_params(a=0.05, b=0.04, sigma=0.08, r0=0.05)
    plot_zero_coupon_bond_prices()
