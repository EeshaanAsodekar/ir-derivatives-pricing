"""
Module: cir_swaption_pricing.py
Description: Prices an ATM swaption using the calibrated CIR model. The swaption 
expires in 1 month and covers a 3-month swap (ending at 4 months). The fair swap rate 
is calculated from zero-coupon bond prices, and the swaption is priced using Black's formula.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from pathlib import Path

# Add project root to sys.path for module imports.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.cir import simulate_cir

# === Calibrated CIR Parameters (from Question 2 calibration) ===
# TODO: update to latest parameters
a_calib = 0.05    # Mean reversion speed
b_calib = 0.04    # Long-run mean rate
sigma_calib = 0.08  # Volatility
r0 = 0.052        # Initial short rate

# === Simulation Settings ===
num_paths = 10000  # Number of Monte Carlo paths
dt = 0.01          # Time step size

# === Swaption Specifications ===
T_swaption_expiry = 1 / 12  # Swaption expiry: 1 month (in years)
T_swap_end = 4 / 12         # Swap ends at 4 months (swap tenor = 3 months)
Delta = 0.25                # Accrual factor for 3-month swap (0.25 year)

def compute_zero_coupon_price(rates_df, T, dt):
    """
    Compute the zero-coupon bond price P(T) using a Riemann sum approximation:
        P(T) = E[exp(-∫_0^T r(t) dt)]
    
    Parameters:
        rates_df (DataFrame): Simulated short-rate paths.
        T (float): Maturity (years) at which to price the bond.
        dt (float): Time step size.
    
    Returns:
        float: The zero-coupon bond price P(T).
    """
    num_steps = int(T / dt)
    integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
    P_T = np.exp(-integral_rt).mean()
    return P_T

def compute_swap_rate_distribution(rates_df, dt, T_expiry, T_swap_end, Delta):
    """
    Computes the distribution of swap rates at swaption expiry.
    
    For each simulation path, the swap rate is defined as:
        F = (P(1M) - P(4M)) / (Delta * P(4M))
    
    Parameters:
        rates_df (DataFrame): Simulated short-rate paths.
        dt (float): Time step size.
        T_expiry (float): Swaption expiry (years).
        T_swap_end (float): Swap maturity (years).
        Delta (float): Accrual factor for the swap.
    
    Returns:
        np.ndarray: Array of swap rates across simulation paths.
    """
    num_steps_expiry = int(T_expiry / dt)
    num_steps_swap = int(T_swap_end / dt)
    # Compute integral for bond price at expiry and at swap end
    integral_rt_1M = rates_df.iloc[:num_steps_expiry].sum(axis=0) * dt
    integral_rt_4M = rates_df.iloc[:num_steps_swap].sum(axis=0) * dt
    P_1M_paths = np.exp(-integral_rt_1M)
    P_4M_paths = np.exp(-integral_rt_4M)
    swap_rates = (P_1M_paths - P_4M_paths) / (Delta * P_4M_paths)
    return swap_rates

def price_swaption(F, sigma_swap, A, T_expiry):
    """
    Prices an ATM swaption using Black's formula. For an ATM option (strike K = F),
    Black's formula simplifies to:
        Price = A * F * [N(d1) - N(d2)]
    where d1 = 0.5 * sigma_swap * sqrt(T_expiry) and d2 = -d1.
    
    Parameters:
        F (float): Fair swap rate (forward swap rate).
        sigma_swap (float): Implied volatility of the swap rate.
        A (float): Annuity (present value of the swap's fixed leg).
        T_expiry (float): Time to expiry of the swaption.
    
    Returns:
        float: The price of the ATM swaption.
    
    Note:
    Black's model is a standard model used for pricing European-style options on interest rates. 
    It's an adaptation of Black-Scholes but designed specifically for interest rate derivatives, 
    such as caps, floors, and swaptions. The model assumes lognormal distribution of forward swap 
    rates, which fits our CIR-generated swap rate distribution.
    """
    d1 = 0.5 * sigma_swap * np.sqrt(T_expiry)
    d2 = -d1
    swaption_price = A * F * (norm.cdf(d1) - norm.cdf(d2))
    return swaption_price

def main():
    # --- Step 1: Simulate the CIR Process up to T_swap_end (4 months) ---
    T_total = T_swap_end  # 4 months simulation
    rates_df = simulate_cir(a_calib, b_calib, sigma_calib, r0, T_total, dt, num_paths)
    
    # --- Step 2: Compute Zero-Coupon Bond Prices at 1M and 4M ---
    P_1M = compute_zero_coupon_price(rates_df, T_swaption_expiry, dt)
    P_4M = compute_zero_coupon_price(rates_df, T_swap_end, dt)
    print(f"P(1M) = {P_1M:.6f}, P(4M) = {P_4M:.6f}")
    
    # --- Step 3: Calculate the Fair Swap Rate ---
    F_swap = (P_1M - P_4M) / (Delta * P_4M)
    print(f"Fair Swap Rate, F = {F_swap:.6f}")
    
    # --- Step 4: Simulate the Distribution of Swap Rates at 1M ---
    swap_rates = compute_swap_rate_distribution(rates_df, dt, T_swaption_expiry, T_swap_end, Delta)
    F_swap_sim = np.mean(swap_rates)
    sigma_swap = np.std(swap_rates)
    print(f"Simulated Fair Swap Rate, F = {F_swap_sim:.6f}")
    print(f"Swap Rate Volatility, sigma_swap = {sigma_swap:.6f}")
    
    # --- Step 5: Compute the Annuity (PV of Fixed Leg) ---
    A = Delta * P_4M
    
    # --- Step 6: Price the ATM Swaption Using Black's Formula ---
    swaption_price = price_swaption(F_swap_sim, sigma_swap, A, T_swaption_expiry)
    print(f"ATM Swaption Price = {swaption_price:.6f}")
    
    # --- Step 7: Plot the Distribution of Simulated Swap Rates ---
    plt.figure(figsize=(8, 5))
    plt.hist(swap_rates, bins=50, density=True, alpha=0.6, label="Simulated Swap Rates")
    x_vals = np.linspace(min(swap_rates), max(swap_rates), 100)
    plt.plot(x_vals, norm.pdf(x_vals, F_swap_sim, sigma_swap), 'r--', label="Normal Fit")
    plt.xlabel("Swap Rate")
    plt.ylabel("Density")
    plt.title("Distribution of Simulated Swap Rates at 1M")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
