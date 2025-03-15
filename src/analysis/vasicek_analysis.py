# src/analysis/vasicek_analysis.py

import sys
from pathlib import Path
# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.vasicek import simulate_vasicek
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def analyze_parameter_sensitivity():
    """
    Runs Vasicek model simulations for different parameter values and 
    analyzes their effects on long-run mean and variance.
    """
    # Define base parameters
    base_params = {
        "r0": 0.05,   # Initial interest rate
        "T": 50,      # Simulation time (years)
        "dt": 0.01,   # Time step
        "num_paths": 10000  # Number of simulation paths
    }

    # Vary parameters
    a_values = [0.02, 0.05, 0.1]      # Lower and higher mean reversion speeds
    sigma_values = [0.05, 0.08, 0.12]  # Lower and higher volatility
    b_values = [0.03, 0.04, 0.05]     # Different long-run means

    # Store results
    results = []

    for a in a_values:
        for sigma in sigma_values:
            for b in b_values:
                # Run simulation without duplicate arguments
                rates_df = simulate_vasicek(
                    a=a,
                    b=b,
                    sigma=sigma,
                    r0=base_params["r0"],
                    T=base_params["T"],
                    dt=base_params["dt"],
                    num_paths=base_params["num_paths"]
                )

                # Compute final mean and variance
                final_rates = rates_df.iloc[-1, :]  # Last time step rates
                empirical_mean = np.mean(final_rates)
                empirical_variance = np.var(final_rates)

                # Theoretical expectations
                theoretical_variance = sigma**2 / (2 * a)

                # Store results
                results.append({
                    "a": a, "b": b, "sigma": sigma,
                    "empirical_mean": empirical_mean,
                    "empirical_variance": empirical_variance,
                    "theoretical_variance": theoretical_variance
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Display results
    print("\nParameter Sensitivity Analysis")
    print(results_df)

    # Save results
    results_df.to_csv("data/vasicek_sensitivity_analysis.csv", index=False)


def plot_convergence_to_mean():
    """
    Plots the convergence of the simulated mean rate towards the long-run mean (b).
    """
    params = {
        "a": 0.05, "b": 0.04, "sigma": 0.08, "r0": 0.05, "T": 50, "dt": 0.01, "num_paths": 10000
    }

    # Run simulation
    rates_df = simulate_vasicek(**params)

    # Compute mean at each time step
    mean_rates = rates_df.mean(axis=1)

    # Plot convergence
    plt.figure(figsize=(10, 5))
    plt.plot(mean_rates, label="Empirical Mean of $r_t$")
    plt.axhline(params["b"], color="red", linestyle="dashed",
                label=f"Theoretical Mean (b={params['b']})")
    plt.title("Convergence of Mean Interest Rate to Long-Run Mean (b)")
    plt.xlabel("Time (Years)")
    plt.ylabel("Interest Rate")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_variance_convergence():
    """
    Plots the evolution of variance over time and compares it to theoretical variance.
    """
    params = {
        "a": 0.05, "b": 0.04, "sigma": 0.08, "r0": 0.05, "T": 50, "dt": 0.01, "num_paths": 10000
    }

    # Run simulation
    rates_df = simulate_vasicek(**params)

    # Compute variance at each time step
    variance_rates = rates_df.var(axis=1)
    theoretical_variance = params["sigma"]**2 / (2 * params["a"])

    # Plot variance convergence
    plt.figure(figsize=(10, 5))
    plt.plot(variance_rates, label="Empirical Variance of $r_t$")
    plt.axhline(theoretical_variance, color="red", linestyle="dashed",
                label=f"Theoretical Variance ({theoretical_variance:.4f})")
    plt.title("Convergence of Variance of $r_t$ to Theoretical Limit")
    plt.xlabel("Time (Years)")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    analyze_parameter_sensitivity()
    plot_convergence_to_mean()
    plot_variance_convergence()
