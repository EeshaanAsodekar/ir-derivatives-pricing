import sys
from pathlib import Path
# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.cir import simulate_cir
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats



def analyze_parameter_sensitivity():
    """
    Runs CIR model simulations for different parameter values and 
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
    a_values = [0.02, 0.05, 0.1]      # Mean reversion speed
    sigma_values = [0.05, 0.08, 0.12]  # Volatility
    b_values = [0.03, 0.04, 0.05]     # Long-run mean levels

    # Store results
    results = []

    for a in a_values:
        for sigma in sigma_values:
            for b in b_values:
                # Run CIR simulation
                rates_df = simulate_cir(
                    a=a, b=b, sigma=sigma,
                    r0=base_params["r0"], T=base_params["T"],
                    dt=base_params["dt"], num_paths=base_params["num_paths"]
                )

                # Compute final mean and variance
                final_rates = rates_df.iloc[-1, :]  # Last time step
                empirical_mean = np.mean(final_rates)
                empirical_variance = np.var(final_rates)

                # Theoretical expectation and variance
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
    print("\nCIR Parameter Sensitivity Analysis")
    print(results_df)

    # Save results
    results_df.to_csv("data/cir_sensitivity_analysis.csv", index=False)


def check_distribution_normality():
    """
    Simulates the CIR model and checks if r_t follows a normal distribution.
    Uses kurtosis, percentile ratios, and empirical vs normal CDF.
    """
    # Parameters
    params = {
        "a": 0.05, "b": 0.04, "sigma": 0.08, "r0": 0.05, "T": 50, "dt": 0.01, "num_paths": 10000
    }

    # Run CIR simulation
    rates_df = simulate_cir(**params)

    # Extract last time step values
    final_rates = rates_df.iloc[-1, :]

    # === 1. Statistical Measures ===
    mean_rt = np.mean(final_rates)
    std_rt = np.std(final_rates)
    skewness = stats.skew(final_rates)
    kurtosis = stats.kurtosis(final_rates)  # Excess kurtosis
    percentile_ratio = (np.percentile(final_rates, 90) - np.percentile(final_rates, 10)) / std_rt

    print(f"\n--- CIR Process Normality Check ---")
    print(f"Mean: {mean_rt:.5f}, Variance: {std_rt**2:.5f}")
    print(f"Skewness: {skewness:.5f} (For normal: ~0)")
    print(f"Kurtosis: {kurtosis:.5f} (For normal: ~0)")
    print(f"Percentile Ratio (90%-10% range / std dev): {percentile_ratio:.5f}")

    # === 2. Normality Tests ===
    shapiro_test = stats.shapiro(final_rates[:5000])  # Shapiro-Wilk test (limited to 5000 samples)
    ks_test = stats.kstest(final_rates, 'norm', args=(mean_rt, std_rt))  # Kolmogorov-Smirnov test
    jb_test = stats.jarque_bera(final_rates)  # Jarque-Bera test

    print("\n--- Statistical Normality Tests ---")
    print(f"Shapiro-Wilk test p-value: {shapiro_test.pvalue:.5f} (p < 0.05 -> not normal)")
    print(f"Kolmogorov-Smirnov test p-value: {ks_test.pvalue:.5f} (p < 0.05 -> not normal)")
    print(f"Jarque-Bera test p-value: {jb_test.pvalue:.5f} (p < 0.05 -> not normal)")


    # Plot histogram vs normal
    plt.figure(figsize=(12, 5))
    plt.hist(final_rates, bins=50, density=True,
             alpha=0.6, label="CIR Distribution")
    x = np.linspace(min(final_rates), max(final_rates), 100)
    plt.plot(x, stats.norm.pdf(x, np.mean(final_rates), np.std(
        final_rates)), label="Normal Dist", linestyle="dashed")
    plt.xlabel("r_t values")
    plt.ylabel("Density")
    plt.title("Histogram of CIR vs Normal Distribution")
    plt.legend()
    plt.show()

    # Plot empirical vs normal CDF
    plt.figure(figsize=(12, 5))
    empirical_cdf = np.sort(final_rates)
    theoretical_cdf = stats.norm.cdf(
        empirical_cdf, np.mean(final_rates), np.std(final_rates))
    plt.plot(empirical_cdf, np.linspace(
        0, 1, len(empirical_cdf)), label="Empirical CDF")
    plt.plot(empirical_cdf, theoretical_cdf,
             label="Normal CDF", linestyle="dashed")
    plt.xlabel("r_t values")
    plt.ylabel("Cumulative Probability")
    plt.title("Empirical CDF vs Normal CDF")
    plt.legend()
    plt.show()


def plot_convergence_to_mean():
    """
    Plots the convergence of the simulated mean rate towards the long-run mean (b).
    """
    params = {
        "a": 0.05, "b": 0.04, "sigma": 0.08, "r0": 0.05, "T": 50, "dt": 0.01, "num_paths": 10000
    }

    # Run simulation
    rates_df = simulate_cir(**params)

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
    rates_df = simulate_cir(**params)

    # Compute variance at each time step
    variance_rates = rates_df.var(axis=1)
    theoretical_variance = (params["sigma"]**2 *params["b"]) / (2 * params["a"])

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
    check_distribution_normality()
    plot_convergence_to_mean()
    plot_variance_convergence()
