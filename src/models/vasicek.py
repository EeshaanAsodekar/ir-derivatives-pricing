# src/models/vasicek.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_vasicek(a: float, b: float, sigma: float, r0: float, T: float, dt: float, num_paths: int = 1) -> pd.DataFrame:
    """
    Simulates the Vasicek interest rate process using Euler discretization.

    Args:
        a (float): Speed of mean reversion.
        b (float): Long-run mean interest rate.
        sigma (float): Volatility of the process.
        r0 (float): Initial short-term rate.
        T (float): Total simulation time in years.
        dt (float): Time step size.
        num_paths (int, optional): Number of simulation paths. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing simulated interest rate paths.
    """
    # Number of time steps
    N = int(T / dt)

    # Generate time points
    time_grid = np.linspace(0, T, N)

    # Initialize rates matrix
    rates = np.zeros((num_paths, N))

    # Set the initial rate for all paths
    rates[:, 0] = r0

    for i in range(1, N):
        # Generate random shocks from a normal distribution with mean 0 and variance Δt
        epsilon = np.random.normal(0, np.sqrt(dt), num_paths)

        # Euler discretization of the Vasicek model:
        # dr_t = a(b - r_t) dt + σ dB_t
        #
        # Discretized form:
        # r_{t_i} = r_{t_{i-1}} + a(b - r_{t_{i-1}}) Δt + σ ε_i
        #
        # where:
        # - a: Speed of mean reversion (higher a → faster reversion to b)
        # - b: Long-run mean level of interest rates
        # - σ: Volatility of the process
        # - Δt: Time step
        # - ε_i ~ N(0, Δt): Normally distributed random shock
        rates[:, i] = rates[:, i - 1] + a * \
            (b - rates[:, i - 1]) * dt + sigma * epsilon

    # Convert rates matrix to a DataFrame with time index and path labels
    return pd.DataFrame(rates.T, index=time_grid, columns=[f'Path {i+1}' for i in range(num_paths)])


def plot_vasicek_simulation(rates_df: pd.DataFrame) -> None:
    """
    Plots the simulated Vasicek interest rate paths.

    Args:
        rates_df (pd.DataFrame): DataFrame containing simulated interest rate paths.
    """
    # Set figure size
    plt.figure(figsize=(10, 5))

    # Plot each simulated path with transparency
    plt.plot(rates_df, alpha=0.75)

    # Set title and labels
    plt.title("Simulated Vasicek Interest Rate Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Interest Rate")

    # Add grid for readability
    plt.grid(True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Define model parameters for simulation
    a = 0.05  # Speed of mean reversion
    b = 0.04  # Long-run mean interest rate
    sigma = 0.08  # Volatility of the process
    r0 = 0.05  # Initial short-term rate
    T = 50  # Total simulation time (years)
    dt = 0.01  # Time step size
    num_paths = 10000  # Number of simulation paths

    # Run Vasicek simulation
    rates_df = simulate_vasicek(a, b, sigma, r0, T, dt, num_paths)

    # Plot the simulated interest rate paths
    plot_vasicek_simulation(rates_df)
