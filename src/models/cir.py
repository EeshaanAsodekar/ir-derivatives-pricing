import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def simulate_cir(a: float, b: float, sigma: float, r0: float, T: float, dt: float, num_paths: int = 1) -> pd.DataFrame:
    """
    Simulates the Cox-Ingersoll-Ross (CIR) interest rate process using Euler-Maruyama discretization.

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
        # Generate random shocks from a normal distribution
        epsilon = np.random.normal(0, np.sqrt(dt), num_paths)

        # Euler-Maruyama discretization of the CIR model:
        # dr_t = a(b - r_t) dt + σ sqrt(r_t) dB_t
        rates[:, i] = rates[:, i - 1] + \
            a * (b - rates[:, i - 1]) * dt + \
            sigma * np.sqrt(np.maximum(rates[:, i - 1], 0)) * epsilon

    # Convert rates matrix to a DataFrame with time index and path labels
    return pd.DataFrame(rates.T, index=time_grid, columns=[f'Path {i+1}' for i in range(num_paths)])


def plot_cir_simulation(rates_df: pd.DataFrame) -> None:
    """
    Plots the simulated CIR interest rate paths.

    Args:
        rates_df (pd.DataFrame): DataFrame containing simulated interest rate paths.
    """
    # Set figure size
    plt.figure(figsize=(10, 5))

    # Plot each simulated path with transparency
    plt.plot(rates_df, alpha=0.75)

    # Set title and labels
    plt.title("Simulated CIR Interest Rate Paths")
    plt.xlabel("Time (Years)")
    plt.ylabel("Interest Rate")

    # Add grid for readability
    plt.grid(True)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    # Define model parameters for simulation
    a = 0.01  # Speed of mean reversion
    b = 0.06  # Long-run mean interest rate
    sigma = 0.08  # Volatility of the process
    r0 = 0.05  # Initial short-term rate
    T = 50  # Total simulation time (years)
    dt = 0.01  # Time step size
    num_paths = 1000  # Number of simulation paths

    # Run CIR simulation
    rates_df = simulate_cir(a, b, sigma, r0, T, dt, num_paths)

    # Plot the simulated interest rate paths
    plot_cir_simulation(rates_df)
