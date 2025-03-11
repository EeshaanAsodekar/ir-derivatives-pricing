"""
Module: cir_swaption_pricing.py
Description: 
  (A) Prices an ATM swaption using the calibrated CIR model. The swaption 
      expires in 1 month (T=1/12) and covers a 3-month swap (ending at 4 months). 
      The fair swap rate is calculated from zero-coupon bond prices, and the swaption 
      is also priced via a Black-style formula on the forward swap rate.

  (B) Computes the implied volatility for an ATM put on the zero-coupon bond 
      (maturity = 4M), assuming lognormal bond-price dynamics (Black-Scholes). 
      We find the 'model-based' put price from the same CIR simulation 
      and then invert to find the Black-Scholes implied vol.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq  # for implied-vol root-finding
import sys
from pathlib import Path

# Add project root to sys.path for module imports (adjust as needed).
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.cir import simulate_cir  # <- your own module

# === Calibrated CIR Parameters (from Question 2 calibration) ===
# TODO: Replace these with your actual calibrated values
a_calib = 0.05      # Mean reversion speed
b_calib = 0.04      # Long-run mean rate
sigma_calib = 0.08  # Volatility
r0 = 0.052          # Initial short rate

# === Simulation Settings ===
num_paths = 10000   # Number of Monte Carlo paths
dt = 0.01           # Time step size

# === Swaption & Bond Specs ===
T_swaption_expiry = 1 / 12  # Swaption expiry: 1 month (in years)
T_swap_end = 4 / 12         # Swap ends at 4 months (swap tenor = 3 months)
Delta = 0.25                # Accrual factor for 3-month swap (0.25 year)

# ------------------------------------------------------------------------------
# 1) Utility Functions
# ------------------------------------------------------------------------------

def compute_zero_coupon_price(rates_df, T, dt):
    """
    Compute the zero-coupon bond price P(T) using a Riemann sum approximation:
        P(T) = E[ exp(-∫_{0 to T} r(u) du) ].
    
    Parameters
    ----------
    rates_df : DataFrame
        Simulated short-rate paths (rows=time steps, cols=paths).
    T : float
        Maturity (in years) at which to price the bond.
    dt : float
        Time step size.
    
    Returns
    -------
    float
        The zero-coupon bond price P(T).
    """
    num_steps = int(T / dt)
    # For each path, sum up r(u)*dt from u=0 to u=T
    integral_rt = rates_df.iloc[:num_steps].sum(axis=0) * dt
    # Bond price is E[exp(- integral of r)]
    P_T = np.exp(-integral_rt).mean()
    return P_T

def compute_swap_rate_distribution(rates_df, dt, T_expiry, T_swap_end, Delta):
    """
    Computes the distribution of swap rates (F) as of the swaption expiry T_expiry.

    For each path, define:
        P_expiry = exp(- ∫_{0 to T_expiry} r(u) du)
        P_end    = exp(- ∫_{0 to T_swap_end} r(u) du)
      Then the forward swap rate (for the single payment at T_swap_end) is
        F = (P_expiry - P_end) / [Delta * P_end]
    
    This is effectively the rate that makes the value of fixed leg = value of floating leg.
    
    Returns
    -------
    swap_rates : np.ndarray
        Array of size [num_paths,], containing each path's forward swap rate at T_expiry.
    """
    num_steps_expiry = int(T_expiry / dt)
    num_steps_swap   = int(T_swap_end / dt)
    
    # Integral up to 1M
    integral_0_to_1M = rates_df.iloc[:num_steps_expiry].sum(axis=0) * dt
    # Integral up to 4M
    integral_0_to_4M = rates_df.iloc[:num_steps_swap].sum(axis=0) * dt
    
    P_1M_paths = np.exp(-integral_0_to_1M)
    P_4M_paths = np.exp(-integral_0_to_4M)
    
    # Single-payment 'swap rate'
    swap_rates = (P_1M_paths - P_4M_paths) / (Delta * P_4M_paths)
    return swap_rates

def calc_swaption_price_mc(rates_df, dt, T_expiry, T_swap_end, Delta, strike):
    """
    Pure Monte Carlo pricing of a single-payment payer swaption:
      - The swaption expires at T_expiry.
      - The underlying swap has one payment at T_swap_end,
        with the fixed rate = 'strike'.
      - Notional is 1, so payoff at T_swap_end is:
            max( F_swap(T_expiry) - strike, 0 ) * Delta.
      - Then discount that payoff from T_swap_end back to time 0
        using the pathwise short rates.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Simulated short-rate paths (rows=time steps, columns=paths).
    dt : float
        Time step size in years.
    T_expiry : float
        Swaption expiry (in years).
    T_swap_end : float
        Swap final payment time (in years).
    Delta : float
        Year-fraction for the swap payment (e.g., 0.25 for 3M).
    strike : float
        Strike rate (the ATM fixed rate for the swap).

    Returns
    -------
    float
        MC-estimated present value of the payer swaption.
    """
    # -- Indexes for time steps --
    steps_expiry = int(T_expiry / dt)
    steps_end    = int(T_swap_end / dt)

    # -- 1) Discount factor from time-0 to time T_swap_end for each path --
    integral_0_to_end = rates_df.iloc[:steps_end].sum(axis=0) * dt
    df_0_to_end = np.exp(-integral_0_to_end)

    # -- 2) Compute the forward swap rate at T_expiry (path-by-path) --
    #    F_swap(T_expiry) = (P(T_expiry) - P(T_swap_end)) / (Delta * P(T_swap_end))
    integral_0_to_exp  = rates_df.iloc[:steps_expiry].sum(axis=0) * dt
    P_expiry_path      = np.exp(-integral_0_to_exp)
    P_end_path         = np.exp(-integral_0_to_end)
    swap_rate_path     = (P_expiry_path - P_end_path) / (Delta * P_end_path)

    # -- 3) Payoff at T_swap_end = max( swap_rate - strike, 0 ) * Delta
    payoff_at_end = np.maximum(swap_rate_path - strike, 0.0) * Delta

    # -- 4) Discount payoff from T_swap_end back to time 0
    discounted_payoff = payoff_at_end * df_0_to_end

    # -- 5) Monte Carlo price = average across all paths
    return discounted_payoff.mean()


def black_atm_swaption_price(F, sigma_swap, A, T_expiry):
    """
    Prices an *ATM* swaption using the Black formula (adapted to interest rates).
    
    For an ATM option (strike K = F), the standard Black formula for a call is:
        Call = A * F * [Phi(d1) - Phi(d2)]
      where 
        d1 = 0.5 * sigma_swap * sqrt(T_expiry)
        d2 = d1 - sigma_swap * sqrt(T_expiry) = -d1
      A = "annuity" = present value of the notional times the accrual factor, discounted
    
    Returns
    -------
    float
        The price of the ATM (payer) swaption.
    """
    # ATM simplification => forward = strike => moneyness = 1 => forward log-moneyness = 0
    d1 = 0.5 * sigma_swap * np.sqrt(T_expiry)
    d2 = -d1
    price = A * F * (norm.cdf(d1) - norm.cdf(d2))
    return price

# ------------------------------------------------------------------------------
# 2) Additional Tools for PART (3b): Bond-Put Implied Vol
# ------------------------------------------------------------------------------

def black_scholes_put_price(S0, K, r, sigma, T):
    """
    Black-Scholes price for a European put on a non-dividend-paying asset.
    S0 : current underlying price
    K  : strike
    r  : risk-free rate (continuously compounded)
    sigma : volatility
    T  : time to maturity
    """
    if T <= 0:
        return max(K - S0, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    # Put price in standard Black-Scholes
    put_val = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_val

def implied_vol_put_bisect(target_price, S0, K, r, T, tol=1e-6):
    """
    Numerically solve for the Black-Scholes implied volatility of a European put,
    using a simple bisection search.
    """
    # Quick checks:
    # If target_price is 0 or negative, implied vol is 0.
    # If target_price is large, might be the vol is huge.
    if target_price < 1e-12:
        return 0.0
    
    vol_lower, vol_upper = 1e-8, 2.0  # Arbitrary bracket for interest-rate vol
    for _ in range(200):
        mid = 0.5 * (vol_lower + vol_upper)
        p_mid = black_scholes_put_price(S0, K, r, mid, T)
        
        if abs(p_mid - target_price) < tol:
            return mid
        
        if p_mid > target_price:
            vol_upper = mid
        else:
            vol_lower = mid
    
    return 0.5 * (vol_lower + vol_upper)

def simulate_bond_put_price(rates_df, dt, T_expiry, T_bond_maturity, strike):
    """
    Using CIR simulation, compute the model price of a European put on a zero-coupon bond:
    - The bond matures at T_bond_maturity (> T_expiry).
    - The option expires at T_expiry.
    - The strike is 'strike' (assume we want an ATM scenario => strike ~ bond's current price).
    
    Steps:
      1) For each path, we find the bond's value at T_expiry:
           BondVal_at_expiry = exp(-∫_{T_expiry to T_bond_maturity} r(u) du)
      2) The put payoff at T_expiry is max(strike - BondVal_at_expiry, 0).
      3) Discount that payoff back to time 0 by multiplying by 
         exp(-∫_{0 to T_expiry} r(u) du) on each path.
      4) Average across paths for the fair value.
    """
    num_steps_expiry = int(T_expiry / dt)
    num_steps_bond   = int(T_bond_maturity / dt)
    
    # integrals from 0->expiry
    integral_0_to_Exp = rates_df.iloc[:num_steps_expiry].sum(axis=0) * dt
    # integrals from 0->bond maturity
    integral_0_to_Bnd = rates_df.iloc[:num_steps_bond].sum(axis=0) * dt
    
    # bond value at T_expiry (pathwise):
    # This is e^[- ∫_{T_expiry to T_bond_maturity} r(u) du]
    # = e^[-( ∫_{0 to T_bond_maturity} r - ∫_{0 to T_expiry} r )]
    bond_val_at_expiry = np.exp(-(integral_0_to_Bnd - integral_0_to_Exp))
    
    # Put payoff at T_expiry
    payoff_at_expiry = np.maximum(strike - bond_val_at_expiry, 0.0)
    
    # Discount payoff from T_expiry back to 0
    discount_0_to_expiry = np.exp(-integral_0_to_Exp)
    discounted_payoff = payoff_at_expiry * discount_0_to_expiry
    
    # Model price = average of discounted payoff
    put_price = discounted_payoff.mean()
    return put_price

# ------------------------------------------------------------------------------
# 3) Main Flow
# ------------------------------------------------------------------------------
def main():
    # ----------------------
    # STEP (1): Simulate CIR up to T_swap_end = 4M
    # ----------------------
    T_total = T_swap_end  # We need rates out to 4 months
    rates_df = simulate_cir(a_calib, b_calib, sigma_calib, r0, T_total, dt, num_paths)
    
    # ----------------------
    # STEP (2): Zero-coupon bond prices at 1M & 4M
    # ----------------------
    P_1M = compute_zero_coupon_price(rates_df, T_swaption_expiry, dt)
    P_4M = compute_zero_coupon_price(rates_df, T_swap_end, dt)
    print(f"P(1M) = {P_1M:.6f},  P(4M) = {P_4M:.6f}")
    
    # ----------------------
    # STEP (3): Calculate the (theoretical) Fair Swap Rate
    #            F_swap = (P_1M - P_4M) / [Delta * P_4M]
    # ----------------------
    F_swap = (P_1M - P_4M) / (Delta * P_4M)
    print(f"Fair Swap Rate (analytical)  = {F_swap:.6f}")
    
    # ----------------------
    # STEP (4): Simulate distribution of forward swap rates at T=1M
    #           Then compute average & std as a proxy for F, sigma.
    # ----------------------
    swap_rates = compute_swap_rate_distribution(
        rates_df, dt, T_swaption_expiry, T_swap_end, Delta
    )
    F_swap_sim = np.mean(swap_rates)
    sigma_swap = np.std(swap_rates)
    print(f"Simulated Fair Swap Rate = {F_swap_sim:.6f}")
    print(f"Swap Rate Volatility     = {sigma_swap:.6f}")
    

    # Getting a pure MC based price
    strike_atm = F_swap_sim.copy()

    mc_swaption_price = calc_swaption_price_mc(
        rates_df=rates_df,
        dt=dt,
        T_expiry=T_swaption_expiry,
        T_swap_end=T_swap_end,
        Delta=Delta,
        strike=strike_atm
    )
    print(f"ATM Swaption Price (Pure MC) = {mc_swaption_price:.6f}")


    # ----------------------
    # STEP (5): Compute the "annuity" factor A for a single payment
    #           = Delta * discount factor to 0 from T=4M
    # ----------------------
    A = Delta * P_4M  # in a 1-payment swap, this is standard
    
    # ----------------------
    # STEP (6): (Q3a) Price the ATM Swaption using the Black formula
    #           If we treat the forward swap rate as lognormal with vol sigma_swap
    # ----------------------
    atm_swaption_price = black_atm_swaption_price(F_swap_sim, sigma_swap, A, T_swaption_expiry)
    print(f"(3a) ATM Swaption Price (Black w/ CIR-avg vol) = {atm_swaption_price:.6f}")
    
    # ----------------------
    # STEP (7): (Q3b) Implied volatility for a put on the 4M zero-coupon bond
    #           a) We find the bond-put price from the CIR model
    #           b) Then invert Black-Scholes for that put to get implied vol
    #
    #   According to typical lecture notes, an ATM put on the bond means strike ~ P_4M
    #   or possibly the forward price P_4M / P_1M.  We'll do a simple "strike = P_4M".
    # ----------------------
    
    # 7a) Model-based bond-put price via simulation
    strike_bond = P_4M   # for an "ATM" put if we treat spot ~ P_4M
    bond_put_price_model = simulate_bond_put_price(
        rates_df, dt, T_swaption_expiry, T_swap_end, strike_bond
    )
    
    # 7b) Back out implied vol under Black-Scholes
    #     Let's interpret the "spot" as the current bond price = P_4M,
    #     time to expiry = 1M, and risk-free ~ 0 for a short horizon 
    #     (or you could do r = -ln(P_1M)/T_swaption_expiry if you prefer).
    S0_bond = P_4M
    K_bond  = P_4M
    T_exp   = T_swaption_expiry
    r_short = 0.0  # or approximate from P_1M if desired
    
    implied_vol_bond = implied_vol_put_bisect(
        target_price=bond_put_price_model, 
        S0=S0_bond, 
        K=K_bond, 
        r=r_short, 
        T=T_exp
    )
    
    print(f"(3b) Model-based Bond Put Price = {bond_put_price_model:.6f}")
    print(f"(3b) Implied Vol for Bond (ATM put) = {implied_vol_bond:.4%}")
    
    # ----------------------
    # Plot: Distribution of Simulated Swap Rates
    # ----------------------
    plt.figure(figsize=(8, 5))
    plt.hist(swap_rates, bins=50, density=True, alpha=0.6, label="Simulated Swap Rates")
    x_vals = np.linspace(min(swap_rates), max(swap_rates), 200)
    plt.plot(x_vals, norm.pdf(x_vals, F_swap_sim, sigma_swap), 'r--', label="Normal Fit")
    plt.xlabel("Swap Rate")
    plt.ylabel("Density")
    plt.title("Distribution of Simulated Swap Rates at 1M")
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
