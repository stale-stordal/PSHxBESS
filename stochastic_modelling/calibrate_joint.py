"""
Joint Calibration — Spot + FCR
================================
Run this once (or whenever the underlying price data changes) to calibrate
both MRS models via EM and save the parameters to disk.

    python calibrate_joint.py

Output
------
stochastic_modelling/results/params.csv
    One row per series (Spot, FCR) with all MRS parameters.

After this file exists, forecast_joint.py loads it directly and skips EM,
so every simulation run uses identical calibrated parameters.
"""

import numpy as np

from stochastic_modelling.spot_price_speed_new import (
    load_prices,
    CSV_FILE, DATE_COL, PRICE_COL,
    INITIAL_PARAMS as INITIAL_PARAMS_SPOT,
)
from stochastic_modelling.fcr_pricing_new import (
    load_fcr,
    calibrate_series,
    estimate_correlation,
    save_all_params_to_csv,
    plot_convergence,
    FCR_FILE,
    FCR_ROLLING_WINDOW,
    INITIAL_PARAMS_FCR,
    PARAMS_FILE,
    CORRELATIONS,
)


# FCR model selection:  "3regime" uses base+spike+drop (fcr_pricing_new.py)
#                       "2regime" uses base+spike only  (fcr_pricing_tworegimes.py)
FCR_MODEL = "3regime"

CALIBRATION_START       = "2000-01-01"   # spot start — full pre-crisis history
CALIBRATION_END         = None           # use all data up to present
FCR_CALIBRATION_START   = "2015-01-01"   # FCR start  — all available FCR data
FCR_CALIBRATION_END     = None           # use all data up to present
SPOT_ROLLING_WINDOW = 8760      # hours (~2 years) for spot L/Z split

# Weighted EM: crisis observations receive CRISIS_WEIGHT instead of 1.0.
# Pre-crisis data drives the calibrated parameters; crisis years are included
# but with reduced influence so recent market behaviour is not ignored entirely.
SPOT_CRISIS_START = "2020-01-01"   # spot: energy crisis from 2020
FCR_CRISIS_START  = "2021-01-01"   # FCR: price surge started 2021
CRISIS_WEIGHT     = 0.5            # crisis obs weight (0 = exclude, 1 = equal)

# Spike/drop threshold percentiles.
# Only the top SPIKE_PCT% of Z can be spikes; only the bottom DROP_PCT% can be drops.
# 90/10 → each tail is ~10% of observations; base regime owns 80%.
# Lower values (e.g. 75/25) make regimes less selective and bias alpha1/beta1.
SPIKE_PCT = 85  # forrige er med 90/10
DROP_PCT  = 15

# EM convergence settings.
# Reduce MAX_ITER or raise TOL for faster (less precise) calibration.
# 200 / 1e-4 is a good quick-check run; 500 / 1e-6 is production quality.
MAX_ITER = 1000    
EM_TOL   = 1e-4

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    # ------------------------------------------------------------------
    # 1. Calibrate spot on Z (deseasonalised minus rolling mean)
    # ------------------------------------------------------------------
    import pandas as pd
    print("Loading and calibrating spot price (EM)...")
    df_spot_raw = load_prices(CSV_FILE, DATE_COL, PRICE_COL)
    if CALIBRATION_START is not None:
        df_spot_raw = df_spot_raw[df_spot_raw.index >= CALIBRATION_START]
    if CALIBRATION_END is not None:
        df_spot_raw = df_spot_raw[df_spot_raw.index <= CALIBRATION_END]

    spot_weights = np.where(df_spot_raw.index >= SPOT_CRISIS_START,
                            CRISIS_WEIGHT, 1.0)
    n_spot_crisis = (spot_weights < 1.0).sum()
    print(f"  Crisis downweight ({SPOT_CRISIS_START}+): "
          f"{n_spot_crisis:,} obs at w={CRISIS_WEIGHT}  "
          f"({n_spot_crisis / len(spot_weights):.1%} of sample)")

    theta_spot, Z_spot, L0_spot, _, _, logliks_spot = calibrate_series(
        df_spot_raw, INITIAL_PARAMS_SPOT, label="Spot",
        rolling_window=SPOT_ROLLING_WINDOW,
        obs_weights=spot_weights,
        spike_pct=SPIKE_PCT, drop_pct=DROP_PCT,
        max_iter=MAX_ITER, tol=EM_TOL,
    )

    # ------------------------------------------------------------------
    # 2. Calibrate FCR on Z
    # ------------------------------------------------------------------
    print("\nLoading and calibrating FCR (EM)...")
    df_fcr_raw = load_fcr(FCR_FILE)
    if FCR_CALIBRATION_START is not None:
        df_fcr_raw = df_fcr_raw[df_fcr_raw.index >= FCR_CALIBRATION_START]
    if FCR_CALIBRATION_END is not None:
        df_fcr_raw = df_fcr_raw[df_fcr_raw.index <= FCR_CALIBRATION_END]

    fcr_weights = np.where(df_fcr_raw.index >= FCR_CRISIS_START,
                           CRISIS_WEIGHT, 1.0)
    n_fcr_crisis = (fcr_weights < 1.0).sum()
    print(f"  Crisis downweight ({FCR_CRISIS_START}+): "
          f"{n_fcr_crisis:,} obs at w={CRISIS_WEIGHT}  "
          f"({n_fcr_crisis / len(fcr_weights):.1%} of sample)")
    
    theta_fcr, Z_fcr, L0_fcr, _, _, logliks_fcr = calibrate_series(
        df_fcr_raw, INITIAL_PARAMS_FCR, label="FCR",
        rolling_window=FCR_ROLLING_WINDOW,
        obs_weights=fcr_weights,
        spike_pct=SPIKE_PCT, drop_pct=DROP_PCT,
        max_iter=MAX_ITER, tol=EM_TOL,
        )

    # ------------------------------------------------------------------
    # 3. Estimate correlation on Z residuals — aligned by date
    # ------------------------------------------------------------------
    print("\n--- Spot-FCR correlation (Z residuals) ---")
    Z_spot_ser = (pd.Series(Z_spot, index=df_spot_raw.index)
                  .loc[~df_spot_raw.index.duplicated(keep='first')])
    Z_fcr_ser  = (pd.Series(Z_fcr,  index=df_fcr_raw.index)
                  .loc[~df_fcr_raw.index.duplicated(keep='first')])
    common_idx = Z_spot_ser.index.intersection(Z_fcr_ser.index)
    CORRELATIONS['FCR'] = estimate_correlation(
        Z_spot_ser.loc[common_idx].values,
        Z_fcr_ser.loc[common_idx].values,
        label="FCR",
    )

    # ------------------------------------------------------------------
    # 4. Diagnostics
    # ------------------------------------------------------------------
    print("\n--- Calibration check ---")
    for name, theta, Z in [("Spot", theta_spot, Z_spot), ("FCR", theta_fcr, Z_fcr)]:
        print(f"{name}: alpha1/beta1 = {theta['alpha1']/theta['beta1']:.2f}, "
              f"Z mean = {Z.mean():.2f}, beta1 = {theta['beta1']:.4f}")

    print("\n--- Z quantiles ---")
    for name, Z in [("Spot", Z_spot), ("FCR", Z_fcr)]:
        print(f"{name}: p10={np.percentile(Z,10):.2f}, p25={np.percentile(Z,25):.2f}, "
              f"p50={np.percentile(Z,50):.2f}, p75={np.percentile(Z,75):.2f}, "
              f"p90={np.percentile(Z,90):.2f}")

    # ------------------------------------------------------------------
    # 5. EM convergence plots
    # ------------------------------------------------------------------
    plot_convergence({"Spot": logliks_spot, "FCR": logliks_fcr})

    # ------------------------------------------------------------------
    # 6. Save parameters
    # ------------------------------------------------------------------

    save_all_params_to_csv(
        {"Spot": theta_spot, "FCR": theta_fcr},
        out_path=PARAMS_FILE,
    )
    print(f"\nDone. FCR_MODEL={FCR_MODEL}. Run forecast_joint.py to simulate.")
