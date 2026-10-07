"""
Generate spot_10_paths.csv and FCR_10_paths.csv using the current model pipeline.

Runs 10 simulation paths × 50 years with the same re-seasonalisation,
inflation, and price-cap logic as test_run.py, then saves to this folder.
Run this whenever model parameters or code change, then re-run plot_price_paths.py.

Usage (from repo root):
    python stochastic_modelling/analysis/figures_10simulations_results/generate_10_paths.py
"""
import os
import sys
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCH_DIR  = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # stochastic_modelling/
sys.path.insert(0, STOCH_DIR)

from stochastic_modelling.spot_price_speed_new import (
    load_prices, deseasonalise, _reseasonalise,
    CSV_FILE, DATE_COL, PRICE_COL,
    INITIAL_PARAMS, GOU_THETA, GOU_KAPPA, GOU_SIGMA,
)
from stochastic_modelling.fcr_pricing_new import (
    load_fcr, simulate_joint, load_params_from_csv,
    FCR_FILE, FCR_ROLLING_WINDOW, PARAMS_FILE,
    INITIAL_PARAMS_FCR,
)
from fcr_pricing_tworegimes import simulate_joint_2r

N_PATHS        = 10
N_YEARS        = 50
N_STEPS        = N_YEARS * 8760
INFLATION_RATE = 0.02
SPOT_ROLLING_WINDOW = 8760

# ---------------------------------------------------------------------------
# Load & deseasonalise
# ---------------------------------------------------------------------------
print("Loading data...")
df_spot_raw = load_prices(CSV_FILE, DATE_COL, PRICE_COL)
df_fcr_raw  = load_fcr(FCR_FILE)

df_spot_ds, ltsc_spot = deseasonalise(df_spot_raw.copy(), n_harmonics=2, hours_per_year=8760)
df_fcr_ds,  ltsc_fcr  = deseasonalise(df_fcr_raw.copy(),  n_harmonics=2, hours_per_year=8760)

X_spot = df_spot_ds["X"].values
X_fcr  = df_fcr_ds["X"].values

w_s    = min(SPOT_ROLLING_WINDOW, len(X_spot) // 2)
L_spot = (pd.Series(X_spot, index=df_spot_ds.index)
          .rolling(window=w_s, center=True, min_periods=w_s // 4)
          .mean().ffill().bfill().values)
Z_spot  = X_spot - L_spot
L0_spot = float(L_spot[-1])

w_f   = min(FCR_ROLLING_WINDOW, len(X_fcr) // 2)
L_fcr = (pd.Series(X_fcr, index=df_fcr_ds.index)
         .rolling(window=w_f, center=True, min_periods=w_f // 4)
         .mean().ffill().bfill().values)
Z_fcr  = X_fcr - L_fcr
L0_fcr = float(L_fcr[-1])

# ---------------------------------------------------------------------------
# Load calibrated parameters
# ---------------------------------------------------------------------------
if os.path.exists(PARAMS_FILE):
    print(f"Loading calibrated parameters from {PARAMS_FILE}")
    _params    = load_params_from_csv(PARAMS_FILE)
    theta_spot = _params["Spot"]
    theta_fcr  = _params["FCR"]
    theta_spot["P"]   = np.array(theta_spot["P"],   dtype=float)
    theta_spot["rho"] = np.array(theta_spot["rho"], dtype=float)
    theta_fcr["P"]    = np.array(theta_fcr["P"],    dtype=float)
    theta_fcr["rho"]  = np.array(theta_fcr["rho"],  dtype=float)
    _meta = pd.read_csv(PARAMS_FILE)
    _fcr_row = _meta[_meta["series"] == "FCR"]
    _fcr_regimes = int(_fcr_row["n_regimes"].iloc[0]) if "n_regimes" in _meta.columns else 3
    _sim_fn = simulate_joint_2r if _fcr_regimes == 2 else simulate_joint
    print(f"  FCR model: {_fcr_regimes}-regime")
else:
    print(f"WARNING: {PARAMS_FILE} not found — using INITIAL_PARAMS")
    theta_spot = dict(INITIAL_PARAMS)
    theta_spot["q2"]  = float(np.percentile(Z_spot, 75))
    theta_spot["q3"]  = float(np.percentile(Z_spot, 25))
    theta_spot["P"]   = np.array(theta_spot["P"],   dtype=float)
    theta_spot["rho"] = np.array(theta_spot["rho"], dtype=float)
    theta_fcr = dict(INITIAL_PARAMS_FCR)
    theta_fcr["q2"]  = float(np.percentile(Z_fcr, 75))
    theta_fcr["q3"]  = float(np.percentile(Z_fcr, 25))
    theta_fcr["P"]   = np.array(theta_fcr["P"],    dtype=float)
    theta_fcr["rho"] = np.array(theta_fcr["rho"],  dtype=float)
    _sim_fn = simulate_joint

# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------
n_overlap = min(len(Z_spot), len(Z_fcr))
rho_corr  = float(np.corrcoef(np.diff(Z_spot[:n_overlap]),
                               np.diff(Z_fcr[:n_overlap]))[0, 1])
print(f"Correlation rho = {rho_corr:.4f}")
print(f"Simulating {N_PATHS} paths × {N_YEARS} years...")

spot_paths, _, fcr_paths, _ = _sim_fn(
    theta_spot=theta_spot,
    theta_fcr=theta_fcr,
    x0_spot=float(Z_spot[-1]),
    x0_fcr=float(Z_fcr[-1]),
    L0_spot=L0_spot,
    L0_fcr=L0_fcr,
    n_steps=N_STEPS,
    n_paths=N_PATHS,
    rho_corr=rho_corr,
)
print("Simulation done.")

# ---------------------------------------------------------------------------
# Re-seasonalise + inflation + price cap  (identical to test_run.py)
# ---------------------------------------------------------------------------
reseason_spot, future_dates_spot = _reseasonalise(spot_paths, df_spot_ds, ltsc_spot)
reseason_fcr,  future_dates_fcr  = _reseasonalise(fcr_paths,  df_fcr_ds,  ltsc_fcr)

if INFLATION_RATE != 0.0:
    t_years       = np.arange(1, N_STEPS + 1) / 8760
    infl          = (1.0 + INFLATION_RATE) ** t_years
    reseason_spot = reseason_spot * infl[:, np.newaxis]
    reseason_fcr  = reseason_fcr  * infl[:, np.newaxis]

spot_price_min = float(df_spot_raw["price"].min())
spot_price_max = float(df_spot_raw["price"].max())
fcr_price_min  = float(df_fcr_raw["price"].min())
fcr_price_max  = float(df_fcr_raw["price"].max())

reseason_spot = np.clip(reseason_spot, spot_price_min, spot_price_max)
reseason_fcr  = np.clip(reseason_fcr,  fcr_price_min,  fcr_price_max)

print(f"Spot after clip: min={reseason_spot.min():.1f}  max={reseason_spot.max():.1f}  mean={reseason_spot.mean():.1f}")
print(f"FCR  after clip: min={reseason_fcr.min():.1f}   max={reseason_fcr.max():.1f}   mean={reseason_fcr.mean():.1f}")

# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
path_cols = [f"path_{i}" for i in range(N_PATHS)]

spot_df = pd.DataFrame(reseason_spot, index=future_dates_spot, columns=path_cols)
spot_df.index.name = "timestamp"
spot_out = os.path.join(SCRIPT_DIR, "spot_10_paths.csv")
spot_df.to_csv(spot_out)
print(f"Saved {spot_out}")

fcr_df = pd.DataFrame(reseason_fcr, index=future_dates_fcr, columns=path_cols)
fcr_df.index.name = "timestamp"
fcr_out = os.path.join(SCRIPT_DIR, "FCR_10_paths.csv")
fcr_df.to_csv(fcr_out)
print(f"Saved {fcr_out}")
print("Done. Re-run plot_price_paths.py to update the figure.")
