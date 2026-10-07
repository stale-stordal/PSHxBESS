"""
Shared runner for all sensitivity_*.py scripts.

All simulation constants (SPIKE_DECAY_RATE, SPIKE_FLOOR_FRAC,
SIGMA_DECAY_RATE, SIGMA_FLOOR_FRAC, GOU_SIGMA, FCR_GOU_SIGMA) are
module-level globals in fcr_pricing_new that are looked up at call
time — so monkey-patching fcr_mod before calling run() is sufficient
to override any of them for both spot and FCR.

Baseline values (from fcr_pricing_new / spot_price_speed_new):
    SPIKE_DECAY_RATE = 0.1    SIGMA_DECAY_RATE = 0.1
    SPIKE_FLOOR_FRAC = 0.2    SIGMA_FLOOR_FRAC = 0.2
    GOU_SIGMA        = 0.2    FCR_GOU_SIGMA    = 0.2
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from numba import njit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCH_DIR  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, STOCH_DIR)

import stochastic_modelling.fcr_pricing_new as fcr_mod
from stochastic_modelling.spot_price_speed_new import (
    load_prices, deseasonalise,
    CSV_FILE, DATE_COL, PRICE_COL,
)
from stochastic_modelling.fcr_pricing_new import (
    load_fcr, load_params_from_csv, simulate_joint, save_joint_paths_to_csv,
    FCR_FILE, FCR_ROLLING_WINDOW, PARAMS_FILE,
)
from stochastic_modelling.forecast_joint import RESULTS_DIR

N_YEARS             = 80
N_SIMULATIONS       = 1000
INFLATION_RATE      = 0.02
SPOT_ROLLING_WINDOW = 8760

# Common random numbers: every scenario is seeded identically so the ONLY
# difference between output files is the swept parameter, not Monte-Carlo noise.
SEED = 42

# Knobs a scenario may override. Snapshotted at import (canonical baseline) so
# run() can reset them on every call — overrides never leak between scenarios
# when several run() calls share one process (e.g. run_all.py).
_KNOBS = (
    "SPIKE_DECAY_RATE", "SIGMA_DECAY_RATE",
    "SPIKE_FLOOR_FRAC", "SIGMA_FLOOR_FRAC",
    "GOU_SIGMA",        "FCR_GOU_SIGMA",
    "GOU_KAPPA",        "FCR_GOU_KAPPA",
)
_DEFAULTS = {k: getattr(fcr_mod, k) for k in _KNOBS}


@njit(cache=True)
def _seed_numba(s):
    # Seeds Numba's internal RNG (used inside _simulate_joint_core). This is a
    # SEPARATE stream from NumPy's global RNG (used by simulate_geo_ou), so both
    # must be seeded for a fully reproducible joint simulation.
    np.random.seed(s)


def run(run_name: str, overrides: dict) -> None:
    """
    Parameters
    ----------
    run_name  : output sub-folder name under sensitivity_runs/results/
    overrides : {attr_name: value} applied to fcr_mod before simulation
    """
    out_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Run: {run_name}")
    # Reset every knob to its canonical default first, so overrides from a
    # previous run() call in the same process cannot leak into this one.
    for attr, val in _DEFAULTS.items():
        setattr(fcr_mod, attr, val)
    for attr, val in overrides.items():
        setattr(fcr_mod, attr, val)
        print(f"  override  fcr_mod.{attr} = {val}")
    print(f"  seed = {SEED} (common random numbers across scenarios)")
    print(f"{'='*55}")

    n_forecast = N_YEARS * 8760

    # ----------------------------------------------------------------
    # Load + deseasonalise
    # ----------------------------------------------------------------
    df_spot_raw = load_prices(CSV_FILE, DATE_COL, PRICE_COL)
    df_fcr_raw  = load_fcr(FCR_FILE)
    df_spot_ds, ltsc_spot = deseasonalise(df_spot_raw.copy(), n_harmonics=2, hours_per_year=8760)
    df_fcr_ds,  ltsc_fcr  = deseasonalise(df_fcr_raw.copy(),  n_harmonics=2, hours_per_year=8760)
    X_spot = df_spot_ds["X"].values
    X_fcr  = df_fcr_ds["X"].values

    # ----------------------------------------------------------------
    # Two-factor L/Z split
    # ----------------------------------------------------------------
    w_s    = min(SPOT_ROLLING_WINDOW, len(X_spot) // 2)
    L_spot = (pd.Series(X_spot, index=df_spot_ds.index)
              .rolling(w_s, center=True, min_periods=w_s // 4)
              .mean().ffill().bfill().values)
    Z_spot  = X_spot - L_spot
    L0_spot = float(L_spot[-1])

    w_f   = min(FCR_ROLLING_WINDOW, len(X_fcr) // 2)
    L_fcr = (pd.Series(X_fcr, index=df_fcr_ds.index)
             .rolling(w_f, center=True, min_periods=w_f // 4)
             .mean().ffill().bfill().values)
    Z_fcr  = X_fcr - L_fcr
    L0_fcr = float(L_fcr[-1])

    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------
    all_thetas = load_params_from_csv(PARAMS_FILE)
    theta_spot = all_thetas["Spot"]
    theta_fcr  = all_thetas["FCR"]
    _meta = pd.read_csv(PARAMS_FILE)
    _fcr_regimes = (
        int(_meta.loc[_meta["series"] == "FCR", "n_regimes"].iloc[0])
        if "n_regimes" in _meta.columns else 3
    )
    _sim_fn = simulate_joint
    print(f"  FCR model: {_fcr_regimes}-regime")

    # ----------------------------------------------------------------
    # Correlation
    # ----------------------------------------------------------------
    n = min(len(Z_spot), len(Z_fcr))
    rho_corr = float(np.corrcoef(np.diff(Z_spot[:n]), np.diff(Z_fcr[:n]))[0, 1])
    print(f"  rho(spot, FCR) = {rho_corr:.4f}")

    # ----------------------------------------------------------------
    # Simulate
    # ----------------------------------------------------------------
    print(f"  Simulating {N_SIMULATIONS} paths × {N_YEARS} years ...")
    np.random.seed(SEED)   # NumPy RNG  -> simulate_geo_ou (level factor L)
    _seed_numba(SEED)      # Numba RNG  -> _simulate_joint_core (Z + regimes)
    t0 = time.perf_counter()
    spot_paths, _, fcr_paths, _ = _sim_fn(
        theta_spot = theta_spot,
        theta_fcr  = theta_fcr,
        x0_spot    = float(Z_spot[-1]),
        x0_fcr     = float(Z_fcr[-1]),
        L0_spot    = L0_spot,
        L0_fcr     = L0_fcr,
        n_steps    = n_forecast,
        n_paths    = N_SIMULATIONS,
        rho_corr   = rho_corr,
    )
    print(f"  Done in {time.perf_counter() - t0:.1f}s")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    _spot_bounds = (float(df_spot_raw["price"].min()), float(df_spot_raw["price"].max()))
    _fcr_bounds  = (float(df_fcr_raw["price"].min()),  float(df_fcr_raw["price"].max()))
    save_joint_paths_to_csv(
        spot_paths, fcr_paths,
        df_spot_ds, df_fcr_ds,
        ltsc_spot, ltsc_fcr,
        out_dir          = out_dir,
        inflation_rate   = INFLATION_RATE,
        spot_price_bounds = _spot_bounds,
        fcr_price_bounds  = _fcr_bounds,
    )
    print(f"  Saved to {out_dir}/")
