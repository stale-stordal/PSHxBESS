"""
Joint Spot + FCR Forecast
=========================
Runs one Monte Carlo simulation of correlated spot and FCR price paths,
using parameters pre-calibrated by calibrate_joint.py.

Usage
-----
    python calibrate_joint.py   # once, or when data changes
    python forecast_joint.py    # run as many times as needed

Output
------
stochastic_modelling/results/spot.parquet
stochastic_modelling/results/FCR.parquet
"""

import gc
import os
import time
import threading

import numpy as np
import pandas as pd
from tqdm import tqdm

from stochastic_modelling.spot_price_speed_new import (
    load_prices,
    deseasonalise,
    CSV_FILE, DATE_COL, PRICE_COL,
)
from stochastic_modelling.fcr_pricing_new import (
    load_fcr,
    load_params_from_csv,
    simulate_joint,
    save_joint_paths_to_csv,
    summarise_paths,
    plot_all_forecasts,
    FCR_FILE,
    FCR_ROLLING_WINDOW,
    PARAMS_FILE,
    CORRELATIONS,
)

# Output directory for batch files.
RESULTS_DIR = 'stochastic_modelling/output' #TODO: update to Solstorm dir if needed

# =============================================================================
# USER CONFIG - edit these before each run
# =============================================================================

N_YEARS        = 80      # simulation horizon in years
N_BATCHES      = 5       # number of batches; results saved as spot_<i>.parquet / FCR_<i>.parquet
N_SIMULATIONS  = 2000    # paths per batch (N_BATCHES * N_SIMULATIONS total)
INFLATION_RATE = 0.02    # annual nominal inflation applied to all paths; set to 0.0 to disable

N_FORECAST          = N_YEARS * 8760   # derived - do not edit
SPOT_ROLLING_WINDOW = 8760             # must match calibrate_joint.py

# =============================================================================
# MAIN
# =============================================================================

def _run_with_spinner(bar, fn, *args, **kwargs):
    """Run fn(*args, **kwargs) in a thread while animating bar in the main thread."""
    result = [None]
    exc    = [None]

    def _worker():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    while t.is_alive():
        bar.update(0)   # redraws the bar / spinner
        time.sleep(0.1)
    t.join()
    if exc[0] is not None:
        raise exc[0]
    return result[0]


if __name__ == '__main__':

    # Prompt for an optional filename suffix to avoid overwriting prior runs.
    _suffix_in = input("Optional trailing name for output files (press Enter to skip): ").strip()
    FILE_SUFFIX = f"_{_suffix_in}" if _suffix_in else ""

    t_total_start = time.perf_counter()

    STEPS = [
        "Load & deseasonalise",
        "Load parameters",
        "Compute correlation",
        "Simulate joint paths",
        "Save CSV files",
        "Summary statistics",
        "Plot",
    ]
    overall = tqdm(total=len(STEPS), desc="forecast_joint", unit="step",
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")

    # ------------------------------------------------------------------
    # 1. Load, deseasonalise, and apply two-factor L/Z split
    # ------------------------------------------------------------------
    overall.set_postfix_str(STEPS[0])
    t0 = time.perf_counter()
    df_spot_raw = load_prices(CSV_FILE, DATE_COL, PRICE_COL)
    df_fcr_raw  = load_fcr(FCR_FILE)
    df_spot_seas, ltsc_spot = deseasonalise(df_spot_raw.copy(), n_harmonics=2, hours_per_year=8760)
    df_fcr_seas,  ltsc_fcr  = deseasonalise(df_fcr_raw.copy(),  n_harmonics=2, hours_per_year=8760)
    X_spot = df_spot_seas['X'].values
    X_fcr  = df_fcr_seas['X'].values

    # Two-factor split - must use same window as calibrate_joint.py
    w_s    = min(SPOT_ROLLING_WINDOW, len(X_spot) // 2)
    L_spot = (pd.Series(X_spot, index=df_spot_seas.index)
              .rolling(window=w_s, center=True, min_periods=w_s // 4)
              .mean().ffill().bfill().values)
    Z_spot  = X_spot - L_spot
    L0_spot = float(L_spot[-1])

    w_f   = min(FCR_ROLLING_WINDOW, len(X_fcr) // 2)
    L_fcr = (pd.Series(X_fcr, index=df_fcr_seas.index)
             .rolling(window=w_f, center=True, min_periods=w_f // 4)
             .mean().ffill().bfill().values)
    Z_fcr  = X_fcr - L_fcr
    L0_fcr = float(L_fcr[-1])

    overall.update(1)
    tqdm.write(f"  {STEPS[0]:<28} {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # 2. Load pre-calibrated parameters
    # ------------------------------------------------------------------
    overall.set_postfix_str(STEPS[1])
    t0 = time.perf_counter()
    all_thetas = load_params_from_csv(PARAMS_FILE)
    theta_spot = all_thetas['Spot']
    theta_fcr  = all_thetas['FCR']
    _meta = pd.read_csv(PARAMS_FILE)
    _fcr_row = _meta[_meta["series"] == "FCR"]
    _fcr_regimes = int(_fcr_row["n_regimes"].iloc[0]) if "n_regimes" in _meta.columns else 3
    _sim_fn = simulate_joint
    overall.update(1)
    tqdm.write(f"  {STEPS[1]:<28} {time.perf_counter() - t0:.2f}s  (FCR {_fcr_regimes}-regime)")

    # ------------------------------------------------------------------
    # 3. Correlation
    # ------------------------------------------------------------------
    overall.set_postfix_str(STEPS[2])
    t0 = time.perf_counter()
    n = min(len(Z_spot), len(Z_fcr))
    CORRELATIONS['FCR'] = float(np.corrcoef(np.diff(Z_spot[:n]), np.diff(Z_fcr[:n]))[0, 1])
    overall.update(1)
    tqdm.write(f"  {STEPS[2]:<28} {time.perf_counter() - t0:.2f}s  "
               f"(ρ = {CORRELATIONS['FCR']:.4f})")

    # ------------------------------------------------------------------
    # 4 & 5. Joint simulation and save (in batches)
    # ------------------------------------------------------------------
    _spot_bounds = (float(df_spot_raw['price'].min()), float(df_spot_raw['price'].max()))
    _fcr_bounds  = (float(df_fcr_raw['price'].min()),  float(df_fcr_raw['price'].max()))
    tqdm.write(f"  Inflation rate: {INFLATION_RATE:.1%} per year")
    tqdm.write(f"  Running {N_BATCHES} batches × {N_SIMULATIONS} paths "
               f"({N_BATCHES * N_SIMULATIONS} total)")

    overall.set_postfix_str(STEPS[3])
    t_sim_total = 0.0
    t_save_total = 0.0
    spot_paths = fcr_paths = None
    for batch_idx in range(1, N_BATCHES + 1):
        # --- simulate
        # Free the previous batch's arrays before allocating the next one.
        # Without this, the old paths stay referenced while simulate_joint
        # allocates the new ones, doubling peak memory and risking an OOM kill.
        spot_paths = fcr_paths = None
        gc.collect()
        t0 = time.perf_counter()
        sim_bar = tqdm(
            total=0,
            desc=f"  Batch {batch_idx}/{N_BATCHES}: simulating {N_SIMULATIONS} paths × {N_FORECAST} steps",
            bar_format="{desc} … {elapsed}", leave=False,
        )
        spot_paths, _, fcr_paths, _ = _run_with_spinner(
            sim_bar, _sim_fn,
            theta_spot = theta_spot,
            theta_fcr  = theta_fcr,
            x0_spot    = float(Z_spot[-1]),
            x0_fcr     = float(Z_fcr[-1]),
            L0_spot    = L0_spot,
            L0_fcr     = L0_fcr,
            n_steps    = N_FORECAST,
            n_paths    = N_SIMULATIONS,
            rho_corr   = CORRELATIONS['FCR'],
        )
        sim_bar.close()
        t_sim_batch = time.perf_counter() - t0
        t_sim_total += t_sim_batch
        tqdm.write(f"  Batch {batch_idx}/{N_BATCHES} simulated in {t_sim_batch:.2f}s")

        # --- save and rename to add _<batch_idx> suffix
        overall.set_postfix_str(STEPS[4])
        t0 = time.perf_counter()
        save_joint_paths_to_csv(
            spot_paths, fcr_paths,
            df_spot_seas, df_fcr_seas,
            ltsc_spot, ltsc_fcr,
            out_dir=RESULTS_DIR,
            inflation_rate=INFLATION_RATE,
            spot_price_bounds=_spot_bounds,
            fcr_price_bounds=_fcr_bounds,
        )
        for base in ('spot', 'FCR'):
            src = os.path.join(RESULTS_DIR, f'{base}.parquet')
            dst = os.path.join(RESULTS_DIR, f'{base}_{batch_idx}{FILE_SUFFIX}.parquet')
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)
        t_save_batch = time.perf_counter() - t0
        t_save_total += t_save_batch
        tqdm.write(f"  Batch {batch_idx}/{N_BATCHES} saved as spot_{batch_idx}{FILE_SUFFIX}.parquet / "
                   f"FCR_{batch_idx}{FILE_SUFFIX}.parquet in {t_save_batch:.2f}s")
        overall.set_postfix_str(STEPS[3])

    overall.update(1)
    tqdm.write(f"  {STEPS[3]:<28} {t_sim_total:.2f}s")
    overall.update(1)
    tqdm.write(f"  {STEPS[4]:<28} {t_save_total:.2f}s  -> {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 6. Summary statistics
    # ------------------------------------------------------------------
    overall.set_postfix_str(STEPS[5])
    t0 = time.perf_counter()
    results_dict = {
        'Spot': summarise_paths(spot_paths),
        'FCR':  summarise_paths(fcr_paths),
    }
    overall.update(1)
    tqdm.write(f"  {STEPS[5]:<28} {time.perf_counter() - t0:.2f}s")

    for name, res in results_dict.items():
        tqdm.write(f"\n  {name} forecast (deseasonalised):")
        for step in [1, 5, 10, N_FORECAST]:
            if step < len(res['mean']):
                tqdm.write(f"    Step {step:>3} - mean: {res['mean'][step]:>7.2f}  "
                           f"80% CI: [{res['lower'][step]:>7.2f}, {res['upper'][step]:>7.2f}]")

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    overall.set_postfix_str(STEPS[6])
    t0 = time.perf_counter()
    plot_all_forecasts({'Spot': Z_spot, 'FCR': Z_fcr}, results_dict)
    overall.update(1)
    tqdm.write(f"  {STEPS[6]:<28} {time.perf_counter() - t0:.2f}s")

    overall.set_postfix_str("done")
    overall.close()
    tqdm.write(f"\n  Total: {time.perf_counter() - t_total_start:.2f}s")
