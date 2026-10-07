"""
MRS Model for FCR Prices
========================
Based on: Bakke & Fleten (2016), Section 3.1.2

Calibrates FCR (Frequency Containment Reserve) as an independent
three-regime MRS model (same structure as the spot price model):

    FCR   — Frequency Containment Reserve (single price)
             data/prices/FCR.csv   columns: Timestamp, Price_FCR

The series is deseasonalised, calibrated via EM, and can be forecast
independently or jointly with the spot price via simulate_joint().

Correlation with spot (Bakke & Fleten p.490):
    Set in CORRELATIONS dict below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit

from stochastic_modelling.spot_price_speed_new import (
    Calibrator,
    deseasonalise,
    save_paths_to_csv,
    simulate_geo_ou,
    SPIKE_DECAY_RATE, SPIKE_FLOOR_FRAC,
    GOU_THETA, GOU_KAPPA, GOU_SIGMA,
)

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR    = os.path.join(os.path.dirname(__file__), "../data/prices/FCR")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
N_HISTORY   = 150  # How many historical steps to show in forecast plots

# Spike/drop sigma decay for FCR — only spike (r=2) and drop (r=3) regimes decay.
# Base regime (r=1) is mean-reverting and needs no decay (consistent with spot model).
# FCR market is less structurally volatile than spot, so floor fraction is higher.
SIGMA_DECAY_RATE = 0.1   # λ — annual decay rate
SIGMA_FLOOR_FRAC = 0.2  # σ_∞ = SIGMA_FLOOR_FRAC × σ̂  (long-run floor)
# Spot-side decay in joint simulation is imported: SPIKE_DECAY_RATE, SPIKE_FLOOR_FRAC

# FCR long-term factor (GOU) — tune against market expectations for FCR prices
FCR_GOU_THETA = 200.0  # mean from 2015-2025 = 118 — slightly lower long-term mean
FCR_GOU_KAPPA = 0.05   # mean-reversion speed (same as spot)
FCR_GOU_SIGMA = 0.2   # annual volatility of the FCR level
FCR_ROLLING_WINDOW = 8760      # hours used for rolling mean to extract L_t (1 year; shorter than spot due to limited history)

# File paths
FCR_FILE  = os.path.join(os.path.dirname(__file__), "input/FCR_NO2_NOK.csv")

# Column definitions
FCR_COLS  = {"timestamp": "Timestamp",
             "price":     "Price_FCR"}

# Correlation with spot base regime (Bakke & Fleten p.490)
# Estimate empirically with estimate_correlation() and fill in here.
CORRELATIONS = {
    "FCR": 0.0,
}

PARAMS_FILE   = f"{RESULTS_DIR}/params.csv"  # Written by calibrate_joint.py


# =============================================================================
# DATA LOADING — one function per file type
# =============================================================================

def load_fcr(filepath=FCR_FILE):
    """
    Load FCR CSV and return a single DataFrame.

    CSV format: Timestamp, Price_FCR
    Returns: df  (with DatetimeIndex and 'price' column)
    """
    df = pd.read_csv(filepath, parse_dates=[FCR_COLS["timestamp"]])
    df = df.sort_values(FCR_COLS["timestamp"]).reset_index(drop=True)
    df.index  = df[FCR_COLS["timestamp"]]
    df        = df[[FCR_COLS["price"]]].copy()
    df.columns = ["price"]

    print(f"FCR loaded: {len(df)} observations")
    print(f"  price: [{df['price'].min():.2f}, {df['price'].max():.2f}]")
    return df


# =============================================================================
# CALIBRATION HELPER — wraps deseasonalise + Calibrator.fit for one series
# =============================================================================

def calibrate_series(df, initial_params, label="series",
                     max_iter=500, tol=1e-6, rolling_window=None,
                     obs_weights=None, spike_pct=90, drop_pct=10):
    """
    Deseasonalise, optionally apply two-factor L/Z split, and calibrate.

    Parameters
    ----------
    df             : DataFrame with 'price' column and DatetimeIndex
    initial_params : dict — starting EM parameters
    label          : str  — name used in print output
    max_iter, tol  : EM convergence settings
    rolling_window : int or None — if given, computes L_t as rolling mean and
                     calibrates MRS on Z_t = X_t - L_t (two-factor model)
    obs_weights    : array-like of length len(df), or None — passed to
                     Calibrator.fit(); values < 1 downweight observations

    Returns
    -------
    theta       : dict — calibrated parameters
    Z           : ndarray — series used for calibration (Z_t or X_t)
    L0          : float  — last value of L_t (0.0 if no rolling_window)
    df_out      : DataFrame — deseasonalise() output (needed for re-seasonalisation)
    ltsc_coeffs : ndarray — LTSC coefficients
    logliks     : list — log-likelihood per EM iteration
    """
    print(f"\n{'='*55}")
    print(f"Calibrating: {label}")
    print(f"{'='*55}")

    df_out, ltsc_coeffs = deseasonalise(df.copy(), n_harmonics=2,
                                        hours_per_year=8760)
    X = df_out["X"].values

    if rolling_window is not None:
        import pandas as pd
        w = min(rolling_window, len(X) // 2)
        L = (pd.Series(X, index=df_out.index)
             .rolling(window=w, center=True, min_periods=w // 4)
             .mean().ffill().bfill().values)
        Z  = X - L
        L0 = float(L[-1])
        params = dict(initial_params)
    else:
        Z  = X
        L0 = 0.0
        params = dict(initial_params)

    Z_fit = Z
    print(f"  Z mean={Z.mean():.3f}  std={Z.std():.3f}  "
          f"p10={float(np.percentile(Z,10)):.3f}  "
          f"p90={float(np.percentile(Z,90)):.3f}")

    # Align obs_weights to Z_fit (same length as Z / df)
    fit_weights = None
    if obs_weights is not None:
        fit_weights = np.asarray(obs_weights, dtype=float)
        if len(fit_weights) != len(Z_fit):
            raise ValueError(
                f"obs_weights length {len(fit_weights)} != Z length {len(Z_fit)}"
            )

    calibrator = Calibrator(max_iter=max_iter, tol=tol, verbose=True)
    theta, logliks = calibrator.fit(Z_fit, params, obs_weights=fit_weights,
                                    spike_pct=spike_pct, drop_pct=drop_pct)

    print(f"\n  Check: alpha1/beta1 = {theta['alpha1']/theta['beta1']:.4f}  "
          f"(Z mean = {Z.mean():.4f})")

    return theta, Z, L0, df_out, ltsc_coeffs, logliks


# =============================================================================
# JOINT SIMULATION WITH SPOT — JIT-compiled core
# (Bakke & Fleten 2016, Section 3.1.2, Equation 8)
#
# FCR base innovation is correlated with the spot innovation via:
#   eps_fcr = ρ * eps_spot + √(1-ρ²) * eps_indep
# =============================================================================

@njit(cache=True)
def _simulate_joint_core(n_steps, n_paths,
                          x0_spot, x0_fcr,
                          Ps, rho_init_s,
                          a1s, b1s, g1s, a2s, a3s, q2s, q3s,
                          s1s_all, s2s_all, s3s_all,
                          Pf, rho_init_f,
                          a1f, b1f, g1f, a2f, a3f, q2f, q3f,
                          s1f_all, s2f_all, s3f_all,
                          rho_corr):
    """
    JIT-compiled joint simulation loop for spot + FCR.

    All sigma arrays are pre-computed (shape: n_steps,) so decay is free inside
    the loop. Explicit inner loops over n_paths run at C speed under njit.
    cache=True writes compiled bytecode to disk so re-runs skip recompilation.
    """
    perp = np.sqrt(max(1.0 - rho_corr ** 2, 0.0))

    # float32 paths/latents and int8 regimes (values 0/1/2) to keep the
    # per-batch footprint down; mean-reverting dynamics make float32 safe.
    spot_paths   = np.zeros((n_steps + 1, n_paths), dtype=np.float32)
    spot_lat     = np.zeros((n_steps + 1, n_paths), dtype=np.float32)
    spot_regimes = np.zeros((n_steps + 1, n_paths), dtype=np.int8)
    fcr_paths    = np.zeros((n_steps + 1, n_paths), dtype=np.float32)
    fcr_lat      = np.zeros((n_steps + 1, n_paths), dtype=np.float32)
    fcr_regimes  = np.zeros((n_steps + 1, n_paths), dtype=np.int8)

    # Initial conditions
    for p in range(n_paths):
        spot_paths[0, p] = x0_spot
        spot_lat[0, p]   = x0_spot
        fcr_paths[0, p]  = x0_fcr
        fcr_lat[0, p]    = x0_fcr

    # Initial regimes from stationary distributions
    cum_s = np.cumsum(rho_init_s)
    cum_f = np.cumsum(rho_init_f)
    u_s   = np.random.uniform(0.0, 1.0, n_paths)
    u_f   = np.random.uniform(0.0, 1.0, n_paths)
    for p in range(n_paths):
        for r in range(3):
            if u_s[p] <= cum_s[r]:
                spot_regimes[0, p] = r
                break
        for r in range(3):
            if u_f[p] <= cum_f[r]:
                fcr_regimes[0, p] = r
                break

    for t in range(1, n_steps + 1):
        s1s_t = s1s_all[t - 1]
        s2s_t = s2s_all[t - 1]
        s3s_t = s3s_all[t - 1]
        s1f_t = s1f_all[t - 1]
        s2f_t = s2f_all[t - 1]
        s3f_t = s3f_all[t - 1]

        # Regime transitions
        u_mc_s = np.random.uniform(0.0, 1.0, n_paths)
        u_mc_f = np.random.uniform(0.0, 1.0, n_paths)
        for p in range(n_paths):
            cum = 0.0
            for r in range(3):
                cum += Ps[spot_regimes[t - 1, p], r]
                if u_mc_s[p] <= cum:
                    spot_regimes[t, p] = r
                    break
            cum = 0.0
            for r in range(3):
                cum += Pf[fcr_regimes[t - 1, p], r]
                if u_mc_f[p] <= cum:
                    fcr_regimes[t, p] = r
                    break

        # Shared spot innovation (common factor for correlation, Equation 8)
        eps_spot = np.random.standard_normal(n_paths)
        eps_indep = np.random.standard_normal(n_paths)
        eps_fcr   = rho_corr * eps_spot + perp * eps_indep

        # Spot: base latent process (Equation 4)
        for p in range(n_paths):
            bl = spot_lat[t - 1, p]
            spot_lat[t, p] = a1s + (1.0 - b1s) * bl + s1s_t * (abs(bl) ** g1s) * eps_spot[p]

        # Spot: spike/drop draws (Equations 5, 6)
        eps2s = np.random.standard_normal(n_paths)
        eps3s = np.random.standard_normal(n_paths)
        for p in range(n_paths):
            spike = q2s + np.exp(a2s + s2s_t * eps2s[p])
            drop  = q3s - np.exp(a3s + s3s_t * eps3s[p])
            r = spot_regimes[t, p]
            if r == 0:
                spot_paths[t, p] = spot_lat[t, p]
            elif r == 1:
                spot_paths[t, p] = spike
            else:
                spot_paths[t, p] = drop

        # FCR: base latent process (Equation 4, correlated innovation)
        for p in range(n_paths):
            bl = fcr_lat[t - 1, p]
            fcr_lat[t, p] = a1f + (1.0 - b1f) * bl + s1f_t * (abs(bl) ** g1f) * eps_fcr[p]

        # FCR: spike/drop draws (Equations 5, 6)
        eps2f = np.random.standard_normal(n_paths)
        eps3f = np.random.standard_normal(n_paths)
        for p in range(n_paths):
            spike = q2f + np.exp(a2f + s2f_t * eps2f[p])
            drop  = q3f - np.exp(a3f + s3f_t * eps3f[p])
            r = fcr_regimes[t, p]
            if r == 0:
                fcr_paths[t, p] = fcr_lat[t, p]
            elif r == 1:
                fcr_paths[t, p] = spike
            else:
                fcr_paths[t, p] = drop

    return spot_paths, spot_regimes, fcr_paths, fcr_regimes


def simulate_joint(theta_spot, theta_fcr,
                   x0_spot, x0_fcr,
                   L0_spot, L0_fcr,
                   n_steps, n_paths,
                   rho_corr=None):
    """
    Simulate spot price and FCR jointly using a two-factor model.

    Each series: X_t = L_t (GOU) + Z_t (MRS). The JIT core simulates the
    correlated Z paths; GOU paths are then added on top.

    Spot decay uses SPIKE_DECAY_RATE / SPIKE_FLOOR_FRAC (imported from spot model).
    FCR  decay uses SIGMA_DECAY_RATE  / SIGMA_FLOOR_FRAC (defined in this file).

    Parameters
    ----------
    theta_spot : dict  — calibrated spot parameters
    theta_fcr  : dict  — calibrated FCR parameters
    x0_spot    : float — last observed Z_t for spot (deseasonalised residual)
    x0_fcr     : float — last observed Z_t for FCR
    L0_spot    : float — last observed L_t for spot
    L0_fcr     : float — last observed L_t for FCR
    n_steps    : int   — simulation horizon in hours
    n_paths    : int   — number of Monte Carlo paths
    rho_corr   : float or None — correlation (defaults to CORRELATIONS['FCR'])

    Returns
    -------
    spot_paths   : ndarray (n_steps+1, n_paths) — L + Z paths for spot
    spot_regimes : ndarray (n_steps+1, n_paths)
    fcr_paths    : ndarray (n_steps+1, n_paths) — L + Z paths for FCR
    fcr_regimes  : ndarray (n_steps+1, n_paths)
    """
    if rho_corr is None:
        rho_corr = CORRELATIONS['FCR']

    t_years = np.arange(1, n_steps + 1) / 8760

    # Spot: base regime constant, spike/drop decay to floor
    s1s_all = np.full(n_steps, theta_spot['sigma1'])
    s2s_inf = SPIKE_FLOOR_FRAC * theta_spot['sigma2']
    s3s_inf = SPIKE_FLOOR_FRAC * theta_spot['sigma3']
    s2s_all = s2s_inf + (theta_spot['sigma2'] - s2s_inf) * np.exp(-SPIKE_DECAY_RATE * t_years)
    s3s_all = s3s_inf + (theta_spot['sigma3'] - s3s_inf) * np.exp(-SPIKE_DECAY_RATE * t_years)

    # FCR: same structure
    s1f_all = np.full(n_steps, theta_fcr['sigma1'])
    s2f_inf = SIGMA_FLOOR_FRAC * theta_fcr['sigma2']
    s3f_inf = SIGMA_FLOOR_FRAC * theta_fcr['sigma3']
    s2f_all = s2f_inf + (theta_fcr['sigma2'] - s2f_inf) * np.exp(-SIGMA_DECAY_RATE * t_years)
    s3f_all = s3f_inf + (theta_fcr['sigma3'] - s3f_inf) * np.exp(-SIGMA_DECAY_RATE * t_years)

    spot_Z_paths, spot_regimes, fcr_Z_paths, fcr_regimes = _simulate_joint_core(
        n_steps, n_paths,
        float(x0_spot), float(x0_fcr),
        np.array(theta_spot['P'], dtype=np.float64),
        np.array(theta_spot['rho'], dtype=np.float64),
        theta_spot['alpha1'], theta_spot['beta1'], theta_spot['gamma1'],
        theta_spot['alpha2'], theta_spot['alpha3'],
        theta_spot['q2'],     theta_spot['q3'],
        s1s_all, s2s_all, s3s_all,
        np.array(theta_fcr['P'], dtype=np.float64),
        np.array(theta_fcr['rho'], dtype=np.float64),
        theta_fcr['alpha1'], theta_fcr['beta1'], theta_fcr['gamma1'],
        theta_fcr['alpha2'], theta_fcr['alpha3'],
        theta_fcr['q2'],     theta_fcr['q3'],
        s1f_all, s2f_all, s3f_all,
        float(rho_corr),
    )

    # Add GOU long-term factor to each series. simulate_geo_ou returns float64;
    # cast to float32 up front so the L + Z addition stays float32 end to end
    # (no full-size float64 temporary) and the returned arrays remain float32.
    spot_L_paths = simulate_geo_ou(L0_spot, GOU_THETA,     GOU_KAPPA,     GOU_SIGMA,     n_steps, n_paths).astype(np.float32)
    fcr_L_paths  = simulate_geo_ou(L0_fcr,  FCR_GOU_THETA, FCR_GOU_KAPPA, FCR_GOU_SIGMA, n_steps, n_paths).astype(np.float32)

    spot_paths = spot_L_paths + spot_Z_paths
    fcr_paths  = fcr_L_paths + fcr_Z_paths
    return spot_paths, spot_regimes, fcr_paths, fcr_regimes


def summarise_paths(paths):
    """Compute mean, 10th and 90th percentile across Monte Carlo paths."""
    return {
        'mean'  : paths.mean(axis=1),
        'lower' : np.percentile(paths, 10, axis=1),
        'upper' : np.percentile(paths, 90, axis=1),
        'paths' : paths,
    }


def save_joint_paths_to_csv(spot_paths, fcr_paths,
                             spot_df, fcr_df,
                             spot_ltsc_coeffs, fcr_ltsc_coeffs,
                             out_dir, start_dt=None, freq='h',
                             inflation_rate=0.0,
                             spot_price_bounds=None, fcr_price_bounds=None):
    """
    Re-seasonalise and save spot + FCR paths from simulate_joint() to CSV.

    Writes spot.parquet and FCR.parquet to out_dir.

    Parameters
    ----------
    spot_paths        : (n_steps+1, n_paths) — deseasonalised spot paths
    fcr_paths         : (n_steps+1, n_paths) — deseasonalised FCR paths
    spot_df           : historical spot DataFrame from deseasonalise()
    fcr_df            : historical FCR DataFrame from deseasonalise()
    spot_ltsc_coeffs  : ndarray — spot LTSC coefficients
    fcr_ltsc_coeffs   : ndarray — FCR LTSC coefficients
    out_dir           : str — directory to write CSV files (created if missing)
    start_dt          : first forecast timestamp (default: inferred from spot_df)
    freq              : pandas frequency string (default 'h')
    spot_price_bounds : (min, max) or None — clip spot paths to historical range
    fcr_price_bounds  : (min, max) or None — clip FCR paths to historical range
    """
    os.makedirs(out_dir, exist_ok=True)

    spot_floor = spot_price_bounds[0] if spot_price_bounds is not None else None
    spot_cap   = spot_price_bounds[1] if spot_price_bounds is not None else None
    fcr_floor  = fcr_price_bounds[0]  if fcr_price_bounds  is not None else None
    fcr_cap    = fcr_price_bounds[1]  if fcr_price_bounds  is not None else None

    save_paths_to_csv(
        spot_paths, spot_df, spot_ltsc_coeffs,
        out_path=os.path.join(out_dir, 'spot.parquet'),
        start_dt=start_dt, freq=freq,
        inflation_rate=inflation_rate,
        price_floor=spot_floor, price_cap=spot_cap,
    )
    save_paths_to_csv(
        fcr_paths, fcr_df, fcr_ltsc_coeffs,
        out_path=os.path.join(out_dir, 'FCR.parquet'),
        start_dt=start_dt, freq=freq,
        inflation_rate=inflation_rate,
        price_floor=fcr_floor, price_cap=fcr_cap,
    )

    print(f"Saved spot.parquet and FCR.parquet to {out_dir}/")


def save_all_params_to_csv(thetas_dict, out_path):
    """
    Save calibrated parameters for all balancing series to a single CSV.

    Each series becomes one row; columns are the same as save_params_to_csv
    (series, alpha1, beta1, sigma1, gamma1, long_run_mean, alpha2, sigma2,
    q2, alpha3, sigma3, q3, P_00…P_22, rho_0/1/2).

    Parameters
    ----------
    thetas_dict : dict {series_name: theta} — calibrated parameters per series
    out_path    : str — output CSV path (parent directory created if needed)
    """
    rows = []
    for name, theta in thetas_dict.items():
        P   = np.array(theta['P'])
        # Compute true stationary distribution π from P (solve πP = π, sum=1).
        # theta['rho'] is smooth[0] from EM — the t=0 regime prob, not stationary.
        A       = np.vstack([P.T - np.eye(3), np.ones((1, 3))])
        b       = np.zeros(4); b[-1] = 1.0
        rho_raw, *_ = np.linalg.lstsq(A, b, rcond=None)
        rho     = np.abs(rho_raw) / np.abs(rho_raw).sum()
        row = {'series': name,
               'alpha1':        float(theta['alpha1']),
               'beta1':         float(theta['beta1']),
               'sigma1':        float(theta['sigma1']),
               'gamma1':        float(theta['gamma1']),
               'long_run_mean': float(theta['alpha1'] / theta['beta1']),
               'alpha2': float(theta['alpha2']),
               'sigma2': float(theta['sigma2']),
               'q2':     float(theta['q2']),
               'alpha3': float(theta['alpha3']),
               'sigma3': float(theta['sigma3']),
               'q3':     float(theta['q3'])}
        for i in range(3):
            for j in range(3):
                row[f'P_{i}{j}'] = float(P[i, j])
        for i in range(3):
            row[f'rho_{i}'] = float(rho[i])
        rows.append(row)

    combined = pd.DataFrame(rows)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"Saved parameters for {len(thetas_dict)} series -> {out_path}")
    return combined


# =============================================================================
# CORRELATION ESTIMATION - used in case we shall run joiunt simulation together with simulation of the spot price. 
# =============================================================================

def estimate_correlation(X_spot, X_bal, label=""):
    """
    Estimate ρ between spot and balancing base regime increments.
    Bakke & Fleten (2016, p.490): computed on first differences.
    """
    rho = float(np.corrcoef(np.diff(X_spot), np.diff(X_bal))[0, 1])
    print(f"  Correlation spot vs {label}: rho = {rho:.4f}")
    return rho


# =============================================================================
# PLOTTING
# =============================================================================

def plot_all_forecasts(X_dict, results_dict):
    """
    Plot forecast fan charts for all balancing series in a grid.

    Parameters
    ----------
    X_dict       : dict {series_name: ndarray} — deseasonalised historical series
    results_dict : dict {series_name: summarise_paths output}
    """
    names  = list(X_dict.keys())
    n      = len(names)
    ncols  = 2
    nrows  = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes      = axes.flatten()

    for i, name in enumerate(names):
        ax       = axes[i]
        hist     = X_dict[name][-N_HISTORY:]
        results  = results_dict[name]
        idx_hist = np.arange(len(hist))
        idx_fore = np.arange(len(hist) - 1,
                              len(hist) - 1 + len(results['mean']))

        ax.plot(idx_hist, hist, color='black', linewidth=1.2, label='Historical')
        ax.fill_between(idx_fore, results['lower'], results['upper'],
                        alpha=0.25, color='darkorange', label='80% interval')
        ax.plot(idx_fore, results['mean'], color='darkorange',
                linewidth=2, label='Mean forecast')
        ax.axvline(x=len(hist) - 1, color='red', linestyle='--',
                   alpha=0.5, label='Forecast start')
        ax.set_title(name.replace("_", " "))
        ax.set_xlabel('Time step (hours)')
        ax.set_ylabel('Deseasonalised price')
        ax.legend(fontsize=8)

    # Hide any unused subplot panels
    for j in range(len(names), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('FCR Price Forecast — MRS Model', fontsize=13)
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/balancing_forecasts.png', dpi=150)
    plt.show()
    print(f"Saved {RESULTS_DIR}/balancing_forecasts.png")


def plot_convergence(logliks_dict):
    """Plot EM convergence for all series."""
    fig, axes = plt.subplots(1, len(logliks_dict),
                              figsize=(4 * len(logliks_dict), 3))
    if len(logliks_dict) == 1:
        axes = [axes]
    for ax, (name, logliks) in zip(axes, logliks_dict.items()):
        ax.plot(logliks, color='darkorange')
        ax.set_title(f'{name} — EM convergence')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log-likelihood')
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(f'{RESULTS_DIR}/balancing_convergence.png', dpi=150)
    plt.show()


# =============================================================================
# PARAMETER I/O — load pre-calibrated thetas from CSV
# =============================================================================

def load_params_from_csv(path):
    """
    Load calibrated parameters saved by calibrate_joint.py (or save_all_params_to_csv).

    Returns
    -------
    dict {series_name: theta}  — same structure as Calibrator.fit() output
    """
    df = pd.read_csv(path)
    thetas = {}
    for _, row in df.iterrows():
        P = [[row[f'P_{i}{j}'] for j in range(3)] for i in range(3)]
        rho = [row[f'rho_{i}'] for i in range(3)]
        thetas[row['series']] = {
            'alpha1': row['alpha1'], 'beta1':  row['beta1'],
            'sigma1': row['sigma1'], 'gamma1': row['gamma1'],
            'alpha2': row['alpha2'], 'sigma2': row['sigma2'],
            'alpha3': row['alpha3'], 'sigma3': row['sigma3'],
            'q2': row['q2'],         'q3':     row['q3'],
            'P':  P,                 'rho':    rho,
        }
    return thetas


# =============================================================================
# INITIAL PARAMETERS
#
# Separate starting guesses per product since their price dynamics differ.
# Adjust alpha1 to match the typical quiet-period level in each series.
# alpha2/alpha3 are LOG-SCALE: exp(alpha) ≈ typical excess above Q75 / below Q25.
# =============================================================================

INITIAL_PARAMS_FCR = {
    'alpha1': 10.0,  'beta1': 0.3,  'sigma1': 5.0,  'gamma1': 0.0,
    'alpha2': 2.5,   'sigma2': 0.8,
    'alpha3': 2.5,   'sigma3': 0.8,
    'P':   [[0.96, 0.02, 0.02], [0.20, 0.75, 0.05], [0.20, 0.05, 0.75]],
    'rho': [0.92, 0.04, 0.04],
}

