"""
Load the simulation parquet files (one per market, 2 000 paths × ~70 years of
hourly prices in thesis units) and compute per-path, per-year metrics.

For each path and year we compute:
    yearly_mean
    daily_spread        : mean over the year of (daily max - daily min)
    weekly_spread       : mean over the year of (weekly max - weekly min)
    intra_year_amp      : (max monthly mean - min monthly mean) within the year
    hourly_volatility   : std of hourly log-returns (decimals -> %)

Output: analysis/data/sim_metrics_<market>.npz  (n_paths × n_years arrays)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

from stochastic_modelling.analysis._constants import (
    DATA_OUT, N_PATHS, N_SIM_YEARS, SPOT_START_YEAR, FCR_START_YEAR,
    load_sim_parquet,
)


METRIC_KEYS = (
    "yearly_mean", "daily_spread", "weekly_spread", "intra_year_amp", "hourly_vol",
)


def _empty_metrics(n_paths: int, n_years: int) -> dict:
    return {k: np.full((n_paths, n_years), np.nan, dtype=np.float64) for k in METRIC_KEYS}


def compute_year_metrics(prices: pd.DataFrame, last_log_prev):
    """Compute the five metrics for a single year of hourly prices.

    prices: DataFrame with hourly DatetimeIndex and one column per path.
    last_log_prev: 1-D np.ndarray of log-prices at the last hour of the previous
        year, or None.

    Returns (metrics_dict, last_log_this_year).
    """
    vals = prices.to_numpy(dtype=np.float64, copy=False)
    idx = prices.index

    yearly_mean = np.nanmean(vals, axis=0)

    # Daily spread: max - min within each calendar day, then mean across days.
    day_key = idx.normalize()
    grp_day = prices.groupby(day_key)
    daily_spread = (grp_day.max() - grp_day.min()).mean(axis=0).to_numpy()

    # Weekly spread: 7-day buckets indexed by day-of-year // 7
    # (matches the historical computation).
    week_key = ((idx.dayofyear - 1) // 7).astype(np.int32)
    wdf = prices.copy()
    wdf["__w"] = week_key
    wmax = wdf.groupby("__w").max().drop(columns="__w", errors="ignore")
    wmin = wdf.groupby("__w").min().drop(columns="__w", errors="ignore")
    weekly_spread = (wmax - wmin).mean(axis=0).to_numpy()

    # Intra-year amplitude: range of monthly means within the year.
    mdf = prices.copy()
    mdf["__m"] = idx.month
    monthly_mean = mdf.groupby("__m").mean().drop(columns="__m", errors="ignore")
    intra_year_amp = (monthly_mean.max(axis=0) - monthly_mean.min(axis=0)).to_numpy()

    # Hourly log-return volatility (%).
    with np.errstate(invalid="ignore", divide="ignore"):
        logp = np.where(vals > 0, np.log(np.maximum(vals, 1e-12)), np.nan)
    if last_log_prev is not None:
        full = np.vstack([last_log_prev[None, :], logp])
        dlog = np.diff(full, axis=0)
    else:
        dlog = np.diff(logp, axis=0)
    hourly_vol = np.nanstd(dlog, axis=0) * 100.0

    last_log = logp[-1]
    return (
        {
            "yearly_mean":    yearly_mean,
            "daily_spread":   daily_spread,
            "weekly_spread":  weekly_spread,
            "intra_year_amp": intra_year_amp,
            "hourly_vol":     hourly_vol,
        },
        last_log,
    )


def process_market(market: str, start_year: int) -> None:
    print(f"[{market}] loading parquet ...", flush=True)
    t0 = time.time()
    df = load_sim_parquet(market)
    print(f"  shape {df.shape}, loaded in {time.time() - t0:.1f}s", flush=True)

    if df.shape[1] != N_PATHS:
        print(f"  warning: expected {N_PATHS} paths, got {df.shape[1]}")
    n_paths = df.shape[1]

    metrics = _empty_metrics(n_paths, N_SIM_YEARS)

    last_log = None
    for year, sub in df.groupby(df.index.year):
        year_idx = int(year) - start_year
        if year_idx < 0 or year_idx >= N_SIM_YEARS:
            last_log = None
            continue
        t1 = time.time()
        ym, last_log = compute_year_metrics(sub, last_log)
        for k, arr in ym.items():
            metrics[k][:, year_idx] = arr
        print(f"  [{market}] year {int(year)}: {sub.shape[0]:>5d} h, "
              f"{time.time() - t1:.1f}s", flush=True)

    out_path = os.path.join(DATA_OUT, f"sim_metrics_{market}.npz")
    np.savez_compressed(
        out_path,
        yearly_mean=metrics["yearly_mean"],
        daily_spread=metrics["daily_spread"],
        weekly_spread=metrics["weekly_spread"],
        intra_year_amp=metrics["intra_year_amp"],
        hourly_vol=metrics["hourly_vol"],
        start_year=start_year,
        n_paths=n_paths,
    )
    print(f"Saved {out_path}", flush=True)


def main():
    market = sys.argv[1] if len(sys.argv) > 1 else "all"
    if market in ("all", "spot"):
        process_market("spot", SPOT_START_YEAR)
    if market in ("all", "fcr"):
        process_market("fcr", FCR_START_YEAR)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total: {time.time() - t0:.0f}s")
