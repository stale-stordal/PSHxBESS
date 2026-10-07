"""
Compute yearly historical metrics for NO2 spot (2000-2025) and FCR (2015-2025).

For each year produces:
  yearly_mean         : mean hourly price
  intra_day_spread    : mean over the year of (daily max - daily min)
  intra_week_spread   : mean over the year of (weekly max - weekly min)
  intra_year_amp      : (max monthly mean - min monthly mean) within the year
  hourly_volatility   : std of hourly log-returns (decimals -> %)

Output: thesis_results/data/historical_metrics.csv
        thesis_results/data/historical_hourly_returns.csv  (for backtest QQ)
"""
import os
import glob
import numpy as np
import pandas as pd

from stochastic_modelling.analysis._constants import (
    DA_DIR, FCR_DIR, SPOT_AREA, DATA_OUT,
)

os.makedirs(DATA_OUT, exist_ok=True)


def yearly_metrics(df: pd.DataFrame, price_col: str) -> dict:
    p = df[price_col].astype(float).values
    if len(p) < 24:
        return None
    ts = df.index

    # Daily spread
    daily = df[price_col].resample("D").agg(["min", "max"]).dropna()
    daily_spread = (daily["max"] - daily["min"]).mean()

    # Weekly spread
    weekly = df[price_col].resample("W").agg(["min", "max"]).dropna()
    weekly_spread = (weekly["max"] - weekly["min"]).mean()

    # Intra-year amplitude (range of monthly means)
    monthly = df[price_col].resample("MS").mean().dropna()
    intra_year_amp = monthly.max() - monthly.min() if len(monthly) >= 2 else np.nan

    # Hourly log-returns (as percent)
    p_pos = p[p > 0]
    if len(p_pos) > 100:
        rets = np.diff(np.log(p_pos))
        hourly_vol = rets.std() * 100.0
    else:
        hourly_vol = np.nan

    return {
        "yearly_mean": float(np.mean(p)),
        "intra_day_spread": float(daily_spread),
        "intra_week_spread": float(weekly_spread),
        "intra_year_amp": float(intra_year_amp) if not np.isnan(intra_year_amp) else np.nan,
        "hourly_volatility_pct": float(hourly_vol),
        "n_hours": int(len(p)),
    }


def load_da_year(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["delivery_start"])
    df = df.set_index("delivery_start").sort_index()
    df["price"] = df[f"{SPOT_AREA} Price (NOK)"].astype(float)
    # Resample to hourly mean (some files mix hourly and 15-min ISPs)
    return df[["price"]].resample("h").mean().dropna()


def load_fcr_year(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ISP(UTC)"])
    df = df.set_index("ISP(UTC)").sort_index()
    df["price"] = df["price_nok_mw"].astype(float)
    return df[["price"]].resample("h").mean().dropna()


def collect(market: str):
    if market == "spot":
        files = sorted(glob.glob(os.path.join(DA_DIR, "DA_*_hourly.csv")))
        loader = load_da_year
    else:
        files = sorted(glob.glob(os.path.join(FCR_DIR, "FCR_*_cleaned.csv")))
        loader = load_fcr_year

    frames = []
    rows = []
    for f in files:
        df = loader(f)
        if df.empty:
            continue
        # group by year — file may not be exactly one year
        for yr, sub in df.groupby(df.index.year):
            metrics = yearly_metrics(sub, "price")
            if metrics is None:
                continue
            metrics["year"] = int(yr)
            metrics["market"] = market
            rows.append(metrics)
        frames.append(df)
    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    full = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    return out, full


def main():
    print("Processing historical spot (NO2)...")
    spot_yearly, spot_full = collect("spot")
    print(f"  {len(spot_yearly)} year-rows")

    print("Processing historical FCR (NO2)...")
    fcr_yearly, fcr_full = collect("fcr")
    print(f"  {len(fcr_yearly)} year-rows")

    out = pd.concat([spot_yearly, fcr_yearly], ignore_index=True)
    out_path = os.path.join(DATA_OUT, "historical_metrics.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # Hourly log-returns (subsampled to 200k for backtest QQ figure)
    rng = np.random.default_rng(0)
    for name, full in [("spot", spot_full), ("fcr", fcr_full)]:
        if full.empty:
            continue
        p = full["price"].values
        p = p[p > 0]
        rets = np.diff(np.log(p))
        if len(rets) > 200_000:
            rets = rng.choice(rets, 200_000, replace=False)
        np.save(os.path.join(DATA_OUT, f"historical_returns_{name}.npy"), rets)
        print(f"Saved historical_returns_{name}.npy ({len(rets)} samples)")


if __name__ == "__main__":
    main()
