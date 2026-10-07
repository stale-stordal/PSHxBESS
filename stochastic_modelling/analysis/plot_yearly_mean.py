"""
Fan charts of yearly mean spot and FCR prices, with historical context.

Reads:
    data/sim_metrics_<market>.npz    (n_paths × n_years arrays in thesis units)
    data/historical_metrics.csv      (NO2 historical 2000-2025 spot, 2015-2025 FCR)

Outputs:
    figures/fan_yearly_mean_spot.pdf
    figures/fan_yearly_mean_fcr.pdf
    data/decadal_summary.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stochastic_modelling.analysis._constants import (
    DATA_OUT, FIG_OUT, UNIT_SPOT, UNIT_FCR,
)
from thesis_style import COLORS, FIGSIZE, apply_style

apply_style("half")
os.makedirs(FIG_OUT, exist_ok=True)

YEAR_CAP = 2075


def load_sim_yearly_mean(market: str):
    """Return (years, yearly_mean_array) for the given market, capped at YEAR_CAP."""
    npz_path = os.path.join(DATA_OUT, f"sim_metrics_{market}.npz")
    with np.load(npz_path) as d:
        ym = d["yearly_mean"]
        start_year = int(d["start_year"])
    years = np.arange(start_year, start_year + ym.shape[1])
    mask  = years <= YEAR_CAP
    return years[mask], ym[:, mask]


def percentiles_per_year(arr: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "mean": np.nanmean(arr, axis=0),
        "p10":  np.nanpercentile(arr, 10, axis=0),
        "p25":  np.nanpercentile(arr, 25, axis=0),
        "p50":  np.nanpercentile(arr, 50, axis=0),
        "p75":  np.nanpercentile(arr, 75, axis=0),
        "p90":  np.nanpercentile(arr, 90, axis=0),
    })


def fan_chart(ax, years, sim_df, hist_df, ylabel, sim_color, hist_color):
    ax.fill_between(years, sim_df["p10"], sim_df["p90"],
                    color=sim_color, alpha=0.18, linewidth=0,
                    label="Simulated 10th–90th percentile")
    ax.fill_between(years, sim_df["p25"], sim_df["p75"],
                    color=sim_color, alpha=0.30, linewidth=0,
                    label="Simulated 25th–75th percentile")
    ax.plot(years, sim_df["p50"], color=sim_color, linewidth=2.2,
            label="Simulated median")

    if hist_df is not None and len(hist_df) > 0:
        ax.plot(hist_df["year"].values, hist_df["yearly_mean"].values,
                color=hist_color, linewidth=2.0, label="Historical (NO2)")

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1)


def main():
    df_hist = pd.read_csv(os.path.join(DATA_OUT, "historical_metrics.csv"))
    df_hist = df_hist[df_hist["n_hours"] >= 8000].copy()

    decadal_rows = []
    for market, color, ylabel in [
        ("spot", COLORS["main"],      f"Yearly mean spot price ({UNIT_SPOT})"),
        ("fcr",  COLORS["secondary"], f"Yearly mean FCR price ({UNIT_FCR})"),
    ]:
        years, ym = load_sim_yearly_mean(market)
        sim_df = percentiles_per_year(ym)

        hist_market = df_hist[df_hist["market"] == market][["year", "yearly_mean"]]
        fig, ax = plt.subplots(figsize=FIGSIZE["half"])
        fan_chart(ax, years, sim_df, hist_market,
                  ylabel=ylabel, sim_color=color, hist_color=COLORS["highlight"])
        plt.tight_layout()
        out = os.path.join(FIG_OUT, f"fan_yearly_mean_{market}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

        # Decadal summary across all paths and years in each decade.
        decade = (years // 10) * 10
        for dec in np.unique(decade):
            mask = decade == dec
            sub = ym[:, mask]
            d_end = int(min(dec + 9, years.max()))
            decadal_rows.append({
                "market": "Spot" if market == "spot" else "FCR",
                "decade": f"{int(dec)}–{d_end}",
                "mean":   float(np.nanmean(sub)),
                "p10":    float(np.nanpercentile(sub, 10)),
                "p50":    float(np.nanpercentile(sub, 50)),
                "p90":    float(np.nanpercentile(sub, 90)),
            })

    summary = pd.DataFrame(decadal_rows)
    out = os.path.join(DATA_OUT, "decadal_summary.csv")
    summary.to_csv(out, index=False)
    print(f"Saved {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
