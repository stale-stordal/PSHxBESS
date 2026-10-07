"""
Fan charts for intra-day, intra-week and intra-year spreads — the BESS- and
PSH-relevant volatility metrics. Also a fan chart of hourly log-volatility.

Reads:
    data/sim_metrics_<market>.npz   (n_paths × n_years arrays)
    data/historical_metrics.csv

Outputs:
    figures/fan_intra_day_spread_<market>.pdf
    figures/fan_intra_week_spread_<market>.pdf
    figures/fan_intra_year_amp_<market>.pdf
    figures/fan_hourly_vol_<market>.pdf
    data/spread_decadal_summary.csv
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

METRIC_LABELS = {
    "daily_spread":    ("intra_day_spread",  "Mean daily price range"),
    "weekly_spread":   ("intra_week_spread", "Mean weekly price range"),
    "intra_year_amp":  ("intra_year_amp",    "Intra-year amplitude (max − min monthly mean)"),
    "hourly_vol":      ("hourly_volatility_pct", "Hourly log-return volatility (%)"),
}


def fan(ax, years, sim, hist_x=None, hist_y=None, color=None, hist_color=None, label_unit=""):
    p10 = np.nanpercentile(sim, 10, axis=0)
    p25 = np.nanpercentile(sim, 25, axis=0)
    p50 = np.nanpercentile(sim, 50, axis=0)
    p75 = np.nanpercentile(sim, 75, axis=0)
    p90 = np.nanpercentile(sim, 90, axis=0)

    ax.fill_between(years, p10, p90, color=color, alpha=0.18, linewidth=0,
                    label="Simulated 10th–90th percentile")
    ax.fill_between(years, p25, p75, color=color, alpha=0.30, linewidth=0,
                    label="Simulated 25th–75th percentile")
    ax.plot(years, p50, color=color, linewidth=2.2, label="Simulated median")

    if hist_x is not None and hist_y is not None and len(hist_y) > 0:
        ax.plot(hist_x, hist_y, color=hist_color, linewidth=2.0, label="Historical (NO2)")

    ax.set_xlabel("Year")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", ncol=1)


def plot_market(market, hist_df, color, hist_color):
    npz_path = os.path.join(DATA_OUT, f"sim_metrics_{market}.npz")
    if not os.path.exists(npz_path):
        print(f"[skip] {npz_path} not yet available")
        return None
    with np.load(npz_path) as d:
        start_year = int(d["start_year"])
        sim = {
            "yearly_mean":    d["yearly_mean"],
            "daily_spread":   d["daily_spread"],
            "weekly_spread":  d["weekly_spread"],
            "intra_year_amp": d["intra_year_amp"],
            "hourly_vol":     d["hourly_vol"],
        }

    n_years = sim["yearly_mean"].shape[1]
    years = np.arange(start_year, start_year + n_years)
    mask  = years <= YEAR_CAP
    years = years[mask]
    sim   = {k: v[:, mask] for k, v in sim.items()}
    unit_label = UNIT_SPOT if market == "spot" else UNIT_FCR

    hist = hist_df[hist_df["market"] == market].copy()
    hist = hist[hist["n_hours"] >= 8000]

    plot_specs = [
        ("daily_spread",   f"Mean daily price range ({unit_label})",     "fan_intra_day_spread"),
        ("weekly_spread",  f"Mean weekly price range ({unit_label})",    "fan_intra_week_spread"),
        ("intra_year_amp", f"Intra-year amplitude ({unit_label})",       "fan_intra_year_amp"),
        ("hourly_vol",     "Hourly log-return volatility (%)",           "fan_hourly_vol"),
    ]
    rows = []
    for sim_key, ylabel, slug in plot_specs:
        hist_col = METRIC_LABELS[sim_key][0]

        fig, ax = plt.subplots(figsize=FIGSIZE["half"])
        fan(ax, years, sim[sim_key],
            hist_x=hist["year"].values, hist_y=hist[hist_col].values,
            color=color, hist_color=hist_color)
        ax.set_ylabel(ylabel)
        plt.tight_layout()
        out = os.path.join(FIG_OUT, f"{slug}_{market}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

        # decadal stats
        for d_start in range(start_year, start_year + n_years, 10):
            d_end = min(d_start + 9, start_year + n_years - 1)
            mask = (years >= d_start) & (years <= d_end)
            sub = sim[sim_key][:, mask]
            rows.append({
                "market": market, "metric": sim_key,
                "decade": f"{d_start}–{d_end}",
                "mean":  np.nanmean(sub),
                "p10":   np.nanpercentile(sub, 10),
                "p50":   np.nanpercentile(sub, 50),
                "p90":   np.nanpercentile(sub, 90),
            })
    return rows


def main():
    hist = pd.read_csv(os.path.join(DATA_OUT, "historical_metrics.csv"))
    hist["market"] = hist["market"].str.lower()

    rows = []
    for market, color in [("spot", COLORS["main"]), ("fcr", COLORS["secondary"])]:
        r = plot_market(market, hist, color=color, hist_color=COLORS["highlight"])
        if r:
            rows.extend(r)

    if rows:
        out = os.path.join(DATA_OUT, "spread_decadal_summary.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
