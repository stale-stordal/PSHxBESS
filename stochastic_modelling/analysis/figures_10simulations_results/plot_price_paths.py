"""
Historical + simulated price paths — spot and FCR.

Historical:  2015 onwards from the raw price CSVs (weekly resampled).
Simulated:   10 paths at hourly resolution, capped at 2050.
             Each path gets a distinct color.

Output:
    price_paths.pdf  (saved in the same folder as this script)
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm

# Script lives at:  stochastic_modelling/analysis/figures_10simulations_results/
# thesis_style.py lives at the repo root (3 levels up).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)
from thesis_style import COLORS, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = os.path.join(REPO_ROOT, "data", "prices")
SIM_DIR   = SCRIPT_DIR
FIG_OUT   = SCRIPT_DIR

SPOT_CSV  = os.path.join(DATA_DIR, "DA",  "DA_NO2_NOK.csv")
FCR_CSV   = os.path.join(DATA_DIR, "FCR", "FCR_NO2_NOK.csv")
SPOT_SIM  = os.path.join(SIM_DIR,  "spot_1_first10.csv")
FCR_SIM   = os.path.join(SIM_DIR,  "FCR_1_first10.csv")

HIST_START = "2015-01-01"
X_MAX      = pd.Timestamp("2050-01-01")
RESAMPLE_HIST = None  # hourly — no resampling

# Per-path colors — 10 distinct muted tones
PATH_COLORS = [cm.tab10(i) for i in range(10)]

# ---------------------------------------------------------------------------
# Load historical data  (weekly resampled)
# ---------------------------------------------------------------------------
def load_historical(path, ts_col, price_col, start, resample):
    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.set_index(ts_col).sort_index()
    s  = df[price_col].astype(float)
    s  = s[s.index >= start]
    return s.dropna() if resample is None else s.resample(resample).mean().dropna()

spot_hist = load_historical(SPOT_CSV, "Timestamp", "Price_DA",  HIST_START, RESAMPLE_HIST)
fcr_hist  = load_historical(FCR_CSV,  "Timestamp", "Price_FCR", HIST_START, RESAMPLE_HIST)

# ---------------------------------------------------------------------------
# Load simulation paths  (hourly, trimmed to X_MAX)
# ---------------------------------------------------------------------------
def load_sim(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < X_MAX]
    return df[[c for c in df.columns if c.startswith("path_")]]

spot_sim = load_sim(SPOT_SIM)
fcr_sim  = load_sim(FCR_SIM)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 10),
    sharex=False,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.25},
)

LW_HIST = 1.2
LW_PATH = 0.7
ALPHA   = 0.55

def _add_panel(ax, hist_ser, sim_df, ylabel, unit):
    ax.plot(hist_ser.index, hist_ser.values,
            color=COLORS["neutral"], lw=LW_HIST, label="Historical", zorder=3)

    for i, col in enumerate(sim_df.columns):
        ax.plot(sim_df.index, sim_df[col].values,
                color=PATH_COLORS[i], lw=LW_PATH, alpha=ALPHA, zorder=2,
                label=f"Path {i+1}" if i == 0 else f"Path {i+1}")

    ax.axvline(sim_df.index[0], color=COLORS["neutral"], lw=0.9,
               linestyle="--", alpha=0.6, zorder=5)

    ax.set_ylabel(f"{ylabel}\n({unit})")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)

    # Compact legend: historical + first path entry to explain colors
    handles, labels = ax.get_legend_handles_labels()
    # Show historical + all 10 path entries
    ax.legend(handles, labels, loc="upper left", frameon=False,
              fontsize=plt.rcParams["legend.fontsize"] - 4,
              ncol=4)


_add_panel(ax1, spot_hist, spot_sim, "Day-ahead spot price", "NOK/MWh")
_add_panel(ax2, fcr_hist,  fcr_sim,  "FCR-D price",          "NOK/MW/h")

for ax, hist_ser, show_xlabel in [
    (ax1, spot_hist, False),
    (ax2, fcr_hist,  True),
]:
    ax.set_xlim(hist_ser.index[0], X_MAX)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    if show_xlabel:
        ax.set_xlabel("Year")

plt.tight_layout()

out = os.path.join(FIG_OUT, "price_paths.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
