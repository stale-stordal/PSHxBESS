"""
Historical + simulated price paths — spot and FCR (weekly resolution).

Same as plot_price_paths.py but both historical and simulated are
resampled to weekly means.

Output:
    price_paths_weekly.pdf  (saved in the same folder as this script)
"""
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
RESAMPLE   = "W"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_historical(path, ts_col, price_col, start, resample):
    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.set_index(ts_col).sort_index()
    s  = df[price_col].astype(float)
    return s[s.index >= start].resample(resample).mean().dropna()

def load_sim(path, resample):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < X_MAX]
    cols = [c for c in df.columns if c.startswith("path_")]
    return df[cols].resample(resample).mean()

spot_hist = load_historical(SPOT_CSV, "Timestamp", "Price_DA",  HIST_START, RESAMPLE)
fcr_hist  = load_historical(FCR_CSV,  "Timestamp", "Price_FCR", HIST_START, RESAMPLE)
spot_sim  = load_sim(SPOT_SIM, RESAMPLE)
fcr_sim   = load_sim(FCR_SIM,  RESAMPLE)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7.5),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
)

LW_HIST = 1.2
LW_PATH = 0.9
ALPHA   = 0.45

def _add_panel(ax, hist_ser, sim_df, ylabel, unit, path_color):
    for col in sim_df.columns:
        ax.plot(sim_df.index, sim_df[col].values,
                color=path_color, lw=LW_PATH, alpha=ALPHA, zorder=2)
    # Single legend proxy for all paths
    ax.plot([], [], color=path_color, lw=LW_PATH, alpha=0.8,
            label="Simulated paths (n=10)")

    ax.plot(hist_ser.index, hist_ser.values,
            color=COLORS["neutral"], lw=LW_HIST, label="Historical", zorder=3)

    ax.axvline(sim_df.index[0], color=COLORS["neutral"], lw=0.9,
               linestyle="--", alpha=0.6, zorder=5)

    ax.set_ylabel(f"{ylabel}\n({unit})")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2)


_add_panel(ax1, spot_hist, spot_sim, "Day-ahead spot price", "NOK/MWh",  COLORS["main"])
_add_panel(ax2, fcr_hist,  fcr_sim,  "FCR-D price",          "NOK/MW/h", COLORS["secondary"])

ax1.set_xlim(spot_hist.index[0], X_MAX)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlabel("Year")

plt.tight_layout()

out = os.path.join(FIG_OUT, "price_paths_weekly.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
