"""
Historical + single simulated price path — spot and FCR (hourly resolution).

Output:
    price_paths_hourly_single.pdf  (saved in the same folder as this script)
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_historical(path, ts_col, price_col, start):
    df = pd.read_csv(path, parse_dates=[ts_col])
    df = df.set_index(ts_col).sort_index()
    s  = df[price_col].astype(float)
    return s[s.index >= start].dropna()

def load_sim_single(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df[df.index < X_MAX]
    col = next(c for c in df.columns if c.startswith("path_"))
    return df[col]

spot_hist = load_historical(SPOT_CSV, "Timestamp", "Price_DA",  HIST_START)
fcr_hist  = load_historical(FCR_CSV,  "Timestamp", "Price_FCR", HIST_START)
spot_path = load_sim_single(SPOT_SIM)
fcr_path  = load_sim_single(FCR_SIM)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7.5),
    sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15},
)

LW_HIST = 1.0
LW_PATH = 0.7

def _add_panel(ax, hist_ser, sim_ser, ylabel, unit, path_color):
    ax.plot(sim_ser.index, sim_ser.values,
            color=path_color, lw=LW_PATH, alpha=0.75, label="Simulated path", zorder=2)
    ax.plot(hist_ser.index, hist_ser.values,
            color=COLORS["neutral"], lw=LW_HIST, label="Historical", zorder=3)
    ax.axvline(sim_ser.index[0], color=COLORS["neutral"], lw=0.9,
               linestyle="--", alpha=0.6, zorder=5)
    ax.set_ylabel(f"{ylabel}\n({unit})")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, ncol=2)


_add_panel(ax1, spot_hist, spot_path, "Day-ahead spot price", "NOK/MWh",  COLORS["main"])
_add_panel(ax2, fcr_hist,  fcr_path,  "FCR-N price",          "NOK/MW/h", COLORS["secondary"])

ax1.set_xlim(spot_hist.index[0], X_MAX)
ax1.xaxis.set_major_locator(mdates.YearLocator(5))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.set_xlabel("Year")

plt.tight_layout()

out = os.path.join(FIG_OUT, "price_paths_hourly_single.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
