"""Historical revenue (2015-2025) continued by the simulated model horizon.

For each technology the figure shows the actual historical dispatch revenue per
calendar year as stacked day-ahead / FCR bars, then continues along the x-axis
with the simulated model: mean total revenue and P5-P95 / P25-P75 envelopes.

Everything is expressed in real 2025 NOK (2 %/yr):
  * historical year Y : nominal * 1.02**(2025 - Y)
  * model op-year k    : calendar 2026 + k, nominal / 1.02**(k + 1)

One figure per technology, since PSH and BESS revenue scales differ by ~16x.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    COLORS, FIGSIZE, LAYOUT,
    apply_style, save_fig,
)

PATHS_DIR = Path(__file__).resolve().parents[1] / "results" / "paths"
MODEL_DIR = Path(__file__).resolve().parents[2] / "roa" / "input_files"

HISTORICAL = {
    "PSH":  PATHS_DIR / "Pump_optimized_results_historical.csv",
    "BESS": PATHS_DIR / "Battery_optimized_results_historical.csv",
}
MODEL = {
    "PSH":  MODEL_DIR / "PSH_optimized_results_combined.csv",
    "BESS": MODEL_DIR / "BESS_optimized_results_combined.csv",
}

INFLATION_RATE = 0.02
BASE_YEAR      = 2025
MODEL_START    = 2026   # calendar year of model operating year 0
N_MODEL_YEARS  = 25     # simulated years to display (2026 .. 2050)

DA_COLOR    = COLORS["main"]
FCR_COLOR   = COLORS["secondary"]
MODEL_COLOR = COLORS["highlight"]


def _historical_real(path: Path) -> pd.DataFrame:
    """Historical per-year DA/FCR revenue in real BASE_YEAR MNOK."""
    df = pd.read_csv(path)
    df["year"] = df["path_id"].astype(int)
    infl = (1.0 + INFLATION_RATE) ** (BASE_YEAR - df["year"])
    df["rev_DA_r"]  = df["rev_DA"]  * infl / 1e6
    df["rev_FCR_r"] = df["rev_FCR"] * infl / 1e6
    return df.sort_values("year")


def _model_real_means(path: Path) -> pd.DataFrame:
    """Per-calendar-year mean DA / FCR revenue in real BASE_YEAR MNOK."""
    df = pd.read_csv(path)
    df = df[df["year"] < N_MODEL_YEARS].copy()
    df["calendar"] = MODEL_START + df["year"].astype(int)
    deflator = (1.0 + INFLATION_RATE) ** (df["calendar"] - BASE_YEAR)
    df["rev_DA_r"]  = df["rev_DA"]  / deflator / 1e6
    df["rev_FCR_r"] = df["rev_FCR"] / deflator / 1e6
    g = df.groupby("calendar")[["rev_DA_r", "rev_FCR_r"]].mean()
    return g


def plot_one(tech: str) -> None:
    hist  = _historical_real(HISTORICAL[tech])
    model = _model_real_means(MODEL[tech])

    fig, ax = plt.subplots(figsize=FIGSIZE["wide"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # Historical stacked DA / FCR bars (actual dispatch).
    hy = hist["year"].to_numpy()
    ax.bar(hy, hist["rev_DA_r"],  color=DA_COLOR,  label="Day-ahead", zorder=3)
    ax.bar(hy, hist["rev_FCR_r"], bottom=hist["rev_DA_r"], color=FCR_COLOR,
           label="FCR", zorder=3)

    # Simulated stacked DA / FCR bars (mean market participation across paths).
    my = model.index.to_numpy()
    ax.bar(my, model["rev_DA_r"],  color=DA_COLOR,  zorder=3)
    ax.bar(my, model["rev_FCR_r"], bottom=model["rev_DA_r"], color=FCR_COLOR,
           zorder=3)

    # Divider + period labels.
    ax.axvline(BASE_YEAR + 0.5, color=COLORS["neutral"], lw=0.8, ls="--", zorder=2)
    ymax = ax.get_ylim()[1]
    ax.text(hy.mean(), ymax * 0.96, "Historical",
            ha="center", va="top", color=COLORS["neutral"])
    ax.text(my.mean(), ymax * 0.96, "Simulated (mean)",
            ha="center", va="top", color=COLORS["neutral"])

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual revenue (real 2025 MNOK)")
    xticks = np.concatenate([hy, my[::5]])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(x)) for x in xticks], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.set_xlim(hy.min() - 0.7, my.max() + 0.7)

    ax.legend(loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
              ncol=2, frameon=False)

    plt.tight_layout()
    save_fig(fig, f"15_historical_and_model_revenue_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="wide")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 15_historical_and_model_revenue_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
