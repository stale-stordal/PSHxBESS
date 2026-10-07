"""Historical revenue per year (2015-2025) for BESS and PSH.

Stacked day-ahead / FCR revenue per calendar year from the historical-price
dispatch runs. One figure per technology, since the absolute revenue scales
differ by ~40x; the two figures share style so they can sit side-by-side in
the thesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    COLORS, FIGSIZE, LAYOUT,
    apply_style, save_fig,
)

PATHS_DIR = Path(__file__).resolve().parents[1] / "results" / "paths"

RESULTS = {
    "BESS": PATHS_DIR / "Battery_optimized_results_historical.csv",
    "PSH":  PATHS_DIR / "Pump_optimized_results_historical.csv",
}


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["path_id"].astype(int)
    return df.sort_values("year")


def plot_one(tech: str) -> None:
    df = _load(RESULTS[tech])

    years = df["year"].to_numpy()
    rev_da = df["rev_DA"].to_numpy() / 1e6
    rev_fcr = df["rev_FCR"].to_numpy() / 1e6

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.bar(years, rev_da, color=COLORS["main"], label="Day-ahead", zorder=3)
    ax.bar(years, rev_fcr, bottom=rev_da, color=COLORS["secondary"],
           label="FCR", zorder=3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Annual revenue (MNOK)")
    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=2, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, f"13_historical_revenue_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 13_historical_revenue_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
