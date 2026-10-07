"""Mean total-revenue trajectory over the 50-year horizon, with a P5–P95
band and an inner P25–P75 band, plotted separately for PSH and BESS.

Two figures are produced: one per technology, since the absolute revenue
scales differ by ~200x and putting them on the same axis would compress the
BESS trajectory into a flat line. The two figures share style and y-label
formatting so that they can sit side-by-side in the thesis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    FIGSIZE, LAYOUT, TECH_COLORS,
    apply_style, load_panel, per_year_stats, save_fig,
)


def _scale_and_label(tech: str) -> tuple[float, str]:
    """Choose a y-axis scale (NOK -> MNOK or BNOK) and matching label."""
    if tech == "PSH":
        return 1e9, "Annual revenue (BNOK)"
    return 1e6, "Annual revenue (MNOK)"


def plot_one(tech: str) -> None:
    df = load_panel(tech)
    stats = per_year_stats(df, "rev_total")

    scale, ylabel = _scale_and_label(tech)
    color = TECH_COLORS[tech]

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    years = stats.index.to_numpy()

    ax.fill_between(
        years, stats["p05"] / scale, stats["p95"] / scale,
        color=color, alpha=0.15, linewidth=0, label="P5 – P95",
    )
    ax.fill_between(
        years, stats["p25"] / scale, stats["p75"] / scale,
        color=color, alpha=0.30, linewidth=0, label="P25 – P75",
    )
    ax.plot(
        years, stats["mean"] / scale,
        color=color, lw=LAYOUT["line_width"], label="Mean",
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(years.min(), years.max())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=3, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, f"02_revenue_timeseries_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 02_revenue_timeseries_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
