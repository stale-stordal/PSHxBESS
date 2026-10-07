"""FCR share of total revenue, year by year, both technologies on one axis.

The figure makes the structural divergence between the two technologies
visible at a glance: PSH is a DA-arbitrage asset that treats FCR as a
small premium (~6%); BESS is the opposite, with FCR providing the bulk
of revenue (~95%) and DA arbitrage acting as a thin top-up. The thin
shading is the per-year P5–P95 envelope across paths, showing how
stable each share is.
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
    apply_style, load_panel, save_fig,
)


def _share_stats(tech: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = load_panel(tech)
    df = df.assign(share=df["rev_FCR"] / df["rev_total"])
    g = df.groupby("year")["share"]
    years = g.mean().index.to_numpy()
    return (
        years,
        g.mean().to_numpy(),
        g.quantile(0.05).to_numpy(),
        g.quantile(0.95).to_numpy(),
    )


def main() -> None:
    apply_style(layout="half")

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    for tech in ("PSH", "BESS"):
        years, mean, p05, p95 = _share_stats(tech)
        color = TECH_COLORS[tech]
        ax.fill_between(years, p05 * 100, p95 * 100,
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(years, mean * 100,
                color=color, lw=LAYOUT["line_width"], label=tech)

    ax.set_xlabel("Year")
    ax.set_ylabel("FCR share of total revenue (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 49)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=2, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, "09_fcr_share")
    plt.close(fig)
    print("  saved 09_fcr_share.pdf")


if __name__ == "__main__":
    main()
