"""Relationship between annual average prices and realised revenue.

For each technology a 2-D hexbin shows the joint density of
(realised price, realised revenue) across (path, year) cells, with the
relevant price axis chosen separately:

  * PSH:  day-ahead price vs total revenue  (DA dominates revenue)
  * BESS: FCR price        vs total revenue (FCR dominates revenue)

The first `BURNIN_YEARS` of each path are dropped before plotting. Year 0
revenues are several standard deviations above the long-run mean — a
transient driven by the optimiser starting from a fixed initial state —
and they would otherwise obscure the price–revenue relationship that the
figure is meant to show.

A linear OLS fit overlays the cloud, with the Pearson correlation
reported in the legend so the figure can stand alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    COLORS, FIGSIZE, LAYOUT, TECH_COLORS,
    apply_style, load_panel, save_fig,
)

BURNIN_YEARS = 5


def _config_for(tech: str) -> dict:
    if tech == "PSH":
        return dict(
            price_col="avg_DA_price",
            price_label="Annual average day-ahead price (NOK/MWh)",
            rev_scale=1e9,
            rev_label="Annual revenue (BNOK)",
        )
    return dict(
        price_col="avg_FCR_price",
        price_label="Annual average FCR price (NOK/MW/h)",
        rev_scale=1e6,
        rev_label="Annual revenue (MNOK)",
    )


def plot_one(tech: str) -> None:
    df = load_panel(tech)
    df = df[df["year"] >= BURNIN_YEARS]
    cfg = _config_for(tech)
    color = TECH_COLORS[tech]

    x = df[cfg["price_col"]].to_numpy()
    y = df["rev_total"].to_numpy() / cfg["rev_scale"]

    # OLS fit (purely descriptive — labelled in the legend).
    slope, intercept = np.polyfit(x, y, 1)
    rho = float(np.corrcoef(x, y)[0, 1])
    fit_x = np.linspace(x.min(), x.max(), 50)
    fit_y = slope * fit_x + intercept

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    hb = ax.hexbin(
        x, y, gridsize=50, mincnt=1,
        cmap="Purples" if tech == "BESS" else "Blues",
        linewidths=0,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("Observations per bin")
    cb.outline.set_visible(False)

    ax.plot(
        fit_x, fit_y,
        color=COLORS["highlight"], lw=LAYOUT["line_width"],
        label=f"OLS fit (ρ = {rho:.2f})",
    )

    # Tight y-limits — focus on the bulk; early-year transients are excluded.
    y_lo, y_hi = np.percentile(y, [0.5, 99.5])
    pad = 0.05 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlim(x.min(), x.max())

    ax.set_xlabel(cfg["price_label"])
    ax.set_ylabel(cfg["rev_label"])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=1, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, f"06_price_vs_revenue_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 06_price_vs_revenue_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
