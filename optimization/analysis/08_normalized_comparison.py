"""PSH vs BESS on a normalised, per-MW-of-power basis.

Both technologies have 10 MWh of storage, but PSH (concept 2) is rated
at 1000 MW while the BESS baseline is 4.6 MW. To put the two cashflows
on a comparable footing, this figure plots NOK per MW of installed
power per year, separated into DA and FCR contributions.

The comparison highlights the structural difference between the two
revenue stacks: per MW, PSH earns far more from energy arbitrage; BESS
extracts almost all of its NOK from the ancillary market.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    BESS_MW, COLORS, FIGSIZE, LAYOUT, PSH_MW,
    apply_style, load_panel, save_fig,
)

# PSH = primary, BESS = secondary (thesis_style palette).
TECH_COLORS = {
    "PSH":  COLORS["main"],
    "BESS": COLORS["secondary"],
}


def _per_mw_year_mean(tech: str, mw: float) -> float:
    df = load_panel(tech)
    yearly = df.groupby("year")["rev_total"].mean().mean()
    return yearly / mw


def plot_grouped_bars() -> None:
    """Total revenue per MW per year, PSH vs BESS."""
    psh_tot  = _per_mw_year_mean("PSH",  PSH_MW)
    bess_tot = _per_mw_year_mean("BESS", BESS_MW)

    techs = ["PSH", "BESS"]
    total = np.array([psh_tot, bess_tot]) / 1e6   # NOK/MW/yr -> MNOK/MW/yr

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    x = np.arange(len(techs))
    width = 0.55

    ax.bar(
        x, total, width,
        color=[TECH_COLORS["PSH"], TECH_COLORS["BESS"]],
        edgecolor="white", linewidth=0.5,
    )

    for i, tot_v in enumerate(total):
        ax.text(i, tot_v, f"{tot_v:,.2f}", ha="center", va="bottom",
                color=COLORS["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(techs)
    ax.set_ylim(top=total.max() * 1.12)
    ax.set_ylabel("Mean annual revenue (MNOK / MW)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.2f}"))

    plt.tight_layout()
    save_fig(fig, "08_normalized_comparison_per_mw")
    plt.close(fig)


def plot_normalized_timeseries() -> None:
    """Mean annual revenue per MW over the horizon for both technologies."""
    psh  = load_panel("PSH")
    bess = load_panel("BESS")

    psh_yr  = psh.groupby("year")["rev_total"].mean()  / PSH_MW  / 1e6
    bess_yr = bess.groupby("year")["rev_total"].mean() / BESS_MW / 1e6

    years = psh_yr.index.to_numpy()

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.plot(years, psh_yr.to_numpy(),
            color=TECH_COLORS["PSH"],  lw=LAYOUT["line_width"],     label="PSH")
    ax.plot(years, bess_yr.to_numpy(),
            color=TECH_COLORS["BESS"], lw=LAYOUT["line_width_sec"], label="BESS")

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean annual revenue (MNOK / MW)")
    ax.set_xlim(years.min(), years.max())
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.2f}"))

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=2, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, "08_normalized_comparison_timeseries")
    plt.close(fig)


def main() -> None:
    apply_style(layout="full")
    plot_grouped_bars()
    print("  saved 08_normalized_comparison_per_mw.pdf")
    plot_normalized_timeseries()
    print("  saved 08_normalized_comparison_timeseries.pdf")


if __name__ == "__main__":
    main()
