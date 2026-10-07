"""Mean DA and FCR revenue, stacked, over the 50-year horizon.

The two technologies sit at very different scales (PSH ~ 1 BNOK/yr,
BESS ~ 5 MNOK/yr) so each gets its own figure. The composition is the
point of interest: PSH earns ~94% of revenue from the day-ahead market
and treats FCR as a small premium, while BESS is the mirror image — its
cashflow is dominated by the FCR market.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    COLORS, FIGSIZE, LAYOUT,
    apply_style, load_panel, save_fig,
)


def _scale_and_label(tech: str) -> tuple[float, str]:
    return 1e6, "Mean annual revenue (MNOK)"


def plot_one(tech: str) -> None:
    df = load_panel(tech)
    yearly = df.groupby("year")[["rev_DA", "rev_FCR"]].mean()

    scale, ylabel = _scale_and_label(tech)
    years = yearly.index.to_numpy()
    da   = yearly["rev_DA"].to_numpy()  / scale
    fcr  = yearly["rev_FCR"].to_numpy() / scale

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.stackplot(
        years, da, fcr,
        colors=[COLORS["main"], COLORS["secondary"]],
        alpha=0.85,
        labels=["Day-ahead", "FCR"],
    )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(years.min(), years.max())
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )

    ax.legend(
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=2, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, f"03_revenue_split_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 03_revenue_split_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
