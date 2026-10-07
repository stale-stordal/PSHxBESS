"""Twelve example revenue trajectories drawn at random.

The percentile envelope plot collapses path-to-path differences into a
band; this companion view shows individual paths so that the reader can
see the year-to-year roughness that the simulation produces, plus
three reference paths picked at the 10th / 50th / 90th percentile of
lifetime mean revenue.

A reproducible seed is used so that the figure is stable across reruns.
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

RNG = np.random.default_rng(seed=42)
N_RANDOM = 12


def _scale_and_label(tech: str) -> tuple[float, str]:
    if tech == "PSH":
        return 1e9, "Annual revenue (BNOK)"
    return 1e6, "Annual revenue (MNOK)"


def plot_one(tech: str) -> None:
    df = load_panel(tech)
    pivot = df.pivot(index="year", columns="path_id", values="rev_total")

    scale, ylabel = _scale_and_label(tech)
    color = TECH_COLORS[tech]
    years = pivot.index.to_numpy()

    # Pick reference paths by lifetime mean revenue percentile.
    lifetime_mean = pivot.mean(axis=0)
    ranks = lifetime_mean.rank(pct=True)
    ref_ids = {
        "P10": (ranks - 0.10).abs().idxmin(),
        "P50": (ranks - 0.50).abs().idxmin(),
        "P90": (ranks - 0.90).abs().idxmin(),
    }
    ref_styles = {
        "P10": ("--", COLORS["highlight"]),
        "P50": ("-",  COLORS["neutral"]),
        "P90": (":",  COLORS["main"]),
    }

    sample_ids = RNG.choice(pivot.columns.values, size=N_RANDOM, replace=False)

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    for pid in sample_ids:
        ax.plot(years, pivot[pid].to_numpy() / scale,
                color=color, lw=0.9, alpha=0.35)

    handles = []
    for tag, pid in ref_ids.items():
        ls, c = ref_styles[tag]
        line, = ax.plot(
            years, pivot[pid].to_numpy() / scale,
            color=c, lw=LAYOUT["line_width"], linestyle=ls,
            label=f"{tag} path",
        )
        handles.append(line)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(years.min(), years.max())
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
        ncol=3, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, f"07_sample_paths_{tech.lower()}")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        plot_one(tech)
        print(f"  saved 07_sample_paths_{tech.lower()}.pdf")


if __name__ == "__main__":
    main()
