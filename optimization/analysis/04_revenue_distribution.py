"""Distribution of lifetime mean annual revenue per path.

Each path has 50 years of operating revenue; we average within a path to
collapse to a single representative value, then draw the distribution
across the 10000 paths. This is the right marginal to show the variance
that survives the price-process simulation: short-horizon noise averages
out within a path, and what's left is path-to-path uncertainty in the
realised price level.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from optimization.analysis._common import (
    FIGSIZE, LAYOUT, TECH_COLORS,
    apply_style, load_panel, save_fig,
)


# Constant nominal inflation embedded in the price paths
# (forecast_joint.INFLATION_RATE). Deflating by (1+r)**year converts the
# optimizer's nominal revenue to real (year-0) terms; see 13_sensitivity_real_prices.py.
INFLATION_RATE = 0.02


def _scale_and_label(tech: str, real: bool) -> tuple[float, str]:
    terms = "real, year-0" if real else "nominal"
    return 1e6, f"Mean annual revenue per path (MNOK, {terms})"


def _deflate(df):
    deflator = (1.0 + INFLATION_RATE) ** df["year"].to_numpy()
    return df["rev_total"].to_numpy() / deflator


def plot_one(tech: str, real: bool = False) -> None:
    df = load_panel(tech)
    if real:
        df = df.assign(rev_total=_deflate(df))
    per_path_mean = df.groupby("path_id")["rev_total"].mean()

    scale, xlabel = _scale_and_label(tech, real)
    color = TECH_COLORS[tech]
    values = per_path_mean.to_numpy() / scale

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    # Focus on the bulk: the right tail (~1% of paths) is long and would
    # otherwise squash the distribution into a thin spike.
    x_hi = float(np.percentile(values, 99))

    ax.hist(
        values, bins=60, range=(float(values.min()), x_hi), density=False,
        color=color, alpha=0.55, edgecolor=color, linewidth=0.4,
        label="Distribution",
    )

    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    ax.axvline(mean_val,   color=color, linestyle="-",  lw=LAYOUT["line_width"],
               label=f"Mean = {mean_val:.1f}")
    ax.axvline(median_val, color=color, linestyle="--", lw=LAYOUT["line_width_sec"],
               label=f"Median = {median_val:.1f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of paths")
    ax.set_xlim(float(values.min()), x_hi)

    # Right-skewed distribution leaves the upper-right corner empty; place the
    # legend there, stacked, instead of on one wide row above the figure.
    ax.legend(loc="upper right", ncol=1, frameon=False)

    plt.tight_layout()
    suffix = "_real" if real else ""
    name = f"04_revenue_distribution_{tech.lower()}{suffix}"
    save_fig(fig, name)
    plt.close(fig)
    return name


def main() -> None:
    apply_style(layout="half")
    for tech in ("PSH", "BESS"):
        for real in (False, True):
            name = plot_one(tech, real=real)
            print(f"  saved {name}.pdf")


if __name__ == "__main__":
    main()
