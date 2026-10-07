"""Mean optimised bids over the horizon.

`bid_DA_buy` and `bid_DA_sell` are annual energy volumes (MWh) traded in
the day-ahead market. `bid_FCR` is annual capacity-hours (MW-h) reserved
for FCR. The two markets serve different purposes — DA is energy
arbitrage, FCR is capacity reservation — but both quantities reduce to
MWh per year, so all three series share a single y-axis in GWh.

Three figures are produced:

  * `05_bid_behavior_psh`       — PSH only, shared y-axis (GWh per year)
  * `05_bid_behavior_bess`      — BESS only, shared y-axis (GWh per year)
  * `05_bid_behavior_combined`  — both techs on a single, shared y-axis,
    expressed as the average hourly bid in % of rated power capacity.
    This normalisation is the only way to compare a 1000 MW PSH plant
    with a 4.6 MW BESS unit on one axis.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import (
    BESS_MW, COLORS, FIGSIZE, LAYOUT, PSH_MW, TECH_COLORS,
    apply_style, load_panel, save_fig,
)

HOURS_PER_YEAR = 8_760

# Spot (DA) uses the primary colour, FCR the secondary colour.
SPOT_COLOR = COLORS["main"]
FCR_COLOR = COLORS["secondary"]


def plot_one(
    tech: str,
    step: float | None = None,
    break_below: float | None = None,
    break_above: float | None = None,
    top_hi: float | None = None,
) -> None:
    df = load_panel(tech)
    yearly = df.groupby("year")[["bid_DA_buy", "bid_DA_sell", "bid_FCR"]].mean()

    scale = 1e3  # MWh -> GWh, same for DA and FCR

    years   = yearly.index.to_numpy()
    da_buy  = yearly["bid_DA_buy"].to_numpy()  / scale
    da_sell = yearly["bid_DA_sell"].to_numpy() / scale
    fcr     = yearly["bid_FCR"].to_numpy()     / scale

    # FCR sits far above the DA series, leaving a wide empty band in between.
    # Use a broken axis that keeps the SAME scale on both panels (identical
    # GWh per unit height): the empty middle is simply cut out, so the FCR
    # slope is not exaggerated relative to the DA series. The panel heights are
    # set proportional to the data range each shows, and both share one tick
    # step.
    da_max = max(da_buy.max(), da_sell.max())

    def nice_step(span, target=5):
        raw = span / target
        mag = 10.0 ** np.floor(np.log10(raw))
        for m in (1, 2, 5, 10):
            if raw <= m * mag:
                return m * mag
        return 10 * mag

    # The gap is cut between `break_below` (top of the bottom panel) and
    # `break_above` (bottom of the top panel); both default to a tight fit.
    bot_lo = 0.0
    bot_hi = break_below if break_below is not None else da_max * 1.15
    if step is None:
        step = nice_step(bot_hi - bot_lo)
    margin = step * 0.5
    top_lo = break_above if break_above is not None else fcr.min() - margin
    if top_hi is None:
        top_hi = fcr.max() + margin

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=FIGSIZE["half"], sharex=True,
        gridspec_kw={"height_ratios": [top_hi - top_lo, bot_hi - bot_lo],
                     "hspace": 0.12},
    )

    def draw(ax):
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        l1, = ax.plot(years, da_buy,  color=SPOT_COLOR, lw=LAYOUT["line_width"],
                      linestyle="-",  label="DA buy")
        l2, = ax.plot(years, da_sell, color=SPOT_COLOR, lw=LAYOUT["line_width_sec"],
                      linestyle="--", label="DA sell")
        l3, = ax.plot(years, fcr,     color=FCR_COLOR,  lw=LAYOUT["line_width"],
                      linestyle="-",  label="FCR")
        return l1, l2, l3

    draw(ax_top)
    l1, l2, l3 = draw(ax_bot)

    ax_top.set_ylim(top_lo, top_hi)
    ax_bot.set_ylim(bot_lo, bot_hi)
    ax_bot.set_xlim(years.min(), years.max())
    # Same tick step and formatting on both panels reinforces the shared scale.
    dec = max(0, int(-np.floor(np.log10(step))))
    fmt = mticker.FormatStrFormatter(f"%.{dec}f")
    for ax in (ax_top, ax_bot):
        ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(fmt)

    # Hide the inner spines/ticks so the two panels read as one broken axis.
    ax_top.spines["bottom"].set_visible(False)
    ax_top.tick_params(axis="x", which="both", length=0)
    ax_bot.spines["top"].set_visible(False)

    # Diagonal break marks across the cut.
    d = 0.012
    bk = dict(color="#3A3A3A", clip_on=False, lw=LAYOUT["spine_lw"])
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **bk)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **bk)

    ax_bot.set_xlabel("Year")
    fig.supylabel("Annual bid volume (GWh)", x=0.0, fontsize=plt.rcParams["axes.labelsize"])

    ax_top.legend(
        handles=[l1, l2, l3],
        loc="upper center", bbox_to_anchor=(0.5, 1.30),
        ncol=3, frameon=False,
    )

    # No tight_layout: it is incompatible with the broken-axis spines; the
    # bbox_inches="tight" crop in save_fig handles the margins.
    save_fig(fig, f"05_bid_behavior_{tech.lower()}")
    plt.close(fig)


def plot_combined() -> None:
    """Both technologies on a shared y-axis, normalised to % of rated power.

    Average hourly bid (MW) ÷ rated power (MW) × 100. DA buy/sell are
    energy bids; the conversion is bid_MWh_per_yr / 8760 hours = average MW.
    FCR is already capacity-hours; bid_MW-h_per_yr / 8760 = average MW.
    """
    rated = {"PSH": PSH_MW, "BESS": BESS_MW}

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    market_styles = {
        "bid_DA_buy":  ("-",  "DA buy"),
        "bid_DA_sell": ("--", "DA sell"),
        "bid_FCR":     (":",  "FCR"),
    }

    handles = []
    for tech in ("PSH", "BESS"):
        df = load_panel(tech)
        yearly = df.groupby("year")[list(market_styles)].mean()
        years = yearly.index.to_numpy()
        for col, (ls, market_label) in market_styles.items():
            pct = yearly[col].to_numpy() / HOURS_PER_YEAR / rated[tech] * 100.0
            line, = ax.plot(
                years, pct,
                color=TECH_COLORS[tech],
                linestyle=ls,
                lw=LAYOUT["line_width"] if ls == "-" else LAYOUT["line_width_sec"],
                label=f"{tech} {market_label}",
            )
            handles.append(line)

    ax.set_xlabel("Year")
    ax.set_ylabel("Average hourly bid (% of rated power)")
    ax.set_xlim(0, 49)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

    # 6 entries on the short half-width canvas: a 2-column legend stays
    # narrower than the axes (clears the y-label) and is lifted clear of the
    # plot so it doesn't overlap the data.
    ax.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.34),
        ncol=2, frameon=False,
    )

    plt.tight_layout()
    save_fig(fig, "05_bid_behavior_combined")
    plt.close(fig)


def main() -> None:
    apply_style(layout="half")
    opts = {
        "PSH":  {"break_below": 85.0, "break_above": 465.0},
        "BESS": {"step": 1.0, "break_below": 1.5, "break_above": 35.0, "top_hi": 38.0},
    }
    for tech in ("PSH", "BESS"):
        plot_one(tech, **opts[tech])
        print(f"  saved 05_bid_behavior_{tech.lower()}.pdf")
    plot_combined()
    print("  saved 05_bid_behavior_combined.pdf")


if __name__ == "__main__":
    main()
