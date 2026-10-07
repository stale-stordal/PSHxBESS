"""Sensitivity analysis: optimizing on real (deflated) prices vs nominal prices.

The price paths fed to the dispatch LP carry a constant 2 %/yr nominal
inflation (`forecast_joint.INFLATION_RATE`). This script reports what the
optimizer results look like in real (year-0) terms instead of nominal terms.

Why no re-optimization is required
----------------------------------
The LP is solved independently for each operating year (years are coupled only
through a 40-60 % end-of-year SoC band, not through prices). Deflating every
price in year ``y`` by the constant ``(1 + r) ** y`` scales that year's
objective by a positive constant, so the argmax — the optimal dispatch — is
unchanged. The within-year inflation tilt (prices drift ~2 % across the 8 760
hours of a year, i.e. ~0.005 %/day) is negligible against daily arbitrage, so
re-running the LP on truly deflated price files would reproduce the same bids.
Real revenue is therefore exactly the nominal optimizer revenue divided by the
per-year deflator — computable directly from the saved panels.

Outputs
-------
  * figures/13_sensitivity_real_prices.{pdf,png}  — nominal vs real annual
    revenue (mean + real percentile band), PSH and BESS panels.
  * tables/tab_sensitivity_real.tex               — pooled nominal vs real
    revenue stats with the real/nominal ratio.
"""

from __future__ import annotations

import pandas as pd

from optimization.analysis._common import (
    COLORS,
    FIGSIZE,
    TABLE_DIR,
    apply_style,
    load_panel,
    save_fig,
)

import matplotlib.pyplot as plt  # noqa: E402 — after _common applies thesis style

# Must match forecast_joint.INFLATION_RATE used to generate the price paths.
INFLATION_RATE = 0.02

TECHS = [
    ("PSH",  COLORS["main"]),
    ("BESS", COLORS["secondary"]),
]

STATS = [
    ("mean", "Mean"),
    ("std",  "Std"),
    ("p05",  "P5"),
    ("p95",  "P95"),
]


def deflate(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with revenue columns converted to year-0 real NOK."""
    out = df.copy()
    deflator = (1.0 + INFLATION_RATE) ** out["year"].to_numpy()
    for col in ("rev_DA", "rev_FCR", "rev_total"):
        out[col] = out[col] / deflator
    return out


def yearly_band(df: pd.DataFrame, col: str, scale: float = 1e6) -> dict:
    g = df.groupby("year")[col]
    return {
        "years": g.mean().index.to_numpy(),
        "mean":  g.mean().to_numpy() / scale,
        "p25":   g.quantile(0.25).to_numpy() / scale,
        "p75":   g.quantile(0.75).to_numpy() / scale,
    }


# ---------------------------------------------------------------------------
# Figure: nominal vs real annual revenue
# ---------------------------------------------------------------------------

def build_figure(panels: dict):
    apply_style()
    fig, axes = plt.subplots(
        2, 1,
        figsize=(FIGSIZE["wide"][0], 7.8),
        sharex=True,
        gridspec_kw={"hspace": 0.16},
    )

    for ax, (tech, color) in zip(axes, TECHS):
        nom = panels[tech]["nominal"]
        rea = panels[tech]["real"]

        # Real percentile band (interquartile) as context.
        ax.fill_between(rea["years"], rea["p25"], rea["p75"],
                        color=color, alpha=0.18, lw=0,
                        label="Real, 25-75th pct.")
        ax.plot(nom["years"], nom["mean"],
                color=COLORS["highlight"], lw=2.2, label="Nominal mean")
        ax.plot(rea["years"], rea["mean"],
                color=color, lw=2.2, label="Real mean (deflated)")

        ax.set_ylabel(f"{tech} annual revenue\n(MNOK)")
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, ncol=3)

    axes[-1].set_xlabel("Operating year")
    axes[0].set_xlim(panels["PSH"]["nominal"]["years"][0],
                     panels["PSH"]["nominal"]["years"][-1])
    return fig


# ---------------------------------------------------------------------------
# Table: pooled nominal vs real (rev_total)
# ---------------------------------------------------------------------------

def pooled_stats(series: pd.Series, scale: float = 1e6) -> dict:
    return {
        "mean": series.mean() / scale,
        "std":  series.std() / scale,
        "p05":  series.quantile(0.05) / scale,
        "p95":  series.quantile(0.95) / scale,
    }


def _fmt(x: float, decimals: int = 2) -> str:
    s = f"{x:,.{decimals}f}"
    int_part, _, frac = s.partition(".")
    int_part = int_part.replace(",", r"\,")
    return f"{int_part}.{frac}" if frac else int_part


def build_table(rows: dict) -> str:
    """rows[tech] = (nominal_stats, real_stats) for rev_total."""
    body = []
    for ti, (tech, _) in enumerate(TECHS):
        if ti > 0:
            body.append(r"  \midrule")
        nom, rea = rows[tech]
        for i, (key, label) in enumerate(STATS):
            left = rf"\multirow{{4}}{{*}}{{{tech}}}" if i == 0 else ""
            ratio = 100.0 * rea[key] / nom[key] if nom[key] else float("nan")
            body.append(
                f"  {left} & {label} & {_fmt(nom[key])} & {_fmt(rea[key])} "
                f"& {_fmt(ratio, 1)} \\\\"
            )
    body = "\n".join(body)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Sensitivity of optimizer revenue to deflating the {INFLATION_RATE:.0%}/yr
inflation embedded in the price paths. Total annual revenue pooled across
10\,000 simulated price paths and 80 operating years per path, in nominal terms
and in real (year-0, deflated) terms. The optimal dispatch is invariant to the
per-year deflator, so real revenue equals nominal revenue divided by
$(1+{INFLATION_RATE:.2f})^{{\,\text{{year}}}}$. All monetary values in million NOK (MNOK).}}
\label{{tab:opt-sensitivity-real}}
\begin{{tabular}}{{llrrr}}
\toprule
Technology & Statistic & Nominal & Real & Real/Nominal (\%) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    panels = {}
    table_rows = {}
    for tech, _ in TECHS:
        nom_df = load_panel(tech)
        real_df = deflate(nom_df)
        panels[tech] = {
            "nominal": yearly_band(nom_df, "rev_total"),
            "real":    yearly_band(real_df, "rev_total"),
        }
        table_rows[tech] = (
            pooled_stats(nom_df["rev_total"]),
            pooled_stats(real_df["rev_total"]),
        )

    fig = build_figure(panels)
    fig_path = save_fig(fig, "13_sensitivity_real_prices")
    plt.close(fig)

    tex = build_table(table_rows)
    tex_path = TABLE_DIR / "tab_sensitivity_real.tex"
    tex_path.write_text(tex)

    print(f"  wrote {fig_path.relative_to(fig_path.parents[1])}")
    print(f"  wrote {tex_path.relative_to(TABLE_DIR.parent)}")
    print()
    for tech, _ in TECHS:
        nom, rea = table_rows[tech]
        drop = 100.0 * (1 - rea["mean"] / nom["mean"])
        print(f"  {tech}: pooled mean nominal {nom['mean']:.2f} MNOK -> "
              f"real {rea['mean']:.2f} MNOK ({drop:.1f}% lower in real terms)")
    print()
    print(tex)


if __name__ == "__main__":
    main()
