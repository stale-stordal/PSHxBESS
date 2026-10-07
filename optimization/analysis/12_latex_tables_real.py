"""Generate the real-terms (deflated) annual revenue LaTeX table.

This mirrors the nominal revenue table in `11_latex_tables.py`, but first
removes the 2 %/yr inflation that `forecast_joint.py` applies to every price
path (`INFLATION_RATE = 0.02`). Each operating year's revenue is deflated by
`(1 + INFLATION_RATE) ** year` to express all values in year-0 (real) NOK,
after which the pooled Mean / Std / P5 / P95 are recomputed across the full
(path, year) panel.

Output:

  * `tab_revenue_real.tex` — annual revenue, DA and FCR blocks (MNOK, real)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import TABLE_DIR, load_panel

# Must match forecast_joint.INFLATION_RATE used to generate the price paths.
INFLATION_RATE = 0.02

STATS = [
    ("mean", "Mean"),
    ("std",  "Std"),
    ("p05",  "P5"),
    ("p95",  "P95"),
]

REV_BLOCKS = [
    ("Day-ahead", "rev_DA"),
    ("FCR",       "rev_FCR"),
]


def _fmt_dec(x: float, decimals: int = 2) -> str:
    s = f"{x:,.{decimals}f}"
    int_part, _, frac_part = s.partition(".")
    int_part = int_part.replace(",", r"\,")
    return f"{int_part}.{frac_part}" if frac_part else int_part


def deflated_stats(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    """Deflate per-(path, year) revenue to real terms and pool the stats."""
    deflator = (1.0 + INFLATION_RATE) ** df["year"]
    out = pd.DataFrame(index=["rev_DA", "rev_FCR"], columns=["mean", "std", "p05", "p95"], dtype=float)
    for metric in ("rev_DA", "rev_FCR"):
        real = df[metric] / deflator
        out.loc[metric] = [real.mean(), real.std(), real.quantile(0.05), real.quantile(0.95)]
    out["tech"] = tech
    out.index.name = "metric"
    return out.reset_index().set_index(["tech", "metric"])


def _block(label: str, metric: str, df: pd.DataFrame, scale: float = 1e6) -> list[str]:
    rows = []
    for i, (stat_key, stat_label) in enumerate(STATS):
        psh_val  = df.loc[("PSH",  metric), stat_key] / scale
        bess_val = df.loc[("BESS", metric), stat_key] / scale
        left = rf"\multirow{{4}}{{*}}{{{label}}}" if i == 0 else ""
        rows.append(f"  {left} & {stat_label} & {_fmt_dec(psh_val)} & {_fmt_dec(bess_val)} \\\\")
    return rows


def build_revenue_table(df: pd.DataFrame) -> str:
    parts = []
    for i, (label, metric) in enumerate(REV_BLOCKS):
        if i > 0:
            parts.append(r"  \midrule")
        parts.extend(_block(label, metric, df))
    body = "\n".join(parts)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Annual revenue summary in real (deflated) terms across 10\,000 simulated price paths and 80 operating years per path. Nominal revenues are deflated to year-0 prices at {INFLATION_RATE:.0%}/yr. All values in million NOK (MNOK).}}
\label{{tab:opt-revenue-real}}
\begin{{tabular}}{{llrr}}
\toprule
Component & Statistic & PSH & BESS \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    psh  = load_panel("PSH")
    bess = load_panel("BESS")
    stats = pd.concat([deflated_stats(psh, "PSH"), deflated_stats(bess, "BESS")])

    rev_tex = build_revenue_table(stats)
    out_path = TABLE_DIR / "tab_revenue_real.tex"
    out_path.write_text(rev_tex)

    print(f"  wrote {out_path.relative_to(TABLE_DIR.parent)}")
    print()
    print("=== Revenue table (real terms) ===")
    print(rev_tex)


if __name__ == "__main__":
    main()
