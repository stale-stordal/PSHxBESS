"""Generate the historical annual-revenue LaTeX table (2015-2025).

Mirrors the structure of `tab_revenue_real.tex` (Component x Statistic x
PSH/BESS), but the sample is the 11 historical operating years (one
deterministic dispatch per calendar year on actual NO2 day-ahead and FCR
prices) rather than the simulated price paths. Values are nominal historical
NOK, so no inflation deflation is applied.

Output:

  * `tab_revenue_historical.tex` — annual revenue, DA and FCR blocks (MNOK)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import TABLE_DIR

PATHS_DIR = Path(__file__).resolve().parents[1] / "results" / "paths"

RESULTS = {
    "PSH":  PATHS_DIR / "Pump_optimized_results_historical.csv",
    "BESS": PATHS_DIR / "Battery_optimized_results_historical.csv",
}

INFLATION_RATE = 0.02
BASE_YEAR = 2025

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


def historical_stats(path: Path, tech: str) -> pd.DataFrame:
    """Pool Mean / Std / P5 / P95 over the historical years for one tech,
    after inflating each year's nominal revenue to BASE_YEAR real NOK."""
    df = pd.read_csv(path)
    inflator = (1.0 + INFLATION_RATE) ** (BASE_YEAR - df["path_id"].astype(int))
    out = pd.DataFrame(index=["rev_DA", "rev_FCR"], columns=["mean", "std", "p05", "p95"], dtype=float)
    for metric in ("rev_DA", "rev_FCR"):
        s = df[metric] * inflator
        out.loc[metric] = [s.mean(), s.std(), s.quantile(0.05), s.quantile(0.95)]
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
\caption{{Annual revenue summary from the historical dispatch on actual NO2 day-ahead and FCR prices over the 11 operating years 2015--2025. Nominal revenues are inflated to {BASE_YEAR} prices at {INFLATION_RATE:.0%}/yr and the statistics pooled across years; all values in real million NOK (MNOK).}}
\label{{tab:opt-revenue-historical}}
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
    stats = pd.concat([
        historical_stats(RESULTS["PSH"],  "PSH"),
        historical_stats(RESULTS["BESS"], "BESS"),
    ])

    rev_tex = build_revenue_table(stats)
    out_path = TABLE_DIR / "tab_revenue_historical.tex"
    out_path.write_text(rev_tex)

    print(f"  wrote {out_path.relative_to(TABLE_DIR.parent)}")
    print()
    print("=== Revenue table (historical) ===")
    print(rev_tex)


if __name__ == "__main__":
    main()
