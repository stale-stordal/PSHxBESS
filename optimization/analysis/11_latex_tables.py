"""Generate LaTeX tables from `tables/summary_overall.csv`.

Two booktabs tables are written to `tables/`:

  * `tab_revenue.tex` — annual revenue, DA and FCR blocks (MNOK)
  * `tab_bids.tex`    — annual bid volumes, DA buy / DA sell / FCR blocks

Layout (both tables): the revenue/bid component is the multirow on the
left, with Mean / Std / P5 / P95 stacked in sub-rows. Columns are the
two technologies (PSH, BESS).
"""

from __future__ import annotations

import pandas as pd

from optimization.analysis._common import TABLE_DIR

STATS = [
    ("mean", "Mean"),
    ("std",  "Std"),
    ("p05",  "P5"),
    ("p95",  "P95"),
]


def _load_overall() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "summary_overall.csv").set_index(["tech", "metric"])


def _fmt_int(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", r"\,")


def _fmt_dec(x: float, decimals: int = 2) -> str:
    s = f"{x:,.{decimals}f}"
    int_part, _, frac_part = s.partition(".")
    int_part = int_part.replace(",", r"\,")
    return f"{int_part}.{frac_part}" if frac_part else int_part


def _block(label: str, metric: str, df: pd.DataFrame, fmt, scale: float = 1.0) -> list[str]:
    """Render one component block: 4 rows (Mean/Std/P5/P95), columns PSH and BESS."""
    rows = []
    for i, (stat_key, stat_label) in enumerate(STATS):
        psh_val  = df.loc[("PSH",  metric), stat_key] / scale
        bess_val = df.loc[("BESS", metric), stat_key] / scale
        left = rf"\multirow{{4}}{{*}}{{{label}}}" if i == 0 else ""
        rows.append(
            f"  {left} & {stat_label} & {fmt(psh_val)} & {fmt(bess_val)} \\\\"
        )
    return rows


# ---------------------------------------------------------------------------
# Revenue table — DA and FCR blocks; values in MNOK.
# ---------------------------------------------------------------------------

REV_BLOCKS = [
    ("Day-ahead", "rev_DA"),
    ("FCR",       "rev_FCR"),
]


def build_revenue_table(df: pd.DataFrame) -> str:
    parts = []
    for i, (label, metric) in enumerate(REV_BLOCKS):
        if i > 0:
            parts.append(r"  \midrule")
        parts.extend(_block(label, metric, df, _fmt_dec, scale=1e6))
    body = "\n".join(parts)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Annual revenue summary across 10\,000 simulated price paths and 80 operating years per path. All values in million NOK (MNOK).}}
\label{{tab:opt-revenue}}
\begin{{tabular}}{{llrr}}
\toprule
Component & Statistic & PSH & BESS \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


# ---------------------------------------------------------------------------
# Bid table — DA buy / DA sell / FCR blocks.
# ---------------------------------------------------------------------------

BID_BLOCKS = [
    ("DA buy (MWh/yr)",  "bid_DA_buy"),
    ("DA sell (MWh/yr)", "bid_DA_sell"),
    ("FCR (MW-h/yr)",    "bid_FCR"),
]


def build_bid_table(df: pd.DataFrame) -> str:
    parts = []
    for i, (label, metric) in enumerate(BID_BLOCKS):
        if i > 0:
            parts.append(r"  \midrule")
        parts.extend(_block(label, metric, df, _fmt_int))
    body = "\n".join(parts)
    return rf"""\begin{{table}}[ht]
\centering
\caption{{Annual bid volumes across 10\,000 simulated price paths and 80 operating years per path. DA bids are energy in MWh, FCR is reserved capacity-hours in MW-h.}}
\label{{tab:opt-bids}}
\begin{{tabular}}{{llrr}}
\toprule
Market & Statistic & PSH & BESS \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    df = _load_overall()

    rev_tex = build_revenue_table(df)
    bid_tex = build_bid_table(df)

    rev_path = TABLE_DIR / "tab_revenue.tex"
    bid_path = TABLE_DIR / "tab_bids.tex"
    rev_path.write_text(rev_tex)
    bid_path.write_text(bid_tex)

    print(f"  wrote {rev_path.relative_to(TABLE_DIR.parent)}")
    print(f"  wrote {bid_path.relative_to(TABLE_DIR.parent)}")
    print()
    print("=== Revenue table ===")
    print(rev_tex)
    print("=== Bid table ===")
    print(bid_tex)


if __name__ == "__main__":
    main()
