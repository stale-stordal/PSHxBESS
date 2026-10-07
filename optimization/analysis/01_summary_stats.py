"""Aggregate summary statistics for the optimiser output.

Produces three CSVs in `tables/`:

  * `summary_overall.csv`    — pooled (path, year) statistics per tech
  * `summary_per_year.csv`   — per-year mean/percentile envelope per tech
  * `summary_per_path.csv`   — per-path lifetime totals (mean, percentiles)

These tables feed the narrative in the thesis Results chapter and are the
numerical backing for the figures produced by sibling scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optimization.analysis._common import TABLE_DIR, load_panel

REV_COLS = ["rev_DA", "rev_FCR", "rev_total"]
BID_COLS = ["bid_DA_buy", "bid_DA_sell", "bid_FCR"]
PRICE_COLS = ["avg_DA_price", "avg_FCR_price"]


def overall_summary(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    """Pooled summary across all (path, year) pairs."""
    cols = REV_COLS + BID_COLS + PRICE_COLS + ["SoC_end_frac"]
    desc = df[cols].agg([
        "mean", "std", "min",
        lambda s: s.quantile(0.05),
        lambda s: s.quantile(0.25),
        "median",
        lambda s: s.quantile(0.75),
        lambda s: s.quantile(0.95),
        "max",
    ]).T
    desc.columns = ["mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"]
    desc.insert(0, "tech", tech)
    desc.index.name = "metric"
    return desc.reset_index()


def per_year_summary(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    """Per-year mean and percentile envelope of total revenue."""
    g = df.groupby("year")
    out = pd.DataFrame({
        "mean_rev_total": g["rev_total"].mean(),
        "std_rev_total":  g["rev_total"].std(),
        "p05_rev_total":  g["rev_total"].quantile(0.05),
        "p50_rev_total":  g["rev_total"].quantile(0.50),
        "p95_rev_total":  g["rev_total"].quantile(0.95),
        "mean_rev_DA":    g["rev_DA"].mean(),
        "mean_rev_FCR":   g["rev_FCR"].mean(),
        "fcr_share_mean": (g["rev_FCR"].sum() / g["rev_total"].sum()),
        "mean_DA_price":  g["avg_DA_price"].mean(),
        "mean_FCR_price": g["avg_FCR_price"].mean(),
    }).reset_index()
    out.insert(0, "tech", tech)
    return out


def per_path_summary(df: pd.DataFrame, tech: str) -> pd.DataFrame:
    """Lifetime total revenue per path."""
    totals = df.groupby("path_id")["rev_total"].sum().rename("lifetime_rev_total")
    summary = totals.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame().T
    summary.insert(0, "tech", tech)
    summary.insert(1, "metric", "lifetime_rev_total")
    return summary


def write_table(df: pd.DataFrame, name: str) -> None:
    path = TABLE_DIR / name
    df.to_csv(path, index=False)
    print(f"  wrote {path.relative_to(TABLE_DIR.parent)}")


def main() -> None:
    psh  = load_panel("PSH")
    bess = load_panel("BESS")

    print("Overall summary…")
    write_table(
        pd.concat([overall_summary(psh, "PSH"), overall_summary(bess, "BESS")], ignore_index=True),
        "summary_overall.csv",
    )

    print("Per-year summary…")
    write_table(
        pd.concat([per_year_summary(psh, "PSH"), per_year_summary(bess, "BESS")], ignore_index=True),
        "summary_per_year.csv",
    )

    print("Per-path lifetime summary…")
    write_table(
        pd.concat([per_path_summary(psh, "PSH"), per_path_summary(bess, "BESS")], ignore_index=True),
        "summary_per_path.csv",
    )

    # Lightweight headline numbers — handy when drafting the chapter.
    print("\nHeadline figures")
    for tech, df in (("PSH", psh), ("BESS", bess)):
        rev = df["rev_total"]
        fcr_share = df["rev_FCR"].sum() / df["rev_total"].sum()
        cv = rev.std() / rev.mean()
        print(
            f"  {tech:5s}  mean rev = {rev.mean():.3e} NOK/yr   "
            f"FCR share = {fcr_share:6.2%}   CV = {cv:5.2%}"
        )


if __name__ == "__main__":
    main()
