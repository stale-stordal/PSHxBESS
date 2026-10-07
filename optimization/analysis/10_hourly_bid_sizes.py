"""Average hourly bid size per market, per technology.

Annual bid volumes are converted to an average per-hour quantity by
dividing by 8760 hours/year. Both absolute MW and the fraction of rated
power capacity are reported, so the table is informative regardless of
plant size. Saves a CSV in `tables/` and prints a formatted summary.

Definitions:

  * DA buy  — average power drawn (charging) per hour, MW
  * DA sell — average power delivered (discharging) per hour, MW
  * DA combined — buy + sell per hour, MW (gross traded volume)
  * FCR — average MW reserved for FCR per hour
  * Combined — DA combined + FCR; the average MW of bid activity per hour
"""

from __future__ import annotations

import pandas as pd

from optimization.analysis._common import BESS_MW, PSH_MW, TABLE_DIR, load_panel

HOURS_PER_YEAR = 8_760


def hourly_bids(tech: str, mw: float) -> dict:
    df = load_panel(tech)
    means = df[["bid_DA_buy", "bid_DA_sell", "bid_FCR"]].mean()

    da_buy_h     = means["bid_DA_buy"]  / HOURS_PER_YEAR
    da_sell_h    = means["bid_DA_sell"] / HOURS_PER_YEAR
    fcr_h        = means["bid_FCR"]     / HOURS_PER_YEAR
    da_combined  = da_buy_h + da_sell_h
    grand_total  = da_combined + fcr_h

    return {
        "tech":              tech,
        "rated_mw":          mw,
        "da_buy_mw":         da_buy_h,
        "da_sell_mw":        da_sell_h,
        "da_combined_mw":    da_combined,
        "fcr_mw":            fcr_h,
        "combined_mw":       grand_total,
        "da_buy_pct_rated":  100 * da_buy_h    / mw,
        "da_sell_pct_rated": 100 * da_sell_h   / mw,
        "da_combined_pct":   100 * da_combined / mw,
        "fcr_pct_rated":     100 * fcr_h       / mw,
        "combined_pct":      100 * grand_total / mw,
    }


def main() -> None:
    rows = [
        hourly_bids("PSH",  PSH_MW),
        hourly_bids("BESS", BESS_MW),
    ]
    table = pd.DataFrame(rows)

    out_path = TABLE_DIR / "hourly_bid_sizes.csv"
    table.to_csv(out_path, index=False)
    print(f"  wrote {out_path.relative_to(TABLE_DIR.parent)}\n")

    fmt = (
        "{tech:5s}  rated {rated_mw:7.1f} MW  |  "
        "DA buy {da_buy_mw:9.3f} MW ({da_buy_pct_rated:5.2f}%)  "
        "DA sell {da_sell_mw:9.3f} MW ({da_sell_pct_rated:5.2f}%)  "
        "DA combined {da_combined_mw:9.3f} MW ({da_combined_pct:5.2f}%)  "
        "FCR {fcr_mw:9.3f} MW ({fcr_pct_rated:5.2f}%)  "
        "Combined {combined_mw:9.3f} MW ({combined_pct:5.2f}%)"
    )
    print("Average hourly bid size (mean across all paths and years)")
    print("-" * 80)
    for row in rows:
        print(fmt.format(**row))


if __name__ == "__main__":
    main()
