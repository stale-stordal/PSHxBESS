"""
Created Spring 2026

@author: einar-m, matheaab, odmoe
"""

"""
Run the optimal LP dispatch on historical NO2 day-ahead and FCR prices
(2015-2025) and save the results to the standard results/paths out-dir.

Each calendar year is treated as one independent path of exactly 365*24
hours: every year is reindexed to a complete hourly grid, the few DST/missing
hours are time-interpolated, and the extra day of leap years is dropped so all
years are equal length. The prepared price columns are handed to the existing
_optimize_all_paths machinery unchanged.
"""

import sys
import time
from pathlib import Path

import pandas as pd

# optimal_dispatch_LP imports `from storage import ...`, so its own directory
# must be importable when running this script from anywhere.
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from optimization.optimal_dispatch_LP import (  # noqa: E402
    NT,
    Battery,
    Pump,
    OUTPUT_DIR,
    _optimize_all_paths,
)

REPO_ROOT = HERE.parent
DA_FILE   = REPO_ROOT / "data" / "prices" / "DA"  / "DA_NO2_NOK.csv"
FCR_FILE  = REPO_ROOT / "data" / "prices" / "FCR" / "FCR_NO2_NOK.csv"

INPUT_DIR = HERE / "input_files"
SPOT_OUT  = INPUT_DIR / "spot_historical.csv"
FCR_OUT   = INPUT_DIR / "FCR_historical.csv"

YEAR_START = 2015
YEAR_END   = 2025
NT_YEAR    = 365 * NT  # 8760 hourly steps per path


def _build_historical_inputs() -> tuple[str, str]:
    """Load the combined NO2 price series, align day-ahead and FCR on common
    hourly timestamps, and write one column per calendar year (each exactly
    NT_YEAR hours) to spot/FCR CSVs in the standard input_files dir.

    Returns the two written file paths.
    """
    da = pd.read_csv(DA_FILE,  parse_dates=["Timestamp"])
    fc = pd.read_csv(FCR_FILE, parse_dates=["Timestamp"])

    merged = (
        da.merge(fc, on="Timestamp", how="inner")
          .set_index("Timestamp")
          .sort_index()
    )
    # Collapse duplicate DST fall-back hours.
    merged = merged[~merged.index.duplicated(keep="first")]

    spot_cols: dict[str, list] = {}
    fcr_cols:  dict[str, list] = {}

    for year in range(YEAR_START, YEAR_END + 1):
        block = merged[merged.index.year == year]
        if block.empty:
            print(f"  Skipping {year}: no data")
            continue

        # Complete hourly grid for the calendar year, then fill the handful of
        # DST/missing hours by time interpolation (and edge ffill/bfill).
        grid = pd.date_range(f"{year}-01-01 00:00", f"{year}-12-31 23:00", freq="h")
        block = block.reindex(grid)
        block["Price_DA"]  = block["Price_DA"].interpolate("time").ffill().bfill()
        block["Price_FCR"] = block["Price_FCR"].interpolate("time").ffill().bfill()

        # Equal-length paths: drop the extra day of leap years.
        block = block.iloc[:NT_YEAR]
        if len(block) != NT_YEAR:
            print(f"  Skipping {year}: only {len(block)} hours after alignment")
            continue

        label = str(year)
        spot_cols[label] = block["Price_DA"].to_numpy()
        fcr_cols[label]  = block["Price_FCR"].to_numpy()

    timestamp = pd.RangeIndex(NT_YEAR)
    spot_df = pd.DataFrame({"timestamp": timestamp, **spot_cols})
    fcr_df  = pd.DataFrame({"timestamp": timestamp, **fcr_cols})

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    spot_df.to_csv(SPOT_OUT, index=False)
    fcr_df.to_csv(FCR_OUT, index=False)

    years = [c for c in spot_df.columns if c != "timestamp"]
    print(f"Prepared {len(years)} historical years ({years[0]}-{years[-1]}), "
          f"{NT_YEAR} hours each.")
    print(f"  Spot -> {SPOT_OUT}")
    print(f"  FCR  -> {FCR_OUT}")
    return str(SPOT_OUT), str(FCR_OUT)


def run_historical_dispatch(storages=None) -> None:
    """Build the historical inputs and run the LP dispatch for each storage,
    saving one results CSV per storage to the standard out-dir.
    """
    if storages is None:
        storages = [Battery(), Pump()]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spot_file, fcr_file = _build_historical_inputs()

    t_start = time.perf_counter()
    for storage in storages:
        storage_cls = type(storage).__name__
        output_file = OUTPUT_DIR / f"{storage_cls}_optimized_results_historical.csv"

        print(f"\n=== Running {storage_cls} on historical prices ===")
        results = _optimize_all_paths(storage, spot_file, fcr_file)
        results = results.sort_values(["path_id", "year"]).reset_index(drop=True)
        results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    elapsed = int(time.perf_counter() - t_start)
    print(f"\nTotal runtime: {elapsed // 3600:02d}:"
          f"{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    run_historical_dispatch()
