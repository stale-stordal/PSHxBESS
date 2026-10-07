"""
Strip inflation from optimized dispatch results.

Reads:
    roa/input_files/PSH_optimized_results_combined.csv
    roa/input_files/BESS_optimized_results_combined.csv

Applies deflator (1 + 0.02)^year to all price/revenue columns,
leaving volume columns (bids, SoC) unchanged.

Output (same folder as this script):
    PSH_real.csv
    BESS_real.csv
"""

import os
import numpy as np
import pandas as pd

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
INPUT_DIR      = os.path.join(REPO_ROOT, "roa", "input_files")

INFLATION_RATE = 0.02
PRICE_COLS     = ["rev_DA", "rev_FCR", "rev_total", "avg_DA_price", "avg_FCR_price"]

FILES = [
    ("PSH_optimized_results_combined.csv",  "PSH_real.csv"),
    ("BESS_optimized_results_combined.csv", "BESS_real.csv"),
]

for src_name, dst_name in FILES:
    df       = pd.read_csv(os.path.join(INPUT_DIR, src_name))
    deflator = (1.0 + INFLATION_RATE) ** df["year"].values

    df_real = df.copy()
    for col in PRICE_COLS:
        if col in df_real.columns:
            df_real[col] = df_real[col] / deflator

    out_path = os.path.join(SCRIPT_DIR, dst_name)
    df_real.to_csv(out_path, index=False)

    yr_max  = df["year"].max()
    d_max   = (1.0 + INFLATION_RATE) ** yr_max
    print(f"  {dst_name}  ({len(df_real):,} rows)  "
          f"deflator range: 1.000 – {d_max:.4f}  (year 0 – {yr_max})")
