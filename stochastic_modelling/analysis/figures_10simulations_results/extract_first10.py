"""
Extract the first N paths from a full simulation parquet into a small CSV.

The plot scripts in this folder (plot_price_paths*.py) read
    spot_1_first10.csv / FCR_1_first10.csv
which are just the first 10 columns of the big spot_1.parquet / FCR_1.parquet.
This script regenerates those slices so they are reproducible.

Usage (from anywhere):
    python extract_first10.py                 # 10 paths from spot_1 / FCR_1
    python extract_first10.py --n 10          # explicit path count
    python extract_first10.py --spot spot_2.parquet --fcr FCR_2.parquet

Input directory resolution (first match wins):
    --input-dir <dir>  ->  STOCH_INPUT_DIR env var  ->  external storage default
"""
import argparse
import os

import pandas as pd

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DIR = "../../../../../storage/users/einami/input_files"


def extract(in_path: str, out_path: str, n: int) -> None:
    cols = [f"path_{i}" for i in range(n)]
    df = pd.read_parquet(in_path, columns=cols)
    df.index.name = df.index.name or "timestamp"
    df.to_csv(out_path)
    print(f"  {os.path.basename(in_path)} -> {os.path.basename(out_path)}  "
          f"({df.shape[0]} rows x {df.shape[1]} paths)")


def main() -> None:
    p = argparse.ArgumentParser(description="Extract first N paths from sim parquets.")
    p.add_argument("--n", type=int, default=10, help="number of paths (default 10)")
    p.add_argument("--spot", default="spot_1.parquet", help="spot parquet filename")
    p.add_argument("--fcr", default="FCR_1.parquet", help="FCR parquet filename")
    p.add_argument("--input-dir", default=None,
                   help="dir holding the parquets (default: $STOCH_INPUT_DIR or external storage)")
    args = p.parse_args()

    input_dir = args.input_dir or os.environ.get("STOCH_INPUT_DIR", EXTERNAL_DIR)
    print(f"Input dir: {input_dir}")

    for fname in (args.spot, args.fcr):
        base = os.path.splitext(fname)[0]
        in_path  = os.path.join(input_dir, fname)
        out_path = os.path.join(SCRIPT_DIR, f"{base}_first{args.n}.csv")
        extract(in_path, out_path, args.n)

    print("Done.")


if __name__ == "__main__":
    main()
