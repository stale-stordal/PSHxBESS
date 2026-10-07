import pandas as pd
import os

prefix = input("Prefix on file: ")

unjoint_dir = os.path.join(os.path.dirname(__file__), "unjoint")
output_path = os.path.join(os.path.dirname(__file__), f"{prefix}_optimized_results_combined.csv")

files = sorted(f for f in os.listdir(unjoint_dir) if f.endswith(".csv"))

dfs = []
for i, fname in enumerate(files):
    df = pd.read_csv(os.path.join(unjoint_dir, fname))
    df["path_id"] = df["path_id"].str.extract(r"(\d+)").astype(int) + i * 2000
    df["path_id"] = "path_" + df["path_id"].astype(str)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(output_path, index=False)
print(f"Saved {combined['path_id'].nunique()} unique path_ids to {output_path}")
