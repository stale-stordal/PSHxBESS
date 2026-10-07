"""
Per-year moments of simulated yearly mean spot and FCR prices.

Scenario-aware like the plot scripts: reads the per-(path, year) metrics from
DATA_OUT and writes the table to FIG_OUT, both of which respect STOCH_SCENARIO
(set by run_all.py). Run standalone it defaults to the merged 10k scenario.

Reads:
    <DATA_OUT>/sim_metrics_spot.npz
    <DATA_OUT>/sim_metrics_fcr.npz

Outputs:
    <FIG_OUT>/yearly_moments_table.csv
    <FIG_OUT>/yearly_moments_table.tex
"""
import os
import sys

import numpy as np
import pandas as pd

# Standalone default: with no scenario chosen, target the merged outputs.
# run_all.py sets STOCH_SCENARIO explicitly, so setdefault leaves it untouched.
os.environ.setdefault("STOCH_SCENARIO", "merged")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stochastic_modelling.analysis._constants import DATA_OUT, FIG_OUT, UNIT_SPOT, UNIT_FCR  # noqa: E402

TARGET_YEARS = [2030, 2035, 2040, 2050, 2075]


def yearly_moments(market: str, years_wanted):
    npz_path = os.path.join(DATA_OUT, f"sim_metrics_{market}.npz")
    with np.load(npz_path) as d:
        ym = d["yearly_mean"]
        start_year = int(d["start_year"])
        n_paths = int(d["n_paths"]) if "n_paths" in d else ym.shape[0]
    years = np.arange(start_year, start_year + ym.shape[1])

    rows = {}
    for y in years_wanted:
        idx = np.where(years == y)[0]
        if len(idx) == 0:
            raise ValueError(f"Year {y} not in simulated range {years[0]}-{years[-1]}")
        col = ym[:, idx[0]]
        rows[y] = {
            "mean": float(np.nanmean(col)),
            "std":  float(np.nanstd(col, ddof=1)),
            "p05":  float(np.nanpercentile(col, 5)),
            "p95":  float(np.nanpercentile(col, 95)),
        }
    return pd.DataFrame.from_dict(rows, orient="index"), n_paths


def build_tex(df: pd.DataFrame, n_paths: int) -> str:
    def f(x: float) -> str:
        return f"{x:,.1f}".replace(",", r"\,")

    n_paths_str = f"{n_paths:,}".replace(",", r"\,")

    body_rows = []
    for y in df.index:
        row = df.loc[y]
        body_rows.append(
            f"  {y} & {f(row['spot_mean'])} & {f(row['fcr_mean'])} "
            f"& {f(row['spot_std'])} & {f(row['fcr_std'])} "
            f"& {f(row['spot_p05'])} & {f(row['fcr_p05'])} "
            f"& {f(row['spot_p95'])} & {f(row['fcr_p95'])} \\\\"
        )
    body = "\n".join(body_rows)

    return rf"""\begin{{table}}[ht]
\centering
\caption{{Per-year moments of simulated yearly mean prices across {n_paths_str} paths. Spot in {UNIT_SPOT}, FCR in {UNIT_FCR}.}}
\label{{tab:yearly-moments}}
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
 & \multicolumn{{2}}{{c}}{{Mean}} & \multicolumn{{2}}{{c}}{{Std. dev.}} & \multicolumn{{2}}{{c}}{{P5}} & \multicolumn{{2}}{{c}}{{P95}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(lr){{8-9}}
Year & Spot & FCR & Spot & FCR & Spot & FCR & Spot & FCR \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main():
    os.makedirs(FIG_OUT, exist_ok=True)

    spot_df, n_paths = yearly_moments("spot", TARGET_YEARS)
    fcr_df, _        = yearly_moments("fcr",  TARGET_YEARS)
    spot = spot_df.add_prefix("spot_")
    fcr  = fcr_df.add_prefix("fcr_")
    df = pd.concat([spot, fcr], axis=1)
    df.index.name = "year"

    ordered = [
        "spot_mean", "fcr_mean",
        "spot_std",  "fcr_std",
        "spot_p05",  "fcr_p05",
        "spot_p95",  "fcr_p95",
    ]
    df = df[ordered]

    csv_path = os.path.join(FIG_OUT, "yearly_moments_table.csv")
    tex_path = os.path.join(FIG_OUT, "yearly_moments_table.tex")
    df.round(2).to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(build_tex(df, n_paths))

    print(f"Saved {csv_path}")
    print(f"Saved {tex_path}")
    print(df.round(1).to_string())


if __name__ == "__main__":
    main()
