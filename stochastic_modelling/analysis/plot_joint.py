"""
Joint Spot-FCR distribution: scatter of yearly mean spot vs FCR, coloured by
decade. Also reports the cross-correlation in simulation vs historical.

Reads:
    data/sim_metrics_spot.npz
    data/sim_metrics_fcr.npz
    data/historical_metrics.csv

Outputs:
    figures/joint_yearly_mean.pdf
    data/correlation_summary.txt
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stochastic_modelling.analysis._constants import DATA_OUT, FIG_OUT, UNIT_SPOT, UNIT_FCR
from thesis_style import COLORS, FIGSIZE, apply_style

apply_style("half")
os.makedirs(FIG_OUT, exist_ok=True)


def load_yearly_long() -> pd.DataFrame:
    """Combine sim_metrics_<market>.npz into a long DataFrame with one row per
    (path_number, year_number) containing yearly mean spot and FCR prices.
    Years are aligned on the intersection of the two markets.
    """
    frames = {}
    for market in ("spot", "fcr"):
        with np.load(os.path.join(DATA_OUT, f"sim_metrics_{market}.npz")) as d:
            ym = d["yearly_mean"]
            start_year = int(d["start_year"])
        years = np.arange(start_year, start_year + ym.shape[1])
        df = pd.DataFrame(ym, columns=years)
        df.index.name = "path_number"
        long = df.reset_index().melt(
            id_vars="path_number", var_name="year_number",
            value_name=f"{market}_yearly_mean",
        )
        long["year_number"] = long["year_number"].astype(int)
        frames[market] = long

    merged = frames["spot"].merge(
        frames["fcr"], on=["path_number", "year_number"], how="inner",
    )
    merged = merged.rename(columns={
        "spot_yearly_mean": "spot",
        "fcr_yearly_mean":  "fcr",
    })
    merged = merged[merged["year_number"] <= 2075]
    merged["decade"] = (merged["year_number"] // 10) * 10
    return merged


def main():
    df = load_yearly_long()

    decade_palette = {
        2020: COLORS["highlight"],
        2030: COLORS["main"],
        2040: COLORS["secondary"],
        2050: "#3FA34D",
        2060: "#D9954B",
        2070: "#7C7C7C",
        2080: "#C8A2C8",
        2090: "#4A4A4A",
    }

    # Collapse to decade-mean per path so the scatter cloud stays legible.
    pdec = df.groupby(["path_number", "decade"]).agg(
        spot=("spot", "mean"),
        fcr =("fcr",  "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE["half"])
    for dec, sub in pdec.groupby("decade"):
        ax.scatter(sub["spot"], sub["fcr"],
                   s=6, alpha=0.35,
                   color=decade_palette.get(int(dec), "#999999"),
                   linewidths=0)

    handles = [Line2D([0], [0], marker="o", linestyle="",
                       color=decade_palette[int(d)], label=f"{int(d)}s",
                       markersize=7, alpha=0.9)
               for d in sorted(pdec["decade"].unique()) if int(d) in decade_palette]
    ax.legend(handles=handles, loc="upper left", ncol=2,
              title="Decade", title_fontsize=12)

    ax.set_xlabel(f"Decade-mean spot price ({UNIT_SPOT})")
    ax.set_ylabel(f"Decade-mean FCR price ({UNIT_FCR})")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    plt.tight_layout()
    out = os.path.join(FIG_OUT, "joint_yearly_mean.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # Pearson correlations
    rho_sim_all = np.corrcoef(df["spot"], df["fcr"])[0, 1]
    rho_sim_dec = np.corrcoef(pdec["spot"], pdec["fcr"])[0, 1]

    hist = pd.read_csv(os.path.join(DATA_OUT, "historical_metrics.csv"))
    hist = hist[hist["n_hours"] >= 8000]
    hs = hist[hist["market"] == "spot"][["year", "yearly_mean"]].rename(
        columns={"yearly_mean": "spot"})
    hf = hist[hist["market"] == "fcr"][["year", "yearly_mean"]].rename(
        columns={"yearly_mean": "fcr"})
    h = hs.merge(hf, on="year")
    rho_hist = np.corrcoef(h["spot"], h["fcr"])[0, 1] if len(h) >= 3 else float("nan")

    n_paths = df["path_number"].nunique()
    n_years = df["year_number"].nunique()
    out_txt = os.path.join(DATA_OUT, "correlation_summary.txt")
    with open(out_txt, "w") as f:
        f.write("Spot-FCR Pearson correlation of yearly mean prices\n")
        f.write("==================================================\n\n")
        f.write(f"Simulated (all {n_paths} paths x {n_years} years "
                f"= {len(df):,} obs): rho = {rho_sim_all:+.4f}\n")
        f.write(f"Simulated (path x decade means):                 "
                f"rho = {rho_sim_dec:+.4f}\n")
        f.write(f"Historical (overlapping years "
                f"{int(h['year'].min())}–{int(h['year'].max())}, "
                f"n={len(h)}):  rho = {rho_hist:+.4f}\n")
    print(open(out_txt).read())


if __name__ == "__main__":
    main()
