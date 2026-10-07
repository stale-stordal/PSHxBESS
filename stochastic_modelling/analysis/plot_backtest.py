"""
Backtest: compare distributional moments of historical hourly log-returns
against a sample of simulated hourly log-returns.

We sample a small number of paths from the simulation parquet so this script
stays light. For each market:

  - Histogram overlay (kernel density) of hourly log-returns
  - QQ plot
  - Moments table (mean, std%, skew, kurtosis, AR(1))

Outputs:
    figures/backtest_returns_<market>.pdf
    figures/backtest_qq_<market>.pdf
    data/backtest_moments.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from stochastic_modelling.analysis._constants import DATA_OUT, FIG_OUT, N_PATHS, load_sim_parquet
from thesis_style import COLORS, FIGSIZE, apply_style

apply_style("half")
os.makedirs(FIG_OUT, exist_ok=True)


def _returns_from_parquet(parquet_path, n_paths_sample, max_hours, rng):
    """Load n_paths_sample random columns from a single parquet and return
    finite hourly log-returns concatenated across those paths.
    """
    chosen = np.sort(rng.choice(N_PATHS, n_paths_sample, replace=False))
    col_names = [f"path_{i}" for i in chosen]
    print(f"    {os.path.basename(parquet_path)}: sampling {n_paths_sample} paths")
    t0 = time.time()
    df = pd.read_parquet(parquet_path, columns=col_names)
    if max_hours is not None:
        df = df.iloc[:max_hours]
    arr = df.to_numpy(dtype=np.float64, copy=False)
    print(f"    shape {arr.shape}, took {time.time() - t0:.1f}s")
    pos = arr > 0
    arr_safe = np.where(pos, arr, np.nan)
    log_arr = np.log(arr_safe)
    rets = np.diff(log_arr, axis=0).flatten()
    return rets[np.isfinite(rets)]


def sample_sim_returns(market, n_paths_sample, max_hours=None):
    """Read a small subset of paths from the simulation parquet(s) and return
    hourly log-returns concatenated across the sampled paths.

    In merged mode (STOCH_MERGED_<MARKET>_FILES set), n_paths_sample columns
    are drawn from each listed parquet so every pair is represented.
    """
    rng = np.random.default_rng(42)
    merged_env = os.environ.get(f"STOCH_MERGED_{market.upper()}_FILES", "").strip()
    if merged_env:
        input_dir = os.environ.get("STOCH_INPUT_DIR", "")
        files = [f.strip() for f in merged_env.split(",") if f.strip()]
        print(f"  reading {market} parquets (merged, {len(files)} files, "
              f"sampling {n_paths_sample} paths each)...")
        rets_list = [
            _returns_from_parquet(os.path.join(input_dir, fname),
                                  n_paths_sample, max_hours, rng)
            for fname in files
        ]
        return np.concatenate(rets_list)

    print(f"  reading {market} parquet (sampling {n_paths_sample} paths)...")
    t0 = time.time()
    chosen = np.sort(rng.choice(N_PATHS, n_paths_sample, replace=False))
    col_names = [f"path_{i}" for i in chosen]
    df = load_sim_parquet(market, columns=col_names)
    if max_hours is not None:
        df = df.iloc[:max_hours]
    arr = df.to_numpy(dtype=np.float64, copy=False)
    print(f"  shape {arr.shape}, took {time.time() - t0:.1f}s")
    pos = arr > 0
    arr_safe = np.where(pos, arr, np.nan)
    log_arr = np.log(arr_safe)
    rets = np.diff(log_arr, axis=0).flatten()
    return rets[np.isfinite(rets)]


def moments(rets):
    return {
        "n":         len(rets),
        "mean_pct":  rets.mean() * 100,
        "std_pct":   rets.std() * 100,
        "skew":      float(skew(rets)),
        "kurtosis":  float(kurtosis(rets)),
        "ar1":       float(np.corrcoef(rets[:-1], rets[1:])[0, 1]) if len(rets) > 2 else np.nan,
    }


def plot_distribution(ax, hist, sim, hist_color, sim_color):
    lo, hi = np.percentile(np.concatenate([hist, sim]), [0.5, 99.5])
    bins = np.linspace(lo, hi, 80)
    ax.hist(hist, bins=bins, density=True, color=hist_color, alpha=0.55,
            label="Historical", edgecolor="none")
    ax.hist(sim,  bins=bins, density=True, histtype="step", color=sim_color,
            linewidth=2.2, label="Simulated")
    ax.set_xlabel("Hourly log-return")
    ax.set_ylabel("Density")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")


def plot_qq(ax, hist, sim, color):
    n = min(len(hist), len(sim), 5000)
    qs = np.linspace(0.5 / n, 1 - 0.5 / n, n)
    hq = np.quantile(hist, qs)
    sq = np.quantile(sim, qs)
    ax.scatter(sq, hq, s=8, alpha=0.5, color=color, linewidths=0)
    lim = max(np.abs(hq).max(), np.abs(sq).max())
    ax.plot([-lim, lim], [-lim, lim], color=COLORS["neutral"],
            linewidth=1.0, linestyle="--")
    ax.set_xlabel("Simulated quantile")
    ax.set_ylabel("Historical quantile")
    ax.set_aspect("equal")
    ax.grid(axis="y")
    ax.set_axisbelow(True)


def main():
    rows = []
    for market, color in [
        ("spot", COLORS["main"]),
        ("fcr",  COLORS["secondary"]),
    ]:
        hist_path = os.path.join(DATA_OUT, f"historical_returns_{market}.npy")
        if not os.path.exists(hist_path):
            print(f"[skip] {hist_path} missing")
            continue
        hist_rets = np.load(hist_path)

        # Sample 30 paths from the first ~5.7 simulated years to keep it light.
        sim_rets = sample_sim_returns(market, n_paths_sample=30, max_hours=50_000)

        # Histogram
        fig, ax = plt.subplots(figsize=FIGSIZE["half"])
        plot_distribution(ax, hist_rets, sim_rets, COLORS["highlight"], color)
        plt.tight_layout()
        out = os.path.join(FIG_OUT, f"backtest_returns_{market}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

        # QQ
        fig, ax = plt.subplots(figsize=FIGSIZE["half"])
        plot_qq(ax, hist_rets, sim_rets, color)
        plt.tight_layout()
        out = os.path.join(FIG_OUT, f"backtest_qq_{market}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")

        h = moments(hist_rets); h["source"] = "historical"; h["market"] = market
        s = moments(sim_rets);  s["source"] = "simulated";  s["market"] = market
        rows.extend([h, s])

    out = os.path.join(DATA_OUT, "backtest_moments.csv")
    df = pd.DataFrame(rows)
    df = df[["market", "source", "n", "mean_pct", "std_pct", "skew", "kurtosis", "ar1"]]
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
