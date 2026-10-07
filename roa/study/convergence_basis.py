"""
Convergence and basis-selection study for the LSMC valuation.

Runs the canonical full_GRnone scenario at every (config, N, seed) triple
in the grid below, captures V_0 / R² / wallclock, and writes a CSV plus
two PNG figures. No methodology change — purely diagnostic.

Usage:
    python -m roa.study.convergence_basis [--quick]

Notes
-----
The bundled CSV has only N_data = 10000 paths. Rungs at N > 10000 use a
bootstrap with replacement on the 10000 base paths (cfg.bootstrap_upsample).
Bootstrap rows are flagged in the CSV (column `bootstrap`); R² and SE on
those rows are biased and should be read with the caveat printed in the
summary.
"""

from __future__ import annotations

import argparse
import math
import time
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roa.config import Config
from roa.investment import (
    compute_psh_payoffs,
    phase1_bess_compound,
    phase2_joint_optimization,
)
from roa.lsmc import (
    build_opex_matrix,
    load_optimizer_data,
    precompute_charged_energy_matrix,
    precompute_psh_da_revenue_matrix,
    precompute_revenue_matrix,
    simulate_bess_capex,
)


# ---------------------------------------------------------------------------
# Study design
# ---------------------------------------------------------------------------

CONFIGS = [
    {"name": "C1", "degree": 3, "vars": ("DA", "K"),         "M": 10},
    {"name": "C2", "degree": 2, "vars": ("DA", "FCR", "K"),  "M": 10},
    {"name": "C3", "degree": 3, "vars": ("DA", "FCR", "K"),  "M": 20},
    {"name": "C4", "degree": 4, "vars": ("DA", "K"),         "M": 15},
]
N_LADDER = (2000, 5000, 10000, 20000, 40000, 80000)
N_DATA = 10000
SEEDS_LARGE = list(range(42, 52))  # 10 seeds, default
SEEDS_TOP_N = list(range(42, 47))  # 5 seeds at N=64000 if wallclock binding


# ---------------------------------------------------------------------------
# Single-run harness
# ---------------------------------------------------------------------------

def _run_one(
    N: int,
    seed: int,
    basis_vars: tuple[str, ...],
    basis_degree: int,
    bootstrap: bool,
) -> dict:
    """One canonical full_GRnone evaluation. Returns a row dict."""
    cfg = Config()
    cfg.psh_concept = 1
    cfg.N = N
    cfg.bootstrap_upsample = bootstrap

    t_total = time.perf_counter()
    data = load_optimizer_data(cfg)            # populates cfg.N, cfg.T_max, cfg.T_def

    price_state = data.mean_DA
    fcr_state   = data.mean_FCR
    capex_paths = simulate_bess_capex(cfg, seed)

    bess_capacities = list(cfg.q_bess_grid)
    revenue_bess = {q: precompute_revenue_matrix(cfg, data, "BESS", q)
                    for q in bess_capacities}
    charge_bess  = {q: precompute_charged_energy_matrix(cfg, data, "BESS", q)
                    for q in bess_capacities}
    opex_bess    = {q: build_opex_matrix(cfg, charge_bess[q], "BESS", q)
                    for q in bess_capacities}

    revenue_psh    = precompute_revenue_matrix(cfg, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh     = precompute_charged_energy_matrix(cfg, data, "PSH", None)
    opex_psh = build_opex_matrix(
        cfg, charge_psh, "PSH", None,
        production_matrix=data.psh_bid_DA_sell,
    )

    p1 = phase1_bess_compound(
        cfg, revenue_bess, opex_bess, capex_paths, price_state,
        fcr_paths=fcr_state,
        basis_vars=basis_vars,
        basis_degree=basis_degree,
    )
    Pi_bess_full = p1["Pi_bess"]

    Pi_psh_full = compute_psh_payoffs(
        cfg, revenue_psh, price_state, opex_psh, revenue_psh_DA,
    )

    INF_PSH  = np.full((cfg.N, cfg.T_def + 1),  -np.inf, dtype=float)
    INF_BESS = np.full((cfg.N, cfg.T_max + 1), -np.inf, dtype=float)

    p2_joint = phase2_joint_optimization(
        cfg, Pi_bess_full, Pi_psh_full, price_state, capex_paths, fcr_state,
        basis_vars=basis_vars, basis_degree=basis_degree,
    )
    p2_b = phase2_joint_optimization(
        cfg, Pi_bess_full, INF_PSH, price_state, capex_paths, fcr_state,
        basis_vars=basis_vars, basis_degree=basis_degree,
    )
    p2_p = phase2_joint_optimization(
        cfg, INF_BESS, Pi_psh_full, price_state, capex_paths, fcr_state,
        basis_vars=basis_vars, basis_degree=basis_degree,
    )

    wallclock = time.perf_counter() - t_total

    p_bess = float(
        ((p2_joint["technology_choices"] == 1) & (p2_joint["exercise_times"] >= 0)).sum()
        / max((p2_joint["exercise_times"] >= 0).sum(), 1)
    )

    return {
        "M":                     math.comb(basis_degree + len(basis_vars), len(basis_vars)),
        "V_0_stoptime":          p2_joint["V0_stoptime"],
        "V_0_BESSonly":          p2_b["V0_stoptime"],
        "V_0_PSHonly":           p2_p["V0_stoptime"],
        "p_bess_given_exercise": p_bess,
        "phase1_r2":             p1.get("phase1_r2_t10"),
        "phase2_r2":             p2_joint.get("phase2_r2_t5"),
        "wallclock_s":           wallclock,
    }


# ---------------------------------------------------------------------------
# Sweep harness
# ---------------------------------------------------------------------------

def _build_grid(quick: bool) -> list[dict]:
    rows: list[dict] = []
    for cfg_spec in CONFIGS:
        for N in N_LADDER:
            # Skip C3, C4 below N=8000 per spec — regression too under-determined.
            if cfg_spec["name"] in ("C3", "C4") and N < 8000:
                continue
            seeds = SEEDS_LARGE
            if N >= 64000 and quick:
                seeds = SEEDS_TOP_N
            for seed in seeds:
                rows.append({
                    "config":        cfg_spec["name"],
                    "basis_degree":  cfg_spec["degree"],
                    "basis_vars":    ",".join(cfg_spec["vars"]),
                    "M":             cfg_spec["M"],
                    "N":             N,
                    "seed":          seed,
                    "bootstrap":     N > N_DATA,
                })
    return rows


def run_sweep(quick: bool, out_dir: Path) -> Path:
    grid = _build_grid(quick)
    print(f"[study] sweep grid: {len(grid)} runs total")
    print(f"[study] N_data = {N_DATA}; rows with N > N_data use bootstrap with replacement")

    rows: list[dict] = []
    grid_t0 = time.perf_counter()
    for k, row_spec in enumerate(grid, start=1):
        basis_vars = tuple(row_spec["basis_vars"].split(","))
        try:
            result = _run_one(
                N=row_spec["N"],
                seed=row_spec["seed"],
                basis_vars=basis_vars,
                basis_degree=row_spec["basis_degree"],
                bootstrap=row_spec["bootstrap"],
            )
        except Exception as exc:                             # pragma: no cover
            print(f"  [{k}/{len(grid)}] FAILED ({row_spec}): {exc!r}")
            continue
        merged = {**row_spec, **result}
        rows.append(merged)
        elapsed = time.perf_counter() - grid_t0
        print(
            f"  [{k:>3d}/{len(grid)}] {row_spec['config']} N={row_spec['N']:>6d} "
            f"seed={row_spec['seed']:<3d} V0={result['V_0_stoptime']/1e6:8.3f} MNOK  "
            f"R²(P1)={(result['phase1_r2'] or float('nan')):.3f}  "
            f"R²(P2)={(result['phase2_r2'] or float('nan')):.3f}  "
            f"wall={result['wallclock_s']:.2f}s  "
            f"(elapsed={elapsed:.0f}s)",
            flush=True,
        )

    df = pd.DataFrame(rows, columns=[
        "config", "basis_degree", "basis_vars", "M", "N", "seed", "bootstrap",
        "V_0_stoptime", "V_0_BESSonly", "V_0_PSHonly", "p_bess_given_exercise",
        "phase1_r2", "phase2_r2", "wallclock_s",
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"convergence_study_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[study] wrote {csv_path}")
    return csv_path


# ---------------------------------------------------------------------------
# Figures + summary
# ---------------------------------------------------------------------------

THESIS_COLORS = {
    "C1": "#4C78A8",   # hydro blue (anchor)
    "C2": "#72B7B2",   # teal
    "C3": "#C44E52",   # red highlight
    "C4": "#6F4E9C",   # purple
}


def _apply_thesis_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "axes.edgecolor": "#3A3A3A",
        "axes.labelcolor": "#3A3A3A",
        "xtick.color": "#3A3A3A",
        "ytick.color": "#3A3A3A",
        "text.color": "#3A3A3A",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "legend.frameon": False,
    })


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["config", "N"], as_index=False).agg(
        V0_mean=("V_0_stoptime", "mean"),
        V0_std=("V_0_stoptime", "std"),
        V0_n=("V_0_stoptime", "size"),
        r2p2_mean=("phase2_r2", "mean"),
        r2p2_std=("phase2_r2", "std"),
        r2p1_mean=("phase1_r2", "mean"),
        r2p1_std=("phase1_r2", "std"),
        wall_mean=("wallclock_s", "mean"),
        bootstrap=("bootstrap", "first"),
    )
    g["V0_se"]    = g["V0_std"] / np.sqrt(g["V0_n"])
    g["V0_cv"]    = g["V0_std"] / g["V0_mean"]
    g["r2p2_se"]  = g["r2p2_std"] / np.sqrt(g["V0_n"])
    g["r2p1_se"]  = g["r2p1_std"] / np.sqrt(g["V0_n"])
    return g


def _plot_one_config(
    ax,
    sub: pd.DataFrame,
    y_col_mean: str,
    y_col_se: str,
    color: str,
    label: str,
    y_scale: float = 1.0,
) -> None:
    """Plot a single config: connect with one line, then overlay markers
    that are filled where bootstrap=False and open where bootstrap=True.
    """
    sub = sub.sort_values("N")
    ax.errorbar(
        sub["N"], sub[y_col_mean] / y_scale,
        yerr=sub[y_col_se] / y_scale,
        color=color, lw=2.0, marker="None",
        capsize=3, label=label,
    )
    # Solid markers for legitimate subsamples (N <= N_data)
    solid = sub[~sub["bootstrap"]]
    if not solid.empty:
        ax.plot(solid["N"], solid[y_col_mean] / y_scale,
                "o", color=color, markerfacecolor=color,
                markeredgecolor=color, markersize=6)
    # Open markers for bootstrap rows (caveat noted)
    boot = sub[sub["bootstrap"]]
    if not boot.empty:
        ax.plot(boot["N"], boot[y_col_mean] / y_scale,
                "o", markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=1.5, markersize=6)


def figure_v0_vs_n(agg: pd.DataFrame, out_path: Path, prod_N: int | None) -> None:
    _apply_thesis_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for cfg_name in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["config"] == cfg_name]
        if sub.empty:
            continue
        _plot_one_config(
            ax, sub, "V0_mean", "V0_se",
            color=THESIS_COLORS[cfg_name],
            label=f"{cfg_name} (M={_config_M(cfg_name)})",
            y_scale=1e6,
        )
    ax.set_xscale("log")
    ax.set_xticks(list(N_LADDER))
    ax.set_xticklabels([f"{n:,}" for n in N_LADDER])
    ax.minorticks_off()
    ax.set_xlabel("Monte Carlo paths $N$ (log scale)")
    ax.set_ylabel("$V_0$ (MNOK)")
    if prod_N is not None:
        ax.axvline(prod_N, color="#4A4A4A", ls="--", lw=0.8, alpha=0.6)
        ax.text(prod_N, ax.get_ylim()[1], f"  production $N$ = {prod_N:,}",
                va="top", ha="left", fontsize=10, color="#4A4A4A")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[study] wrote {out_path}")


def figure_r2_vs_n(agg: pd.DataFrame, out_path: Path, prod_N: int | None) -> None:
    _apply_thesis_style()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for cfg_name in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["config"] == cfg_name]
        if sub.empty:
            continue
        _plot_one_config(
            ax, sub, "r2p2_mean", "r2p2_se",
            color=THESIS_COLORS[cfg_name],
            label=f"{cfg_name} (M={_config_M(cfg_name)})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo paths $N$ (log scale)")
    ax.set_ylabel("Phase 2 in-sample $R^{2}$ at $t=5$")
    if prod_N is not None:
        ax.axvline(prod_N, color="#4A4A4A", ls="--", lw=0.8, alpha=0.6)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[study] wrote {out_path}")


def _config_M(name: str) -> int:
    for c in CONFIGS:
        if c["name"] == name:
            return c["M"]
    raise KeyError(name)


def summarise(agg: pd.DataFrame) -> pd.DataFrame:
    """Stabilisation: smallest N where mean V0 changes < 0.5% between consecutive
    N steps AND across-seed CV < 1%. Production N is one ladder step above the floor.
    """
    out_rows: list[dict] = []
    for cfg_name in ["C1", "C2", "C3", "C4"]:
        sub = agg[agg["config"] == cfg_name].sort_values("N").reset_index(drop=True)
        if sub.empty:
            continue
        floor_N = None
        for k in range(1, len(sub)):
            rel_change = abs(sub["V0_mean"].iloc[k] - sub["V0_mean"].iloc[k-1]) / abs(sub["V0_mean"].iloc[k-1])
            cv         = sub["V0_cv"].iloc[k]
            if rel_change < 0.005 and cv < 0.01:
                floor_N = sub["N"].iloc[k]
                break
        prod_N = (
            int(sub["N"].iloc[min(sub.index[sub["N"] == floor_N][0] + 1, len(sub) - 1)])
            if floor_N is not None else None
        )
        if prod_N is None:
            prod_N = int(sub["N"].iloc[-1])  # fall back to highest N reached
        prod_row = sub[sub["N"] == prod_N].iloc[0]
        out_rows.append({
            "config":           cfg_name,
            "M":                _config_M(cfg_name),
            "N_floor":          int(floor_N) if floor_N is not None else None,
            "production_N":     int(prod_N),
            "V0_at_prod_MNOK":  prod_row["V0_mean"] / 1e6,
            "SE_at_prod_MNOK":  prod_row["V0_se"] / 1e6,
            "CV_at_prod":       float(prod_row["V0_cv"]),
            "wall_at_prod_s":   float(prod_row["wall_mean"]),
            "r2p2_at_prod":     float(prod_row["r2p2_mean"]),
            "bootstrap_at_prod": bool(prod_row["bootstrap"]),
        })
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--quick", action="store_true",
                   help="At N=64000, drop K from 10 seeds to 5.")
    args = p.parse_args(argv)

    out_dir = Path(__file__).resolve().parent.parent / "output_files"
    csv_path = run_sweep(quick=args.quick, out_dir=out_dir)

    df = pd.read_csv(csv_path)
    agg = _aggregate(df)
    summary = summarise(agg)

    fig1 = csv_path.with_name(csv_path.stem + "_fig1_V0_vs_N.png")
    fig1_pdf = csv_path.with_name(csv_path.stem + "_fig1_V0_vs_N.pdf")
    fig2 = csv_path.with_name(csv_path.stem + "_fig2_R2_vs_N.png")
    # Production N is the dispatch-data ceiling (10,000 independent paths);
    # rungs above this are bootstrap-upsampled and only used for convergence
    # diagnostics, so the production marker pins at N_DATA.
    prod_N = N_DATA
    figure_v0_vs_n(agg, fig1, prod_N)
    figure_v0_vs_n(agg, fig1_pdf, prod_N)
    figure_r2_vs_n(agg, fig2, prod_N)

    print()
    print("=" * 88)
    print("  Summary")
    print("=" * 88)
    with pd.option_context("display.float_format", "{:,.4f}".format):
        print(summary.to_string(index=False))
    print()
    print(f"  CSV     : {csv_path}")
    print(f"  Figure 1: {fig1}")
    print(f"  Figure 1 (PDF): {fig1_pdf}")
    print(f"  Figure 2: {fig2}")


if __name__ == "__main__":
    main()
