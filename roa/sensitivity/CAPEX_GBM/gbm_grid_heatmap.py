"""
BESS CAPEX GBM grid sensitivity — PSH Concept 1, full_GRsymmetric.

Runs a full mu × sigma grid (4 × 5 = 20 cells), caches to CSV, then plots
two annotated heatmaps side by side:

  Left  — V0_bess_only (MNOK): how the BESS standalone option value responds
           to the assumed cost-decline trajectory.
  Right — P(PSH | exercise): which technology is chosen across the parameter
           space.

Results are cached in results/gbm_grid_results.csv.
Delete that file to force a re-run.

Usage
-----
    cd masteroppgave
    python roa/sensitivity/CAPEX_GBM/gbm_grid_heatmap.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import roa.main as roa_main
from roa.config import Config

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CACHE_CSV       = OUTPUT_DIR / "gbm_grid_results.csv"
TARGET_SCENARIO = "full_GRsymmetric"

# ---------------------------------------------------------------------------
# Grid design
# ---------------------------------------------------------------------------

MU_VALUES    = [0.0, -0.022, -0.1, -0.2]          # rows  (top → bottom)
SIGMA_VALUES = [0.05, 0.15, 0.25, 0.35, 0.45]     # cols  (left → right)

MU_LABELS    = ["0%", "−2.2%", "−10%", "−20%"]
SIGMA_LABELS = ["0.05", "0.15", "0.25", "0.35", "0.45"]

CANONICAL_MU    = -0.022
CANONICAL_SIGMA =  0.25


# ---------------------------------------------------------------------------
# Sweep (skipped if cache exists)
# ---------------------------------------------------------------------------

def _run_cell(mu: float, sigma: float) -> dict:
    cfg = Config()
    cfg.psh_concept = 1
    cfg.gbm_mu      = mu
    cfg.gbm_sigma   = sigma
    results = roa_main.run_pipeline(cfg, seed=42, allow_nonproduction_shape=True)
    sr    = results["scenarios"][TARGET_SCENARIO]
    joint = sr["runs"]["joint"]
    bess  = sr["runs"]["bess_only"]
    return {
        "gbm_mu":          mu,
        "gbm_sigma":       sigma,
        "V0_joint_mnok":   joint["V0_stoptime"] / 1e6,
        "V0_bess_mnok":    bess["V0_stoptime"]  / 1e6,
        "exercise_rate":   joint["exercise_rate"],
        "p_psh_given_ex":  joint["p_psh_given_ex"],
        "p_bess_given_ex": joint["p_bess_given_ex"],
    }


if CACHE_CSV.exists():
    print(f"[grid] Cache found — loading {CACHE_CSV.name}")
    df = pd.read_csv(CACHE_CSV)
else:
    total  = len(MU_VALUES) * len(SIGMA_VALUES)
    done   = 0
    rows   = []
    for mu in MU_VALUES:
        for sigma in SIGMA_VALUES:
            done += 1
            print(f"  [{done:>2d}/{total}]  mu={mu:+.3f}  sigma={sigma:.2f} ...", end=" ", flush=True)
            rec = _run_cell(mu, sigma)
            rows.append(rec)
            print(f"V0_joint={rec['V0_joint_mnok']:.1f} MNOK  "
                  f"V0_bess={rec['V0_bess_mnok']:.1f} MNOK  "
                  f"p_psh={rec['p_psh_given_ex']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(CACHE_CSV, index=False)
    print(f"\n[grid] Saved cache -> {CACHE_CSV.name}\n")


# ---------------------------------------------------------------------------
# Build matrices
# ---------------------------------------------------------------------------

def _matrix(col: str) -> np.ndarray:
    mat = np.full((len(MU_VALUES), len(SIGMA_VALUES)), np.nan)
    for i, mu in enumerate(MU_VALUES):
        for j, sigma in enumerate(SIGMA_VALUES):
            row = df[(np.abs(df["gbm_mu"] - mu) < 1e-9) &
                     (np.abs(df["gbm_sigma"] - sigma) < 1e-9)]
            if not row.empty:
                mat[i, j] = row[col].values[0]
    return mat

mat_joint = _matrix("V0_joint_mnok")
mat_bess  = _matrix("V0_bess_mnok")
mat_psh   = _matrix("p_psh_given_ex")

# ---------------------------------------------------------------------------
# Thesis style — half figure
# ---------------------------------------------------------------------------

from thesis_style import COLORS, apply_style, figsize
apply_style("half")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

n_rows  = len(MU_VALUES)
n_cols  = len(SIGMA_VALUES)
x_edges = np.arange(n_cols + 1) - 0.5
y_edges = np.arange(n_rows + 1) - 0.5


def _heatmap(mat, cmap, vmin, vmax, fmt, cbar_label, cell_thresh=0.5):
    fig, ax = plt.subplots(figsize=figsize("half"))
    mesh = ax.pcolormesh(
        x_edges, y_edges, mat,
        cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="white", linewidth=1.5,
    )
    for i in range(n_rows):
        for j in range(n_cols):
            val = mat[i, j]
            if np.isnan(val):
                continue
            normed = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            txt_color = "white" if normed > cell_thresh else COLORS["text"]
            ax.text(j, i, fmt(val),
                    ha="center", va="center",
                    fontsize=mpl.rcParams["xtick.labelsize"] - 1,
                    color=txt_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(SIGMA_LABELS)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(MU_LABELS)
    ax.set_xlabel("GBM volatility $\\sigma$")
    ax.set_ylabel("GBM drift $\\mu$")
    ax.invert_yaxis()
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(labelsize=mpl.rcParams["xtick.labelsize"] - 1)

    plt.tight_layout()
    return fig


# --- Figure 1: V0_joint ---
cmap_joint = mcolors.LinearSegmentedColormap.from_list(
    "joint_v0", ["#EAF1F8", "#2F5C85"]
)
fig_joint = _heatmap(
    mat_joint,
    cmap=cmap_joint,
    vmin=mat_joint.min(),
    vmax=mat_joint.max(),
    fmt=lambda v: f"{v:.0f}",
    cbar_label="$V_0^{\\mathrm{joint}}$ (MNOK)",
    cell_thresh=0.55,
)
out_joint = OUTPUT_DIR / "gbm_heatmap_joint_v0.pdf"
fig_joint.savefig(out_joint, bbox_inches="tight")
plt.close(fig_joint)
print(f"Saved {out_joint}")

# --- Figure 2: V0_bess_only ---
cmap_bess = mcolors.LinearSegmentedColormap.from_list(
    "bess_v0", ["#F0EDF7", "#6F4E9C"]
)
fig_bess = _heatmap(
    mat_bess,
    cmap=cmap_bess,
    vmin=mat_bess.min(),
    vmax=mat_bess.max(),
    fmt=lambda v: f"{v:.0f}",
    cbar_label="$V_0^{\\mathrm{BESS}}$ standalone (MNOK)",
    cell_thresh=0.55,
)
out_bess = OUTPUT_DIR / "gbm_heatmap_bess_v0.pdf"
fig_bess.savefig(out_bess, bbox_inches="tight")
plt.close(fig_bess)
print(f"Saved {out_bess}")

# --- Figure 3: P(PSH | exercise) ---
cmap_psh = mcolors.LinearSegmentedColormap.from_list(
    "psh_share", ["#FFFFFF", "#4C78A8", "#6F4E9C", "#C44E52"],
)
fig_psh = _heatmap(
    mat_psh,
    cmap=cmap_psh,
    vmin=0, vmax=1,
    fmt=lambda v: f"{v:.2f}",
    cbar_label="P(PSH $\\mid$ exercise)",
    cell_thresh=0.15,
)
out_psh = OUTPUT_DIR / "gbm_heatmap_psh_share.pdf"
fig_psh.savefig(out_psh, bbox_inches="tight")
plt.close(fig_psh)
print(f"Saved {out_psh}")
