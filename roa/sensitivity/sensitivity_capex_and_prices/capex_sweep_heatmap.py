"""
CAPEX sweep heatmap — PSH Concept 1 price-modelling sensitivity.

Runs the ROA pipeline for every (price scenario, PSH CAPEX) combination
and plots P(PSH | exercise) as an annotated heatmap.

Results cached to capex_sweep_results.csv — delete to re-run.

Usage
-----
    cd masteroppgave
    python roa/sensitivity/sensitivity_capex_and_prices/capex_sweep_heatmap.py
"""

import functools
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

SCRIPT_DIR   = Path(__file__).parent
REPO_ROOT    = SCRIPT_DIR.parent.parent.parent
SENS_CSV_DIR = REPO_ROOT / "optimization" / "results" / "paths"
sys.path.insert(0, str(REPO_ROOT))

import roa.main as roa_main
from roa.config import Config, PshConcept
from thesis_style import COLORS, apply_style

# ---------------------------------------------------------------------------
# Sweep design
# ---------------------------------------------------------------------------

CAPEX_VALUES_NOK = [
    1_500_000_000,
    1_830_000_000,
    2_000_000_000,
    2_500_000_000,
    3_000_000_000,
    3_500_000_000,
    4_000_000_000,
    4_500_000_000,
    5_000_000_000,
]

BASELINE_CAPEX_NOK = 1_830_000_000

TARGET_SCENARIO = "full_GRsymmetric"

# (row label, group, bess_stem, psh_stem) — seed tiles, grouped as in the tornado plot.
SCENARIOS = [
    ("Baseline",             "Baseline",
        "Battery_optimized_results_seed_baseline.csv",
        "Pump_optimized_results_seed_baseline.csv"),
    ("GOU vol = 0.05",       "GOU volatility",
        "Battery_optimized_results_seed_price_drift_0.05.csv",
        "Pump_optimized_results_seed_price_drift_0.05.csv"),
    ("GOU vol = 0.50",       "GOU volatility",
        "Battery_optimized_results_seed_price_drift_0.50.csv",
        "Pump_optimized_results_seed_price_drift_0.50.csv"),
    ("GOU mean reversion = 0.00", "GOU mean reversion",
        "Battery_optimized_results_seed_kappa_0.00.csv",
        "Pump_optimized_results_seed_kappa_0.00.csv"),
    ("GOU mean reversion = 0.20", "GOU mean reversion",
        "Battery_optimized_results_seed_kappa_0.20.csv",
        "Pump_optimized_results_seed_kappa_0.20.csv"),
    ("Spike decay 0.05",     "Spike decay",
        "Battery_optimized_results_seed_spike_decay_0.05.csv",
        "Pump_optimized_results_seed_spike_decay_0.05.csv"),
    ("Spike decay 0.30",     "Spike decay",
        "Battery_optimized_results_seed_spike_decay_0.30.csv",
        "Pump_optimized_results_seed_spike_decay_0.30.csv"),
    ("Spike floor 0.05",     "Spike floor",
        "Battery_optimized_results_seed_spike_floor_0.05.csv",
        "Pump_optimized_results_seed_spike_floor_0.05.csv"),
    ("Spike floor 0.50",     "Spike floor",
        "Battery_optimized_results_seed_spike_floor_0.50.csv",
        "Pump_optimized_results_seed_spike_floor_0.50.csv"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(capex_nok: int) -> Config:
    cfg = Config()
    cfg.psh_concept = 1
    per_mw = capex_nok / 100
    cfg.psh_concepts[0] = PshConcept(
        capex_bygg_nok_per_mw=round(per_mw * 10_800_000 / 18_300_000),
        capex_elek_nok_per_mw=round(per_mw *  5_400_000 / 18_300_000),
        capex_andre_nok_per_mw=round(per_mw * 2_100_000 / 18_300_000),
        psh_mw=100,
    )
    return cfg


def _run(bess_src: Path, psh_src: Path, capex_nok: int) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(bess_src, tmp_path / "BESS_optimized_results_combined.csv")
        shutil.copy(psh_src,  tmp_path / "PSH_optimized_results_combined.csv")
        _orig_dir  = roa_main._INPUT_DIR
        _orig_load = roa_main.load_optimizer_data
        roa_main._INPUT_DIR          = tmp_path
        roa_main.load_optimizer_data = functools.partial(_orig_load, data_dir=tmp_path)
        try:
            cfg     = _make_cfg(capex_nok)
            results = roa_main.run_pipeline(cfg, seed=42, allow_nonproduction_shape=True)
        finally:
            roa_main._INPUT_DIR          = _orig_dir
            roa_main.load_optimizer_data = _orig_load
    return results["scenarios"][TARGET_SCENARIO]["runs"]["joint"]


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

out_csv = SCRIPT_DIR / "capex_sweep_results.csv"

# Filter to scenarios whose files actually exist
active_scenarios = []
for row in SCENARIOS:
    label, group, bess_name, psh_name = row
    if (SENS_CSV_DIR / bess_name).exists() and (SENS_CSV_DIR / psh_name).exists():
        active_scenarios.append(row)
    else:
        print(f"  WARNING: {bess_name} not found — skipped.")

if out_csv.exists():
    print(f"[sweep] CSV found — skipping sweep. Delete {out_csv.name} to re-run.")
else:
    n_rows = len(active_scenarios)
    n_cols = len(CAPEX_VALUES_NOK)
    p_psh_matrix = np.full((n_rows, n_cols), np.nan)
    v0_matrix    = np.full((n_rows, n_cols), np.nan)
    total = n_rows * n_cols
    done  = 0

    for i, (label, group, bess_name, psh_name) in enumerate(active_scenarios):
        for j, capex in enumerate(CAPEX_VALUES_NOK):
            done += 1
            print(f"[{done:>3d}/{total}] {label:22s}  CAPEX={capex/1e9:.2f} BNOK ...",
                  end=" ", flush=True)
            joint = _run(SENS_CSV_DIR / bess_name, SENS_CSV_DIR / psh_name, capex)
            p_psh_matrix[i, j] = joint["p_psh_given_ex"]
            v0_matrix[i, j]    = joint["V0_stoptime"] / 1e6
            print(f"p_psh={p_psh_matrix[i,j]:.3f}  V0={v0_matrix[i,j]:.1f} MNOK")

    rows = []
    for i, (label, _, _, _) in enumerate(active_scenarios):
        for j, capex in enumerate(CAPEX_VALUES_NOK):
            rows.append({
                "scenario":       label,
                "capex_bnok":     capex / 1e9,
                "p_psh_given_ex": p_psh_matrix[i, j],
                "v0_mnok":        v0_matrix[i, j],
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved raw results -> {out_csv.name}")


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

apply_style()

df_plot    = pd.read_csv(out_csv)
df_plot    = df_plot[(df_plot["capex_bnok"] >= 2.5) & (df_plot["capex_bnok"] <= 4.5)]
capex_cols = sorted(df_plot["capex_bnok"].unique())
col_labels = [f"{c:.2g}" for c in capex_cols]
row_labels = [s[0] for s in active_scenarios]
n_cols     = len(capex_cols)
n_data     = len(row_labels)

data_mat = np.array([
    [df_plot[(df_plot["scenario"] == lbl) & (df_plot["capex_bnok"] == c)]["p_psh_given_ex"].values[0]
     for c in capex_cols]
    for lbl in row_labels
])

# Insert a visual gap row between consecutive scenario groups.
GAP     = 0.35
NAN_ROW = np.full((1, n_cols), np.nan)
groups  = [s[1] for s in active_scenarios]

blocks           = []   # rows of the padded matrix (data rows + gap rows)
row_heights      = []   # height of each padded row
data_row_indices = []   # padded-row index of each real data row
prev_group = None
for i in range(n_data):
    if prev_group is not None and groups[i] != prev_group:
        blocks.append(NAN_ROW)
        row_heights.append(GAP)
    data_row_indices.append(len(blocks))
    blocks.append(data_mat[i:i + 1])
    row_heights.append(1)
    prev_group = groups[i]

mat_gaps = np.vstack(blocks)
y_edges  = np.concatenate([[0], np.cumsum(row_heights)])
centers  = [(y_edges[k] + y_edges[k + 1]) / 2 for k in data_row_indices]

fig, ax = plt.subplots(figsize=(9, 7.0))

cmap = mcolors.LinearSegmentedColormap.from_list(
    "psh_share", ["#FFFFFF", "#4C78A8", "#6F4E9C", "#C44E52"],
)
cmap.set_bad("white")

x_edges = np.arange(n_cols + 1) - 0.5
mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(mat_gaps),
                     cmap=cmap, vmin=0, vmax=1,
                     edgecolors="white", linewidth=1.2)

for idx, (data_row, center_y) in enumerate(zip(data_row_indices, centers)):
    for j in range(n_cols):
        val = mat_gaps[data_row, j]
        txt_color = "white" if val > 0.15 else COLORS["text"]
        ax.text(j, center_y, f"{val:.2f}", ha="center", va="center",
                fontsize=10, color=txt_color)

ax.set_xticks(range(n_cols))
ax.set_xticklabels(col_labels)
ax.set_xlabel("PSH CAPEX (BNOK)")
ax.set_yticks(centers)
ax.set_yticklabels(row_labels)
ax.set_ylim(y_edges[-1], y_edges[0])

cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("P(PSH $\\mid$ exercise)")
cbar.ax.tick_params(labelsize=11)
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
out_pdf = SCRIPT_DIR / "capex_sweep_heatmap.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()
print(f"Saved heatmap -> {out_pdf.name}")
