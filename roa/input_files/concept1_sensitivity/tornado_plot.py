"""
Tornado plot — ROA sensitivity analysis for PSH Concept 1.

Reads V0 (joint, full_GRnone) from the JSON files produced by
run_roa_sensitivity.py and draws a horizontal bar chart sorted
by sensitivity magnitude. Baseline from BESS/PSH_optimized_results_concept1.csv run.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from thesis_style import COLORS, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR   = SCRIPT_DIR / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"

# Map JSON stem -> (parameter label, value label shown on bar end)
SCENARIO_MAP = {
    "spike_decay_0.05":  ("Spike decay rate",     "0.05"),
    "spike_decay_0.30":  ("Spike decay rate",     "0.30"),
    "spike_floor_0.05":  ("Spike floor fraction", "0.05"),
    "spike_floor_0.50":  ("Spike floor fraction", "0.50"),
    "price_drift_0.00":  ("GOU volatility",       "0.00"),
    "price_drift_0.75":  ("GOU volatility",       "0.75"),
}

# Low / high value label for each parameter (determines bar direction)
PARAM_ENDS = {
    "Spike decay rate":     ("0.05", "0.30"),
    "Spike floor fraction": ("0.05", "0.50"),
    "GOU volatility":       ("0.00", "0.75"),
}

# ---------------------------------------------------------------------------
# Load V0 values
# ---------------------------------------------------------------------------
def _v0_mnok(path: Path) -> float:
    with open(path) as f:
        d = json.load(f)
    return d["scenarios"]["full_GRnone"]["runs"]["joint"]["V0_stoptime"] / 1e6

raw: dict[str, dict[str, float]] = {}
for stem, (param, val_label) in SCENARIO_MAP.items():
    p = RESULTS_DIR / f"{stem}.json"
    if not p.exists():
        print(f"  WARNING: {p.name} not found — skipped.")
        continue
    raw.setdefault(param, {})[val_label] = _v0_mnok(p)

baseline: float | None = None
if BASELINE_PATH is not None and Path(BASELINE_PATH).exists():
    baseline = _v0_mnok(Path(BASELINE_PATH))
    print(f"Baseline V0 = {baseline:.2f} MNOK")
else:
    print("  WARNING: baseline JSON not found — no reference line.")

# ---------------------------------------------------------------------------
# Build per-parameter (low_v0, high_v0) pairs and sort by range
# ---------------------------------------------------------------------------
data = {}
for param, (lbl_low, lbl_high) in PARAM_ENDS.items():
    vals = raw.get(param, {})
    if lbl_low not in vals or lbl_high not in vals:
        continue
    data[param] = {
        "low_v0":     vals[lbl_low],
        "high_v0":    vals[lbl_high],
        "low_label":  lbl_low,
        "high_label": lbl_high,
    }

params = sorted(
    data.keys(),
    key=lambda p: abs(data[p]["high_v0"] - data[p]["low_v0"]),
    reverse=True,
)

y = np.arange(len(params))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, max(3.5, len(params) * 1.2)))

BAR_H = 0.45

for i, p in enumerate(params):
    low_v  = data[p]["low_v0"]
    high_v = data[p]["high_v0"]
    xmin   = min(low_v, high_v)
    xmax   = max(low_v, high_v)

    ax.barh(i, xmax - xmin, left=xmin, height=BAR_H,
            color=COLORS["main"], alpha=0.75, zorder=2)

    # Place each parameter value label at the end of the bar it actually produced.
    left_label  = data[p]["low_label"]  if low_v <= high_v else data[p]["high_label"]
    right_label = data[p]["high_label"] if low_v <= high_v else data[p]["low_label"]

    pad = 0.15
    ax.text(xmin - pad, i, left_label,  ha="right", va="center", fontsize=11)
    ax.text(xmax + pad, i, right_label, ha="left",  va="center", fontsize=11)

    if baseline is not None:
        ax.plot(baseline, i, marker="D", color=COLORS["neutral"],
                markersize=6, zorder=4,
                label="Base case" if i == 0 else "")

if baseline is not None:
    ax.axvline(baseline, color=COLORS["neutral"], lw=1.0,
               linestyle="--", alpha=0.7, zorder=1)
    ax.legend(loc="lower right", frameon=False)

ax.set_yticks(y)
ax.set_yticklabels(params)
ax.invert_yaxis()
ax.set_xlabel("Real option value $V_0$ (M NOK)")
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", pad=10)

# Small padding to the left of the lowest bar value
all_low = min(min(d["low_v0"], d["high_v0"]) for d in data.values())
x_lo, x_hi = ax.get_xlim()
ax.set_xlim(all_low - 0.05 * (x_hi - all_low), x_hi)

plt.tight_layout()
plt.subplots_adjust(left=0.35)

out = SCRIPT_DIR / "tornado_plot.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
