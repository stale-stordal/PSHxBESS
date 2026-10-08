"""FCR max-bid sensitivity — real-option value V0 per technology.

Reads the per-scenario JSONs produced by run_roa_sensitivity.py and plots
how the LSMC real-option value V0 responds to the FCR market max-bid cap,
for each technology as a standalone option:

  - BESS  : V0 of the BESS-only option   (runs["bess_only"])
  - PSH   : V0 of the PSH-only option     (runs["psh_only"])

Both are read from the ``full_GRsymmetric`` tax scenario (full elavgift +
symmetric resource rent — the realistic PSH treatment). The current
market cap (70 MW) is the intended baseline and is marked with a reference
line.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from thesis_style import COLORS, FIGSIZE, LAYOUT, apply_style

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR  = SCRIPT_DIR / "results"
SCENARIO     = "full_GRsymmetric"     # full elavgift + symmetric resource rent
BASELINE_CAP = 70                     # current market max bid [MW]

TECH_COLORS = {
    "PSH":  COLORS["main"],
    "BESS": COLORS["secondary"],
}

# ---------------------------------------------------------------------------
# Load V0 per cap
# ---------------------------------------------------------------------------
def _cap_from_stem(stem):
    return int(stem.replace("FCR_", ""))

records = []
for p in sorted(RESULTS_DIR.glob("FCR_*.json"), key=lambda x: _cap_from_stem(x.stem)):
    d = json.load(open(p))
    runs = d["scenarios"][SCENARIO]["runs"]
    bess = runs["bess_only"]["V0_stoptime"] / 1e6
    psh  = runs["psh_only"]["V0_stoptime"] / 1e6
    records.append({
        "cap":  _cap_from_stem(d["scenario"]),
        "bess": bess,
        "psh":  psh,   # standalone PSH option value
    })

records.sort(key=lambda r: r["cap"])
caps      = [r["cap"]  for r in records]
bess_v0   = [r["bess"] for r in records]
psh_v0    = [r["psh"]  for r in records]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
apply_style(layout="full")
fig, ax = plt.subplots(figsize=FIGSIZE["full"])
ax.grid(axis="y")
ax.set_axisbelow(True)

ax.plot(caps, psh_v0, color=TECH_COLORS["PSH"], lw=LAYOUT["line_width"],
        marker="o", markersize=7, label="PSH option value")
ax.plot(caps, bess_v0, color=TECH_COLORS["BESS"], lw=LAYOUT["line_width_sec"],
        marker="s", markersize=7, label="BESS option value")

# Reference line marking the current market cap.
ax.axvline(BASELINE_CAP, color=COLORS["neutral"], lw=1.0, ls="--", alpha=0.7)
ax.text(BASELINE_CAP, ax.get_ylim()[1], "  current cap (70 MW)",
        ha="left", va="top", color=COLORS["neutral"], fontsize=12)

# Annotate the BESS plateau (saturates once cap >= BESS power rating).
sat = max(bess_v0)
ax.annotate(f"BESS saturates around {sat:.0f} MNOK",
            xy=(caps[-1], sat), xytext=(caps[-1] - 2, sat + 120),
            ha="right", va="bottom", color=TECH_COLORS["BESS"], fontsize=12)

ax.set_xlabel("FCR market max bid (MW)")
ax.set_ylabel("Real option value $V_0$ (MNOK)")
ax.set_xticks(caps)
ax.set_xlim(min(caps) - 3, max(caps) + 3)
ax.set_ylim(bottom=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

ax.legend(loc="upper center", bbox_to_anchor=LAYOUT["legend_bbox"],
          ncol=2, frameon=False)

plt.tight_layout()
for ext in ("pdf", "png"):
    out = SCRIPT_DIR / f"fcr_sensitivity_option_value.{ext}"
    fig.savefig(out, bbox_inches="tight", dpi=200 if ext == "png" else None)
plt.close(fig)
print(f"Saved {SCRIPT_DIR / 'fcr_sensitivity_option_value.pdf'}")
