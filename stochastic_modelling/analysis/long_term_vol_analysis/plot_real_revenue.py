"""
Real (inflation-stripped) annual revenue — PSH and BESS.

Requires strip_inflation.py to have been run first.

Output: real_revenue_bands.pdf  (same folder as this script)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)
from thesis_style import COLORS, apply_style

apply_style()

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
psh  = pd.read_csv(os.path.join(SCRIPT_DIR, "PSH_real.csv"))
bess = pd.read_csv(os.path.join(SCRIPT_DIR, "BESS_real.csv"))

# ---------------------------------------------------------------------------
# Compute yearly quantiles across paths
# ---------------------------------------------------------------------------
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

def yearly_stats(df, col):
    g = df.groupby("year")[col]
    q = {p: g.quantile(p).values for p in QUANTILES}
    years = g.quantile(0.50).index.values
    return q, years

psh_q,  years = yearly_stats(psh,  "rev_total")
bess_q, _     = yearly_stats(bess, "rev_total")

# Convert to millions NOK
for d in (psh_q, bess_q):
    for k in d:
        d[k] = d[k] / 1e6

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7.5),
    sharex=True,
    gridspec_kw={"hspace": 0.15},
)

def _panel(ax, q, ylabel, color):
    ax.fill_between(years, q[0.10], q[0.90],
                    color=color, alpha=0.15, lw=0, label="10–90th pct.")
    ax.fill_between(years, q[0.25], q[0.75],
                    color=color, alpha=0.30, lw=0, label="25–75th pct.")
    ax.plot(years, q[0.50],
            color=color, lw=1.8, label="Median")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, ncol=3)

_panel(ax1, psh_q,
       "PSH annual revenue\n(real 2025 NOK, MNOK)",
       COLORS["main"])
_panel(ax2, bess_q,
       "BESS annual revenue\n(real 2025 NOK, MNOK)",
       COLORS["secondary"])

ax2.set_xlabel("Year from simulation start")
ax1.set_xlim(years[0], years[-1])

plt.tight_layout()

out = os.path.join(SCRIPT_DIR, "real_revenue_bands.pdf")
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved {out}")
