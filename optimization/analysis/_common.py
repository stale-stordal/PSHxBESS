"""Shared helpers for the optimization-output analysis scripts.

Each plotting script imports this module to load the PSH and BESS
optimizer-output panels and to obtain consistent paths for figure / table
output. Keep this module thin — only logic that is genuinely shared.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Make thesis_style importable from the repo root regardless of CWD.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from thesis_style import COLORS, FIGSIZE, LAYOUT, apply_style  # noqa: E402  (after sys.path edit)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_DIR  = _REPO_ROOT / "roa" / "input_files"
FIG_DIR    = _THIS_DIR / "figures"
TABLE_DIR  = _THIS_DIR / "tables"
FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

PSH_CSV  = INPUT_DIR / "PSH_optimized_results_combined.csv"
BESS_CSV = INPUT_DIR / "BESS_optimized_results_combined.csv"

# ---------------------------------------------------------------------------
# Plant capacities (mirrored from roa/config.py active concept; PSH concept 2,
# BESS Å Energi baseline). Used for per-MW / per-MWh normalisation only.
# ---------------------------------------------------------------------------
PSH_MW       = 1_00.0
PSH_MWH      = 50_000.0
BESS_MW      = 4.6
BESS_MWH     = 10.0

# Display colours per technology (using palette from thesis_style.py).
TECH_COLORS = {
    "PSH":  COLORS["hydro"],
    "BESS": COLORS["secondary"],
}
MARKET_COLORS = {
    "DA":  COLORS["main"],
    "FCR": COLORS["highlight"],
}


def load_panel(tech: str) -> pd.DataFrame:
    """Load the long-format optimiser output for one technology.

    Returns a DataFrame indexed by (path_id, year) with all monetary columns
    in NOK and energy columns in MWh.
    """
    tech = tech.upper()
    if tech == "PSH":
        df = pd.read_csv(PSH_CSV)
    elif tech == "BESS":
        df = pd.read_csv(BESS_CSV)
    else:
        raise ValueError(f"Unknown tech: {tech!r}")
    return df


def per_year_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute per-year mean and percentile envelope for one column."""
    g = df.groupby("year")[col]
    return pd.DataFrame({
        "mean": g.mean(),
        "p05":  g.quantile(0.05),
        "p25":  g.quantile(0.25),
        "p50":  g.median(),
        "p75":  g.quantile(0.75),
        "p95":  g.quantile(0.95),
    })


def save_fig(fig, name: str) -> Path:
    """Persist a figure as both PDF (vector for thesis) and PNG (preview)."""
    out_pdf = FIG_DIR / f"{name}.pdf"
    out_png = FIG_DIR / f"{name}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    return out_pdf


__all__ = [
    "COLORS", "FIGSIZE", "LAYOUT", "apply_style",
    "INPUT_DIR", "FIG_DIR", "TABLE_DIR",
    "PSH_MW", "PSH_MWH", "BESS_MW", "BESS_MWH",
    "TECH_COLORS", "MARKET_COLORS",
    "load_panel", "per_year_stats", "save_fig",
]
