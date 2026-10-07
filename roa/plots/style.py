"""Plot style for ROA figures.

Layered on top of /masteroppgave/thesis_style.py: imports the base palette and rcParams,
adds technology / scenario palettes used across the ROA figures, and exposes a single
apply_style() entry point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

_THESIS_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_THESIS_ROOT))

try:
    from thesis_style import COLORS as _BASE_COLORS, FIGSIZE, apply_style as _apply_base
except ImportError:
    _BASE_COLORS = {
        "main":      "#2F5C85",
        "secondary": "#6F4E9C",
        "highlight": "#C44E52",
        "neutral":   "#4A4A4A",
        "grid":      "#E6E6E6",
        "text":      "#3A3A3A",
    }
    FIGSIZE = {
        "standard":  (11, 6),
        "wide":      (12, 6.5),
        "two_panel": (12, 7.5),
    }
    def _apply_base() -> None:
        mpl.rcParams.update({
            "figure.facecolor":  "white",
            "axes.facecolor":    "white",
            "savefig.facecolor": "white",
            "font.family":       "serif",
            "font.serif":        ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
            "axes.spines.top":   False,
            "axes.spines.right": False,
            "legend.frameon":    False,
        })

COLORS = dict(_BASE_COLORS)

# Technology / decision palette — reused across all ROA figures
TECH_COLORS = {
    "BESS":        "#4C78A8",
    "PSH":         "#D97B66",
    "no_exercise": "#C8C8C8",
}

SCENARIO_ORDER = [
    "full_GRnone",
    "reduced_GRnone",
    "zero_GRnone",
    "full_GRsymmetric",
    "full_GRtruncated",
]

# Forest-plot / heatmap row order: least-negative mean Π^PSH on top, most-
# negative on bottom (mean ordering at PSH Concept 2 = 15.8 BNOK CAPEX from
# the C2 K=30 verification). Used by §5.3 figures so the forest plot and
# the heatmap share a single, intentional row order.
SCENARIO_ORDER_BY_PI_PSH_MEAN = [
    "full_GRsymmetric",
    "zero_GRnone",
    "reduced_GRnone",
    "full_GRnone",
    "full_GRtruncated",
]

SCENARIO_LABELS = {
    "full_GRnone":      "Full electricity tax, no resource rent",
    "reduced_GRnone":   "Reduced electricity tax, no resource rent",
    "zero_GRnone":      "Zero electricity tax, no resource rent",
    "full_GRsymmetric": "Full electricity tax, symmetric resource rent",
    "full_GRtruncated": "Full electricity tax, truncated resource rent",
}

SCENARIO_LABELS_SHORT = {
    "full_GRnone":      "Full / no RR",
    "reduced_GRnone":   "Reduced / no RR",
    "zero_GRnone":      "Zero / no RR",
    "full_GRsymmetric": "Full / sym RR",
    "full_GRtruncated": "Full / trunc RR",
}

SCENARIO_COLORS = {
    "full_GRnone":      "#2F5C85",
    "reduced_GRnone":   "#4C78A8",
    "zero_GRnone":      "#72B7B2",
    "full_GRsymmetric": "#6F4E9C",
    "full_GRtruncated": "#C44E52",
}


def apply_style(layout: str = "full") -> None:
    _apply_base(layout=layout)
    mpl.rcParams.update({
        "axes.axisbelow":   True,
        "mathtext.fontset": "stix",
    })
