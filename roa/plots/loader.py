"""Load ROA pipeline JSON output into a structured form for plotting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ScenarioData:
    name: str
    panel: str
    elavgift_psh: float
    grunnrente_mode: str
    runs: dict
    payoff_diagnostics: dict
    cashflow_decomposition: dict
    psh_tax_decomposition: dict
    Pi_BESS_paths_t0: np.ndarray
    Pi_PSH_paths_t0: np.ndarray


@dataclass
class PipelineResults:
    metadata: dict
    scenarios: dict[str, ScenarioData] = field(default_factory=dict)


def load_results(json_path: str | Path) -> PipelineResults:
    json_path = Path(json_path)
    with open(json_path) as f:
        d = json.load(f)
    out = PipelineResults(metadata=d["metadata"])
    for name, sr in d["scenarios"].items():
        out.scenarios[name] = ScenarioData(
            name=name,
            panel=sr["panel"],
            elavgift_psh=sr["elavgift_nok_per_mwh_psh"],
            grunnrente_mode=sr["grunnrente_mode"],
            runs=sr["runs"],
            payoff_diagnostics=sr["payoff_diagnostics"],
            cashflow_decomposition=sr["cashflow_decomposition"],
            psh_tax_decomposition=sr["psh_tax_decomposition"],
            Pi_BESS_paths_t0=np.asarray(sr["Pi_BESS_paths_t0"], dtype=float),
            Pi_PSH_paths_t0=np.asarray(sr["Pi_PSH_paths_t0"], dtype=float),
        )
    return out
