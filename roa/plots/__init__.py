"""ROA plotting module — publication-ready figures from main_pipeline JSON."""

from .loader import load_results, PipelineResults, ScenarioData
from .figures import (
    collect_pi_t0_k30,
    plot_pi_distributions_t0,
    plot_capex_heatmap,
    make_all_plots,
)

__all__ = [
    "load_results",
    "PipelineResults",
    "ScenarioData",
    "collect_pi_t0_k30",
    "plot_pi_distributions_t0",
    "plot_capex_heatmap",
    "make_all_plots",
]
