"""Sensitivity: high long-run volatility floor  (floor = 50% of σ̂, baseline = 20%)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.sensitivity_runs._run_sensitivity import run

run(
    run_name  = "sensitivity_spike_floor_0.50",
    overrides = {
        "SPIKE_FLOOR_FRAC": 0.50,
        "SIGMA_FLOOR_FRAC": 0.50,
    },
)
