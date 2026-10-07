"""Sensitivity: high structural price drift  (GOU σ = 0.50, baseline = 0.2)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.sensitivity_runs._run_sensitivity import run

run(
    run_name  = "sensitivity_price_drift_0.50",
    overrides = {
        "GOU_SIGMA":     0.50,
        "FCR_GOU_SIGMA": 0.50,
    },
)
