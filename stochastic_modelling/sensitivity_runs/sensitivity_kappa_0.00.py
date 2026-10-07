"""Sensitivity: no level mean-reversion  (kappa = 0.0, baseline = 0.05)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.sensitivity_runs._run_sensitivity import run

run(
    run_name  = "sensitivity_kappa_0.00",
    overrides = {
        "GOU_KAPPA":     0.00,   # spot level (GOU) mean-reversion speed
        "FCR_GOU_KAPPA": 0.00,   # FCR  level (GOU) mean-reversion speed
    },
)