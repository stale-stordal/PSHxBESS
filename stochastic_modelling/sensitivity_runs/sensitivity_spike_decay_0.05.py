"""Sensitivity: slow spike/drop volatility decay  (λ = 0.05, baseline = 0.1)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.sensitivity_runs._run_sensitivity import run

run(
    run_name  = "sensitivity_spike_decay_0.05",
    overrides = {
        "SPIKE_DECAY_RATE": 0.05,   # spot spike/drop σ decay rate
        "SIGMA_DECAY_RATE": 0.05,   # FCR  spike/drop σ decay rate
    },
)
