"""Run all sensitivity scenarios in sequence."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.sensitivity_runs._run_sensitivity import run

scenarios = [
    # Baseline
    ("sensitivity_baseline",          {}),
    # Spike decay
    ("sensitivity_spike_decay_0.05",  {"SPIKE_DECAY_RATE": 0.05, "SIGMA_DECAY_RATE": 0.05}),
    ("sensitivity_spike_decay_0.30",  {"SPIKE_DECAY_RATE": 0.30, "SIGMA_DECAY_RATE": 0.30}),
    # Spike floor
    ("sensitivity_spike_floor_0.05",  {"SPIKE_FLOOR_FRAC": 0.05, "SIGMA_FLOOR_FRAC": 0.05}),
    ("sensitivity_spike_floor_0.50",  {"SPIKE_FLOOR_FRAC": 0.50, "SIGMA_FLOOR_FRAC": 0.50}),
    # GOU volatility
    ("sensitivity_price_drift_0.05",  {"GOU_SIGMA": 0.05, "FCR_GOU_SIGMA": 0.05}),
    ("sensitivity_price_drift_0.50",  {"GOU_SIGMA": 0.50, "FCR_GOU_SIGMA": 0.50}),
    # GOU level mean-reversion speed (kappa)
    ("sensitivity_kappa_0.00",        {"GOU_KAPPA": 0.00, "FCR_GOU_KAPPA": 0.00}),
    ("sensitivity_kappa_0.20",        {"GOU_KAPPA": 0.20, "FCR_GOU_KAPPA": 0.20}),
]

for run_name, overrides in scenarios:
    run(run_name, overrides)
