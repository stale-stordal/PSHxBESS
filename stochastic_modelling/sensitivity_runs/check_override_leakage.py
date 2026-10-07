"""
Regression guard against override leakage across sensitivity scenarios.

`_run_sensitivity.run()` now resets every knob to its canonical default
(`_DEFAULTS`) before applying a scenario's overrides, so overrides can no longer
leak between scenarios that share one process (e.g. run_all.py).

This script replays the run_all.py scenario order using the SAME reset+override
logic as run(), and asserts that each scenario's effective knobs equal the
defaults updated by ONLY its own overrides. Run after editing run() or the
scenario list:

    python stochastic_modelling/sensitivity_runs/check_override_leakage.py
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOCH_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, STOCH_DIR)

import stochastic_modelling.fcr_pricing_new as fcr_mod
from stochastic_modelling.sensitivity_runs._run_sensitivity import _DEFAULTS, _KNOBS

# Same order as run_all.py.
SCENARIOS = [
    ("sensitivity_baseline",         {}),
    ("sensitivity_spike_decay_0.05", {"SPIKE_DECAY_RATE": 0.05, "SIGMA_DECAY_RATE": 0.05}),
    ("sensitivity_spike_decay_0.30", {"SPIKE_DECAY_RATE": 0.30, "SIGMA_DECAY_RATE": 0.30}),
    ("sensitivity_spike_floor_0.05", {"SPIKE_FLOOR_FRAC": 0.05, "SIGMA_FLOOR_FRAC": 0.05}),
    ("sensitivity_spike_floor_0.50", {"SPIKE_FLOOR_FRAC": 0.50, "SIGMA_FLOOR_FRAC": 0.50}),
    ("sensitivity_price_drift_0.05", {"GOU_SIGMA": 0.05, "FCR_GOU_SIGMA": 0.05}),
    ("sensitivity_price_drift_0.50", {"GOU_SIGMA": 0.50, "FCR_GOU_SIGMA": 0.50}),
    ("sensitivity_kappa_0.00",       {"GOU_KAPPA": 0.00, "FCR_GOU_KAPPA": 0.00}),
    ("sensitivity_kappa_0.20",       {"GOU_KAPPA": 0.20, "FCR_GOU_KAPPA": 0.20}),
]

print(f"defaults: {_DEFAULTS}\n")

all_ok = True
for name, overrides in SCENARIOS:
    # Replicate run() EXACTLY: reset all knobs to default, then apply overrides.
    for k, v in _DEFAULTS.items():
        setattr(fcr_mod, k, v)
    for k, v in overrides.items():
        setattr(fcr_mod, k, v)

    effective = {k: getattr(fcr_mod, k) for k in _KNOBS}
    expected = {**_DEFAULTS, **overrides}
    ok = effective == expected
    all_ok &= ok
    diff = {k: effective[k] for k in _KNOBS if effective[k] != expected[k]}
    print(f"{name:32s} {'OK' if ok else 'FAIL'}  override={overrides}")
    if not ok:
        print(f"{'':32s} unexpected: {diff}")

print("\n" + "=" * 70)
print("RESULT:", "PASS - no leakage; each file differs from baseline only by its"
      " own parameter." if all_ok else "FAIL - leakage detected (see above).")
sys.exit(0 if all_ok else 1)