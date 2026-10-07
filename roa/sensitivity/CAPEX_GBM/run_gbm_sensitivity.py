"""
BESS CAPEX GBM parameter sensitivity for PSH Concept 1.

Holds all ROA inputs fixed except the GBM drift (gbm_mu) and volatility
(gbm_sigma) that govern the BESS unit-cost process.  Reports the
full_GRsymmetric tax scenario only.

Canonical values (Config defaults):
    gbm_mu    = -0.022   (−2.2 % / yr cost decline)
    gbm_sigma =  0.25    (25 % annual volatility)

Eight scenarios
---------------
  --- drift variants (sigma fixed at 0.25) ---
  mu_zero            mu= 0.000  sigma=0.25   flat costs
  baseline           mu=-0.022  sigma=0.25   canonical
  mu_neg10           mu=-0.100  sigma=0.25   strong decline
  mu_neg20           mu=-0.200  sigma=0.25   very strong decline
  --- volatility variants (mu fixed at -0.022) ---
  sigma_005          mu=-0.022  sigma=0.05   very low uncertainty
  sigma_015          mu=-0.022  sigma=0.15   low uncertainty
  sigma_035          mu=-0.022  sigma=0.35   high uncertainty
  sigma_045          mu=-0.022  sigma=0.45   very high uncertainty

Reads the canonical combined CSV files from roa/input_files/ (unchanged).
Writes results/gbm_sensitivity_<timestamp>.json.

Usage
-----
    cd masteroppgave
    python roa/sensitivity/CAPEX_GBM/run_gbm_sensitivity.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import roa.main as roa_main
from roa.config import Config

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_SCENARIO = "full_GRsymmetric"

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, float, float]] = [
    # (label,              gbm_mu,   gbm_sigma)
    # Drift variants (sigma fixed at canonical 0.25)
    ("mu_zero",             0.000,   0.25),
    ("baseline",           -0.022,   0.25),
    ("mu_neg10",           -0.100,   0.25),
    ("mu_neg20",           -0.200,   0.25),
    # Volatility variants (mu fixed at canonical -0.022)
    ("sigma_005",          -0.022,   0.05),
    ("sigma_015",          -0.022,   0.15),
    ("sigma_035",          -0.022,   0.35),
    ("sigma_045",          -0.022,   0.45),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_one(mu: float, sigma: float) -> dict:
    """Run K=1 pipeline with overridden GBM parameters."""
    cfg = Config()
    cfg.psh_concept = 1
    cfg.gbm_mu    = mu
    cfg.gbm_sigma = sigma
    return roa_main.run_pipeline(cfg, seed=42, allow_nonproduction_shape=True)


def _extract(results: dict, label: str, mu: float, sigma: float) -> dict:
    """Pull out the target scenario record and attach run metadata."""
    sr = results["scenarios"][TARGET_SCENARIO]
    joint = sr["runs"]["joint"]
    bess  = sr["runs"]["bess_only"]
    return {
        "label":             label,
        "gbm_mu":            mu,
        "gbm_sigma":         sigma,
        "psh_concept":       results["metadata"]["psh_concept"],
        "capex_psh_nok":     results["metadata"]["capex_psh"],
        "N":                 results["metadata"]["N"],
        "tax_scenario":      TARGET_SCENARIO,
        "V0_joint_nok":      joint["V0_stoptime"],
        "V0_bess_only_nok":  bess["V0_stoptime"],
        "exercise_rate":     joint["exercise_rate"],
        "p_psh_given_ex":    joint["p_psh_given_ex"],
        "p_bess_given_ex":   joint["p_bess_given_ex"],
        "SE_nok":            joint["SE"],
        "CI95_low_nok":      joint["CI95_low"],
        "CI95_high_nok":     joint["CI95_high"],
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

print("=" * 64)
print("  BESS CAPEX GBM sensitivity sweep — PSH Concept 1")
print(f"  Target tax scenario: {TARGET_SCENARIO}")
print(f"  Scenarios: {len(SCENARIOS)}  (baseline deduped; 8 unique runs)")
print("=" * 64)

t_start = time.perf_counter()
records: list[dict] = []

for label, mu, sigma in SCENARIOS:
    print(f"\n  [{label}]  mu={mu:+.3f}  sigma={sigma:.3f}")
    t0 = time.perf_counter()
    results = _run_one(mu, sigma)
    elapsed = time.perf_counter() - t0
    rec = _extract(results, label, mu, sigma)
    records.append(rec)
    v0_mnok = rec["V0_joint_nok"] / 1e6
    print(f"    V0 ({TARGET_SCENARIO}) = {v0_mnok:.2f} MNOK"
          f"  ex_rate={rec['exercise_rate']:.1%}"
          f"  p_psh={rec['p_psh_given_ex']:.1%}"
          f"  ({elapsed:.1f}s)")

total_elapsed = time.perf_counter() - t_start

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = OUTPUT_DIR / f"gbm_sensitivity_{timestamp}.json"

payload = {
    "metadata": {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "psh_concept":     1,
        "tax_scenario":    TARGET_SCENARIO,
        "seed":            42,
        "canonical_mu":    -0.022,
        "canonical_sigma": 0.25,
        "n_scenarios":     len(SCENARIOS),
        "elapsed_seconds": round(total_elapsed, 1),
    },
    "results": records,
}

with out_path.open("w") as f:
    json.dump(payload, f, indent=2)

print(f"\n{'='*64}")
print(f"  Done in {total_elapsed:.1f}s  —  {len(records)} scenarios")
print(f"  Saved: {out_path}")
print("=" * 64)
