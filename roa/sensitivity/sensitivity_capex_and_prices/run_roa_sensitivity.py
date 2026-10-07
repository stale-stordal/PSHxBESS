"""
ROA pipeline over all price-modelling sensitivity scenarios for PSH Concept 1.

Also runs a baseline from the same stochastic->optimization pipeline as the
sensitivity scenarios (Battery/Pump_optimized_results_baseline_new_kappa.csv),
saved to results/baseline.json, so the tornado plot has a consistent reference
point at the same N and identical price model (only the swept parameter differs).

Usage
-----
    cd masteroppgave
    python roa/input_files/concept1_sensitivity/run_roa_sensitivity.py
"""

import functools
import json
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent   # masteroppgave/
sys.path.insert(0, str(REPO_ROOT))

import roa.main as roa_main
from roa.config import Config

INPUT_DIR    = REPO_ROOT / "roa" / "input_files"
SENS_CSV_DIR = REPO_ROOT / "optimization" / "results" / "paths"
OUTPUT_DIR   = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


def _run_roa(bess_src: Path, psh_src: Path, psh_concept: int = 1) -> dict:
    """Copy files to temp dir, monkeypatch roa.main, run pipeline, restore."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.copy(bess_src, tmp_path / "BESS_optimized_results_combined.csv")
        shutil.copy(psh_src,  tmp_path / "PSH_optimized_results_combined.csv")

        _orig_input_dir      = roa_main._INPUT_DIR
        _orig_load_optimizer = roa_main.load_optimizer_data

        roa_main._INPUT_DIR          = tmp_path
        roa_main.load_optimizer_data = functools.partial(
            _orig_load_optimizer, data_dir=tmp_path
        )
        try:
            cfg = Config()
            cfg.psh_concept = psh_concept
            return roa_main.run_pipeline(cfg, seed=42, allow_nonproduction_shape=True)
        finally:
            roa_main._INPUT_DIR          = _orig_input_dir
            roa_main.load_optimizer_data = _orig_load_optimizer


def _compact(results: dict, scenario_name: str) -> dict:
    return {
        "scenario":  scenario_name,
        "metadata":  results["metadata"],
        "scenarios": {
            s: {
                "panel":                      sr["panel"],
                "elavgift_nok_per_mwh_bess":  sr["elavgift_nok_per_mwh_bess"],
                "elavgift_nok_per_mwh_psh":   sr["elavgift_nok_per_mwh_psh"],
                "grunnrente_mode":            sr["grunnrente_mode"],
                "runs":                       sr["runs"],
                "cashflow_decomposition":     sr["cashflow_decomposition"],
                "psh_tax_decomposition":      sr["psh_tax_decomposition"],
            }
            for s, sr in results["scenarios"].items()
        },
    }


# ---------------------------------------------------------------------------
# Baseline: baseline_new_kappa files (same pipeline as the sensitivity runs)
# ---------------------------------------------------------------------------
print("=" * 60)
print("  Baseline (baseline_new_kappa — same pipeline as sensitivities)")
print("=" * 60)

bess_base = SENS_CSV_DIR / "Battery_optimized_results_baseline_new_kappa.csv"
psh_base  = SENS_CSV_DIR / "Pump_optimized_results_baseline_new_kappa.csv"

baseline_results = _run_roa(bess_base, psh_base, psh_concept=1)

baseline_path = OUTPUT_DIR / "baseline.json"
with baseline_path.open("w") as f:
    json.dump(_compact(baseline_results, "baseline"), f, indent=2)
v0_base = baseline_results["scenarios"]["full_GRsymmetric"]["runs"]["joint"]["V0_stoptime"] / 1e6
print(f"  Baseline V0 (full_GRsymmetric) = {v0_base:.2f} MNOK")
print(f"  Saved -> results/baseline.json\n")

# ---------------------------------------------------------------------------
# Sensitivity scenarios: Battery_*  <->  Pump_*
# ---------------------------------------------------------------------------
scenarios = {}
for bess_file in sorted(SENS_CSV_DIR.glob("Battery_optimized_results_*new_kappa*.csv")):
    key = bess_file.stem.replace("Battery_optimized_results_", "").replace("_new_kappa", "")
    if key == "baseline":
        continue  # handled explicitly above; not a swept scenario
    psh_file = SENS_CSV_DIR / bess_file.name.replace("Battery_optimized_results_", "Pump_optimized_results_")
    if psh_file.exists():
        scenarios[key] = (bess_file, psh_file)
    else:
        print(f"  WARNING: no matching Pump file for {bess_file.name} — skipped.")

print(f"Found {len(scenarios)} sensitivity scenario(s): {list(scenarios.keys())}\n")

for scenario_name, (bess_src, psh_src) in scenarios.items():
    print(f"{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    results = _run_roa(bess_src, psh_src, psh_concept=1)

    out_path = OUTPUT_DIR / f"{scenario_name}.json"
    with out_path.open("w") as f:
        json.dump(_compact(results, scenario_name), f, indent=2)
    v0 = results["scenarios"]["full_GRsymmetric"]["runs"]["joint"]["V0_stoptime"] / 1e6
    print(f"  V0 (full_GRsymmetric) = {v0:.2f} MNOK")
    print(f"  Saved -> results/{out_path.name}\n")

print(f"All done. Results in {OUTPUT_DIR}/")
