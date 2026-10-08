"""
ROA pipeline over all price-modelling sensitivity scenarios.

For each (Battery_*, Pump_*) file pair in this folder:
  1. Copies them to a temp directory under the canonical file names
     expected by load_optimizer_data().
  2. Monkeypatches roa.main._INPUT_DIR and roa.main.load_optimizer_data
     so run_pipeline() reads from the temp directory.
  3. Runs run_pipeline(allow_nonproduction_shape=True) — needed because
     N=1000 here vs the canonical N=4000.
  4. Saves a compact JSON to results/<scenario_name>.json.
  5. Restores all patches before moving to the next scenario.

No ROA source files are modified.

Usage
-----
    cd masteroppgave
    python roa/input_files/sensitivity_price_modelling/run_roa_sensitivity.py
"""

import functools
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
REPO_ROOT  = SCRIPT_DIR.parent.parent.parent   # masteroppgave/
sys.path.insert(0, str(REPO_ROOT))

import roa.main as roa_main
from roa.config import Config

OUTPUT_DIR = SCRIPT_DIR / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Discover scenario pairs: Battery_*  <->  Pump_*
# ---------------------------------------------------------------------------
scenarios = {}
for bess_file in sorted(SCRIPT_DIR.glob("Battery_optimized_results_*.csv")):
    key = bess_file.stem.replace("Battery_optimized_results_", "")
    psh_file = SCRIPT_DIR / f"Pump_optimized_results_{key}.csv"
    if psh_file.exists():
        scenarios[key] = (bess_file, psh_file)
    else:
        print(f"  WARNING: no matching Pump file for {bess_file.name} — skipped.")

print(f"Found {len(scenarios)} scenario(s): {list(scenarios.keys())}\n")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for scenario_name, (bess_src, psh_src) in scenarios.items():
    print(f"{'='*60}")
    print(f"  Scenario: {scenario_name}")
    print(f"{'='*60}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Rename to canonical names expected by load_optimizer_data()
            shutil.copy(bess_src, tmp_path / "BESS_optimized_results_combined.csv")
            shutil.copy(psh_src,  tmp_path / "PSH_optimized_results_combined.csv")

            # Patch roa.main so both _detect_loader_mode and load_optimizer_data
            # read from tmp_path instead of roa/input_files/
            _orig_input_dir      = roa_main._INPUT_DIR
            _orig_load_optimizer = roa_main.load_optimizer_data

            roa_main._INPUT_DIR          = tmp_path
            roa_main.load_optimizer_data = functools.partial(
                _orig_load_optimizer, data_dir=tmp_path
            )

            try:
                cfg = Config()
                results = roa_main.run_pipeline(
                    cfg,
                    seed=42,
                    allow_nonproduction_shape=True,
                )
            finally:
                roa_main._INPUT_DIR          = _orig_input_dir
                roa_main.load_optimizer_data = _orig_load_optimizer
    except Exception as exc:
        print(f"  SKIPPED {scenario_name}: {type(exc).__name__}: {exc}\n")
        continue

    # ------------------------------------------------------------------
    # Save compact JSON (strip non-serialisable numpy arrays)
    # ------------------------------------------------------------------
    serializable = {
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

    out_path = OUTPUT_DIR / f"{scenario_name}.json"
    with out_path.open("w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved -> results/{out_path.name}\n")

print(f"All done. Results in {OUTPUT_DIR}/")
