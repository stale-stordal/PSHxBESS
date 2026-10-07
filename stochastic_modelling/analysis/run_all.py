"""
Run every analysis script in this directory for one or more (spot, FCR) parquet
pairs.

Three interactive prompts:
  1. Where to load the parquet files from
       - current location (analysis/data/)
       - external storage (../../../storage/users/einami/input_files)
  2. Which file pairs to process
       - hard-coded list of 5 (spot, FCR) placeholder pairs (edit below)
       - specify file names manually, one pair at a time
  3. Run mode
       - per-scenario only (one set of plots per pair)
       - merged only (one set of plots over all paths combined)
       - both

Compute_historical.py and make_calibration_table.py do not depend on the
simulation parquets and are run only once, writing their outputs to
analysis/data/ and analysis/figures/.

For each (spot, FCR) parquet pair, compute_simulation.py runs with
STOCH_INPUT_DIR / STOCH_SPOT_FILE / STOCH_FCR_FILE / STOCH_SCENARIO set, and
its outputs land in analysis/data/<scenario>/ so successive pairs do not
overwrite each other. The historical CSV/NPY files are copied into each
scenario data folder so the plot scripts can find them.

In merged mode the per-scenario sim_metrics_<market>.npz files are stacked
along the path axis (axis 0) into analysis/data/merged/sim_metrics_<market>.npz
and the plot scripts are run once with STOCH_SCENARIO=merged. plot_backtest
samples paths from every parquet via STOCH_MERGED_SPOT_FILES /
STOCH_MERGED_FCR_FILES.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR   = Path(__file__).resolve().parent
LOCAL_DIR    = str((SCRIPT_DIR / "data").resolve())
EXTERNAL_DIR = '../../../../../storage/users/einami/input_files'

# ---- placeholder pairs -- edit these to match your actual parquet files ----
HARDCODED_PAIRS = [
    ("spot_1.parquet", "FCR_1.parquet"),
    ("spot_2.parquet", "FCR_2.parquet"),
    ("spot_3.parquet", "FCR_3.parquet"),
    ("spot_4.parquet", "FCR_4.parquet"),
    ("spot_5.parquet", "FCR_5.parquet"),
]

# Scripts that do not depend on a specific simulation parquet.
ONE_SHOT_SCRIPTS = [
    "compute_historical.py",
    "make_calibration_table.py",
]

# Always run per pair so the npz metric files exist for both per-scenario
# plotting and the merge step.
COMPUTE_SCRIPT = "compute_simulation.py"   # accepts "all" / "spot" / "fcr" as argv[1]

# Plot scripts that consume the per-scenario or merged sim_metrics_*.npz files.
PLOT_SCRIPTS = [
    "plot_yearly_mean.py",
    "plot_joint.py",
    "plot_spreads.py",
    "plot_backtest.py",
    "yearly_moments_table.py",
]

MERGED_SCENARIO = "merged"

# Historical artefacts that plot scripts read from DATA_OUT. They are produced
# by compute_historical.py in analysis/data/ and copied into each scenario
# subfolder so plot scripts find them when DATA_OUT is per-scenario.
HISTORICAL_ARTEFACTS = [
    "historical_metrics.csv",
    "historical_returns_spot.npy",
    "historical_returns_fcr.npy",
]


def _ask(prompt: str, choices: dict) -> str:
    print(prompt)
    for k, v in choices.items():
        print(f"  [{k}] {v}")
    while True:
        choice = input("Select: ").strip()
        if choice in choices:
            return choice
        print("Invalid choice, try again.")


def ask_input_dir() -> str:
    sel = _ask(
        "\nWhere should I load the simulation parquet files from?",
        {
            "1": f"Current location   ({LOCAL_DIR})",
            "2": f"External storage   ({EXTERNAL_DIR})",
        },
    )
    return LOCAL_DIR if sel == "1" else EXTERNAL_DIR


def ask_mode() -> str:
    sel = _ask(
        "\nWhich run mode?",
        {
            "1": "Per-scenario only (one set of plots per pair)",
            "2": "Merged only (one set of plots over all paths combined)",
            "3": "Both per-scenario and merged",
        },
    )
    return {"1": "per", "2": "merged", "3": "both"}[sel]


def ask_pairs() -> list[tuple[str, str]]:
    sel = _ask(
        "\nHow should I select the parquet files to analyse?",
        {
            "1": f"Hard-coded list ({len(HARDCODED_PAIRS)} pairs from run_all.py)",
            "2": "Specify file names manually",
        },
    )
    if sel == "1":
        return list(HARDCODED_PAIRS)

    print("\nEnter pairs as:  <spot file>  <FCR file>")
    print("Press Enter on an empty line to finish.")
    pairs: list[tuple[str, str]] = []
    while True:
        line = input(f"  pair {len(pairs) + 1}: ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 2:
            print("    expected two filenames separated by whitespace")
            continue
        pairs.append((parts[0], parts[1]))
    if not pairs:
        sys.exit("No pairs entered, aborting.")
    return pairs


def scenario_name(spot_file: str) -> str:
    """Derive a folder-safe scenario tag from the spot file name."""
    stem = Path(spot_file).stem
    return stem[len("spot_"):] if stem.lower().startswith("spot_") else stem


def run(script: str, env: dict, *args: str) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=SCRIPT_DIR, env=env, check=True)


def copy_historical_into(scenario_data_dir: Path) -> None:
    scenario_data_dir.mkdir(parents=True, exist_ok=True)
    for name in HISTORICAL_ARTEFACTS:
        src = SCRIPT_DIR / "data" / name
        if src.exists():
            shutil.copy2(src, scenario_data_dir / name)


def merge_sim_metrics(scenarios: list[str]) -> None:
    """Stack per-scenario sim_metrics_<market>.npz along the path axis."""
    merged_dir = SCRIPT_DIR / "data" / MERGED_SCENARIO
    merged_dir.mkdir(parents=True, exist_ok=True)

    for market in ("spot", "fcr"):
        per_scenario_arrays: dict[str, list[np.ndarray]] = {}
        start_year = None
        for scen in scenarios:
            src = SCRIPT_DIR / "data" / scen / f"sim_metrics_{market}.npz"
            if not src.exists():
                print(f"[merge] missing {src}, skipping {market}")
                per_scenario_arrays = {}
                break
            with np.load(src) as d:
                sy = int(d["start_year"])
                if start_year is None:
                    start_year = sy
                elif sy != start_year:
                    print(f"[merge] start_year mismatch for {market}: "
                          f"{sy} vs {start_year} in {src}")
                for key in ("yearly_mean", "daily_spread", "weekly_spread",
                            "intra_year_amp", "hourly_vol"):
                    per_scenario_arrays.setdefault(key, []).append(d[key])

        if not per_scenario_arrays:
            continue

        stacked = {k: np.concatenate(v, axis=0) for k, v in per_scenario_arrays.items()}
        n_paths_total = stacked["yearly_mean"].shape[0]
        out_path = merged_dir / f"sim_metrics_{market}.npz"
        np.savez_compressed(
            out_path,
            yearly_mean=stacked["yearly_mean"],
            daily_spread=stacked["daily_spread"],
            weekly_spread=stacked["weekly_spread"],
            intra_year_amp=stacked["intra_year_amp"],
            hourly_vol=stacked["hourly_vol"],
            start_year=start_year,
            n_paths=n_paths_total,
        )
        print(f"[merge] saved {out_path}  (n_paths={n_paths_total})")


def main() -> None:
    input_dir = ask_input_dir()
    pairs     = ask_pairs()
    mode      = ask_mode()

    print("\n" + "=" * 60)
    print(f"Input dir: {input_dir}")
    print(f"Mode:      {mode}")
    print(f"Pairs ({len(pairs)}):")
    for spot, fcr in pairs:
        print(f"  spot={spot}   fcr={fcr}")
    print("=" * 60)

    base_env = os.environ.copy()
    for script in ONE_SHOT_SCRIPTS:
        run(script, base_env)

    scenarios: list[str] = []
    for spot_file, fcr_file in pairs:
        scen = scenario_name(spot_file)
        scenarios.append(scen)
        print("\n" + "#" * 60)
        print(f"# scenario: {scen}")
        print(f"#   spot = {spot_file}")
        print(f"#   fcr  = {fcr_file}")
        print("#" * 60)

        copy_historical_into(SCRIPT_DIR / "data" / scen)
        (SCRIPT_DIR / "figures" / scen).mkdir(parents=True, exist_ok=True)

        env = base_env.copy()
        env["STOCH_INPUT_DIR"] = input_dir
        env["STOCH_SPOT_FILE"] = spot_file
        env["STOCH_FCR_FILE"]  = fcr_file
        env["STOCH_SCENARIO"]  = scen

        run(COMPUTE_SCRIPT, env)

        if mode in ("per", "both"):
            for script in PLOT_SCRIPTS:
                run(script, env)

    if mode in ("merged", "both"):
        if len(pairs) < 2:
            print("\n[merge] only one pair given — merged mode would just "
                  "duplicate the per-scenario result; skipping.")
        else:
            print("\n" + "#" * 60)
            print(f"# scenario: {MERGED_SCENARIO}  (stacking {len(scenarios)} pairs)")
            print("#" * 60)
            merge_sim_metrics(scenarios)
            copy_historical_into(SCRIPT_DIR / "data" / MERGED_SCENARIO)
            (SCRIPT_DIR / "figures" / MERGED_SCENARIO).mkdir(parents=True, exist_ok=True)

            spot_files = ",".join(s for s, _ in pairs)
            fcr_files  = ",".join(f for _, f in pairs)

            env = base_env.copy()
            env["STOCH_INPUT_DIR"]        = input_dir
            env["STOCH_SCENARIO"]         = MERGED_SCENARIO
            env["STOCH_MERGED_SPOT_FILES"] = spot_files
            env["STOCH_MERGED_FCR_FILES"]  = fcr_files
            # The single-file env vars still need values for any code that
            # reads them at import time; point them at the first pair.
            env["STOCH_SPOT_FILE"] = pairs[0][0]
            env["STOCH_FCR_FILE"]  = pairs[0][1]

            for script in PLOT_SCRIPTS:
                run(script, env)

    print("\nAll analyses finished.")


if __name__ == "__main__":
    main()
