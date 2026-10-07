# Reproducing the stochastic modelling, optimization, and ROA pipeline

This repository evaluates investment options for a battery energy storage system (BESS) and pumped-storage hydropower (PSH).

The pipeline has three stages:

1. **Stochastic modelling**: calibrate and simulate joint day-ahead spot and FCR price paths.
2. **Optimization**: run the Gurobi dispatch model on those price paths and create annual BESS and PSH revenue files.
3. **ROA**: value the investment and deferral options with least-squares Monte Carlo (LSMC).

Run all commands from the repository root:

```bash
cd /path/to/masteroppgave
```

## 1. Install prerequisites

Use Python 3.11 or newer. Create and activate a virtual environment, then install the Python dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The optimization stage requires:

- a working Gurobi installation
- a valid Gurobi license
- enough memory and time for the selected number of price paths

## 2. Prepare the historical price data

The stochastic models read these files:

- `stochastic_modelling/input/DA/DA_NO2_NOK.csv`
- `stochastic_modelling/input/FCR_NO2_NOK.csv`

The expected columns are:

- day-ahead: `Timestamp`, `Price_DA`
- FCR: `Timestamp`, `Price_FCR`

These files are already in the repository. If they are replaced, keep the same filenames and column names.

## 3. Calibrate the stochastic models

Run:

```bash
python stochastic_modelling/calibrate_joint.py
```

This calibrates the spot and FCR regime-switching models and writes:

```text
stochastic_modelling/results/params.csv
```

Calibration is only needed when the historical input data or model specification changes. If a compatible `params.csv` is already available, this step can be skipped.

## 4. Simulate joint price paths

Run:

```bash
python stochastic_modelling/forecast_joint.py
```

The script uses the calibrated parameters and simulates the configured number of 80-year paths. The relevant settings are near the top of `stochastic_modelling/forecast_joint.py`:

- `N_YEARS`
- `N_BATCHES`
- `N_SIMULATIONS`
- `INFLATION_RATE`

The forecast script currently writes its parquet files to:

```text
../../../storage/users/einami/input_files
```

That path is resolved relative to the current working directory. Create the directory or change `RESULTS_DIR` in `forecast_joint.py` to a local directory before running. The optimizer must read the same directory.

The expected output files are pairs such as:

```text
spot_1.parquet
FCR_1.parquet
spot_2.parquet
FCR_2.parquet
```

Each pair contains hourly paths with a `timestamp` column and one column per simulated path.

## 5. Run the dispatch optimization

The optimizer reads matching spot and FCR parquet files, then runs the BESS and PSH dispatch models with Gurobi.

Start it with:

```bash
python optimization/optimal_dispatch_LP.py
```

The script is interactive. Select:

1. both storage technologies
2. the matching price-file number
3. the default storage parameters unless a sensitivity is intended

For a quick single-path test, add `--test`:

```bash
python optimization/optimal_dispatch_LP.py --test
```

The optimizer writes results under:

```text
optimization/results/paths/
```

For the ROA stage, the important result columns are:

- `path_id`
- `year`
- `rev_DA`
- `rev_FCR`
- `rev_total`
- `bid_DA_buy`
- `bid_DA_sell`
- `avg_DA_price`
- `avg_FCR_price`

The ROA loader expects the combined BESS and PSH files to be named:

```text
roa/input_files/BESS_optimized_results_combined.csv
roa/input_files/PSH_optimized_results_combined.csv
```

If the optimizer produced several batches, combine the corresponding files before continuing. The helper is:

```bash
python roa/input_files/combine_results.py
```

It asks for a prefix and combines CSV files from `roa/input_files/unjoint/`. Alternatively, combine the optimizer outputs externally, provided the final files have the canonical names above and retain the required columns.

## 6. Run the ROA valuation

Run the production headline valuation with:

```bash
python -m roa.main
```

The default run uses:

- 30 independent BESS CAPEX seeds
- 10,000 price/dispatch paths
- an 80-year horizon
- five tax and electricity-tax scenarios
- PSH Concept 1 by default in the current `Config`

For a fast one-seed check:

```bash
python -m roa.main --K 1
```

For a custom output directory:

```bash
python -m roa.main --K 30 --out roa/output_files/my_run
```

Results are written as timestamped JSON and CSV files under `roa/output_files/` unless another output directory is supplied.

## 7. Optional plots and studies

Generate figures from a completed ROA JSON result:

```bash
python -m roa.plots path/to/main_pipeline_result.json
```

Run the convergence study:

```bash
python -m roa.study.convergence_basis --quick
```

These are analysis steps and are not required to reproduce the main ROA valuation.

## ROA-only verification

If the purpose is only to verify the final ROA calculations, the upstream stages can be omitted. Copy these items into a clean repository:

```text
requirements.txt
roa/__init__.py
roa/config.py
roa/investment.py
roa/lsmc.py
roa/main.py
roa/input_files/BESS_optimized_results_combined.csv
roa/input_files/PSH_optimized_results_combined.csv
```

Then run:

```bash
python -m roa.main --K 1
```

The two combined CSVs must contain matching `(path_id, year)` rows and the embedded annual price columns `avg_DA_price` and `avg_FCR_price`. The production run additionally requires the canonical data shape of 10,000 paths and 80 years.

## Common problems

### Missing `params.csv`

Run `python stochastic_modelling/calibrate_joint.py`, or place a compatible calibrated file at `stochastic_modelling/results/params.csv`.

### Price parquet files not found

Check that `forecast_joint.py` and `optimal_dispatch_LP.py` point to the same directory. Their current default is the external `storage/users/einami/input_files` location.

### Gurobi license errors

The stochastic and ROA stages do not require Gurobi. The dispatch optimization does. Install/configure Gurobi and verify the license before running the optimizer.

### ROA input file not found

Ensure these exact files exist:

```text
roa/input_files/BESS_optimized_results_combined.csv
roa/input_files/PSH_optimized_results_combined.csv
```

### Non-canonical path count or horizon

Use `--allow-nonproduction-shape` only through a compatible wrapper or direct code change for diagnostic runs. The headline `python -m roa.main` command expects 10,000 paths and 80 years.
