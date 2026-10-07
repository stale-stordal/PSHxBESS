"""
Created Spring 2026

@author: einar-m, matheaab, odmoe
"""

"""
This code is the code by Johnsen et al. (2023), revised and developed by Halbrendt et. al. (2025) to implement aFRR markets and adjusted by Bardal et. al. (2026) to work with batteries and pumps in pumped hydro storage.
"""

import os
import time
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from storage import Storage, Battery, Pump

REPO_ROOT  = Path(__file__).parent.parent
INPUT_DIR  = 'stochastic_modelling/output' # TODO: update to Solstorm dir if neededs
OUTPUT_DIR = REPO_ROOT / "optimization" / "results" / "paths"

########################################################################################################################
# Define ranges
########################################################################################################################

dT = 1              # hourly resolution
NT = int(24/dT)     # number of time steps
T = range(NT)       # range timesteps in horizon (24 hours)

FCR_BID_MAX = 70.0  # Maximum bid size allowed by market [MW]
MAX_YEARS   = 80

########################################################################################################################
# Define optimization model in functions
########################################################################################################################

class DispatchModelLP:
    '''Reusable LP model. Build the Gurobi model once, then update prices and
    initial SoC each call. Equivalent to the standalone optimal_dispatch() but
    avoids reconstructing the model on every year.

    The revenue terms (price * bid) live directly in the objective rather than
    in equality constraints, so per-year updates are a single vectorised
    setAttr("Obj", ...) call. rev_DA and rev_FCR are computed from the
    solution after optimize() for reporting.
    '''

    def __init__(self, storage: Storage, NT_local: int):
        capacity   = storage.capacity
        power      = storage.power
        efficiency = storage.efficiency
        
        FCR_cap = min(power, FCR_BID_MAX)

        self.storage    = storage
        self.NT_local   = NT_local
        self.T_local    = range(NT_local)
        self.capacity   = capacity
        self.efficiency = efficiency

        m = gp.Model("optimal_dispatch_lp")
        m.setParam("OutputFlag", 0)
        m.setParam("Threads", 1)

        SoC         = m.addVars(NT_local + 1, lb=0.0, ub=capacity,              vtype=GRB.CONTINUOUS, name="SoC")
        bid_DA_buy  = m.addVars(NT_local,     lb=0.0, ub=power,                 vtype=GRB.CONTINUOUS, name="bid_DA_buy")
        bid_DA_sell = m.addVars(NT_local,     lb=0.0, ub=power * efficiency,    vtype=GRB.CONTINUOUS, name="bid_DA_sell")
        bid_FCR     = m.addVars(NT_local,     lb=0.0, ub=FCR_cap,  vtype=GRB.CONTINUOUS, name="bid_FCR")

        # Objective coefficients are set per-solve; placeholder is zero.
        m.ModelSense = GRB.MAXIMIZE

        # Initial-SoC constraint: RHS updated each solve.
        c_soc_init = m.addConstr(SoC[0] == 0.0, name="c_soc_init")

        # SoC dynamics — never changes.
        m.addConstrs((SoC[t+1] == SoC[t] + efficiency * bid_DA_buy[t] - (1/efficiency) * bid_DA_sell[t]) for t in self.T_local)

        # End-of-year SoC window.
        m.addConstr(SoC[NT_local] >= capacity * 0.40)
        m.addConstr(SoC[NT_local] <= capacity * 0.60)

        # DA capacity limits.
        m.addConstrs(bid_DA_buy [t] <= (capacity - SoC[t]) / efficiency for t in self.T_local)
        m.addConstrs(bid_DA_sell[t] <=             SoC[t]  * efficiency for t in self.T_local)

        # FCR power limits.
        m.addConstrs(bid_FCR[t] <= power              - bid_DA_buy [t] for t in self.T_local)
        m.addConstrs(bid_FCR[t] <= power * efficiency - bid_DA_sell[t] for t in self.T_local)

        # FCR capacity headroom.
        m.addConstrs(bid_FCR[t] <= (capacity - SoC[t]) / efficiency - bid_DA_buy [t] for t in self.T_local)
        m.addConstrs(bid_FCR[t] <=             SoC[t]  * efficiency  - bid_DA_sell[t] for t in self.T_local)

        # Cache var list views for fast setAttr.
        self._bid_DA_sell_list = [bid_DA_sell[t] for t in self.T_local]
        self._bid_DA_buy_list  = [bid_DA_buy [t] for t in self.T_local]
        self._bid_FCR_list     = [bid_FCR    [t] for t in self.T_local]

        self.m           = m
        self.SoC         = SoC
        self.bid_DA_buy  = bid_DA_buy
        self.bid_DA_sell = bid_DA_sell
        self.bid_FCR     = bid_FCR
        self.c_soc_init  = c_soc_init

    def solve(self, SoC_init: float, price_DA, price_FCR) -> tuple:
        try:
            self.c_soc_init.RHS = float(SoC_init)

            # Objective: maximise sum(price_DA * bid_DA_sell - price_DA * bid_DA_buy + price_FCR * bid_FCR).
            # Vectorise the coefficient prep (prices arrive as float32): the
            # negation runs in C and .tolist() converts to float64 in one pass,
            # replacing per-element Python float loops over ~8760 prices/solve.
            price_DA_arr   = np.asarray(price_DA,  dtype=np.float64)
            price_FCR_arr  = np.asarray(price_FCR, dtype=np.float64)
            price_DA_list  = price_DA_arr.tolist()
            price_FCR_list = price_FCR_arr.tolist()
            self.m.setAttr("Obj", self._bid_DA_sell_list, price_DA_list)
            self.m.setAttr("Obj", self._bid_DA_buy_list,  (-price_DA_arr).tolist())
            self.m.setAttr("Obj", self._bid_FCR_list,     price_FCR_list)

            self.m.optimize()

            NT_local = self.NT_local
            T_local  = self.T_local

            bid_DA_buy_X  = [self.bid_DA_buy [t].X for t in T_local]
            bid_DA_sell_X = [self.bid_DA_sell[t].X for t in T_local]
            bid_FCR_X     = [self.bid_FCR    [t].X for t in T_local]

            data = pd.DataFrame({
                "bid_DA_buy":     bid_DA_buy_X,
                "bid_DA_sell":    bid_DA_sell_X,
                "bid_FCR":        bid_FCR_X,
                "charge_from_DA": bid_DA_buy_X,
            })

            rev_DA_val  = sum((bid_DA_sell_X[t] - bid_DA_buy_X[t]) * price_DA_list[t]  for t in T_local)
            rev_FCR_val = sum( bid_FCR_X    [t]                    * price_FCR_list[t] for t in T_local)

            revenues = pd.DataFrame([{
                "rev_DA" : rev_DA_val,
                "rev_FCR": rev_FCR_val,
            }])
            revenues['rev_total'] = revenues.sum(axis=1)

            objective      = float(self.m.ObjVal)
            soc_trajectory = [self.SoC[t].X for t in range(NT_local + 1)]
            min_soc        = min(soc_trajectory)
            max_soc        = max(soc_trajectory)

            return objective, data, revenues, self.SoC[NT_local].X, min_soc, max_soc

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ': ' + str(e))

        except AttributeError as e:
            print(f'Encountered an attribute error: {e}')


def _optimize_yearly(storage: Storage,
                     prices: pd.DataFrame,
                     path_col: str,
                     days_in_year: int = 365,
                     years_in_horizon: int = 1
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:

    yearly_data_rows = []
    yearly_rev_rows  = []
    SoC              = storage.initial_soc
    NT_year          = days_in_year * NT

    # Build the LP once per path; reuse across years (warm-started by Gurobi).
    model = DispatchModelLP(storage, NT_year)

    for year in range(years_in_horizon):
        prices_block = prices.iloc[year * NT_year : (year + 1) * NT_year]

        t0 = time.perf_counter()
        objective, data, revenues, SoC, min_SoC, max_SoC = model.solve(
            SoC_init  = SoC,
            price_DA  = prices_block['DA' ].to_numpy(),
            price_FCR = prices_block['FCR'].to_numpy(),
        )
        elapsed = time.perf_counter() - t0
        print(f"    Path {path_col} Year {year}: {elapsed:.1f}s")

        year_data = data    .select_dtypes('number').sum()
        year_revs = revenues.select_dtypes('number').sum()
        year_data['objective']       = objective
        year_data['SoC_end_of_year'] = SoC
        year_data['SoC_end_frac']    = SoC / storage.capacity
        year_data['min_SoC']         = min_SoC / storage.capacity
        year_data['max_SoC']         = max_SoC / storage.capacity
        year_data['avg_DA_price']    = prices_block['DA'].mean()
        year_data['avg_FCR_price']   = prices_block['FCR'].mean()

        yearly_data_rows.append({'year': year, **year_data.to_dict()})
        yearly_rev_rows .append({'year': year, **year_revs.to_dict()})

    return pd.DataFrame(yearly_data_rows), pd.DataFrame(yearly_rev_rows)


########################################################################################################################
# Stochastic path optimization
########################################################################################################################

def _configure_storage(storage_cls) -> Storage:
    """Prompt for parameter overrides and instantiate one storage object."""
    defaults = storage_cls()
    attrs = [
        ("capacity",         "Capacity [MWh]",           defaults.capacity),
        ("power",            "Power [MW]",                defaults.power),
        ("efficiency",       "Efficiency [-]",            defaults.efficiency),
        ("initial_soc_frac", "Initial SoC fraction [-]",  defaults.initial_soc / defaults.capacity),
    ]

    print(f"\nDefault parameters for {storage_cls.__name__}:")
    for i, (_, label, val) in enumerate(attrs, start=1):
        print(f"  {i} - {label}: {val}")
    print("  Enter to keep all defaults, or enter a number to edit that parameter.")

    params = {key: val for key, _, val in attrs}
    while True:
        raw = input("Choice (Enter / 1 / 2 / 3 / 4): ").strip()
        if raw == "":
            break
        if raw in ("1", "2", "3", "4"):
            idx = int(raw) - 1
            key, label, _ = attrs[idx]
            new_val = input(f"  New value for {label} [{params[key]}]: ").strip()
            if new_val:
                params[key] = float(new_val)
                print(f"  Set {label} = {params[key]}")
        else:
            print("  Invalid input.")

    storage = storage_cls(
        capacity=params["capacity"],
        power=params["power"],
        efficiency=params["efficiency"],
        initial_soc_frac=params["initial_soc_frac"],
    )
    name = storage_cls.__name__
    print(f"\nRunning with {name}(capacity={storage.capacity}, power={storage.power}, "
          f"efficiency={storage.efficiency}, initial_soc={storage.initial_soc:.2f} MWh)")
    return storage


def _prompt_storage() -> list[Storage]:
    """Interactively ask the user to choose and configure storage type(s)."""
    print("\nSelect storage type:")
    print("  1 - Battery (BESS)")
    print("  2 - Pump (Pumped Storage Hydropower)")
    print("  3 - Both")
    while True:
        choice = input("Enter 1, 2 or 3: ").strip()
        if choice == "1":
            return [_configure_storage(Battery)]
        if choice == "2":
            return [_configure_storage(Pump)]
        if choice == "3":
            return [_configure_storage(Battery), _configure_storage(Pump)]
        print("  Invalid input, please enter 1, 2 or 3.")


def _prompt_input_files() -> tuple[str, str, str]:
    """Ask the user which input files to use.

    Returns (spot_file, fcr_file, label). For numbered choices the files are
    spot_{n}.parquet / FCR_{n}.parquet and the label is the number. For the
    "Other" choice the user supplies both file names and the label is taken
    from the spot file stem.
    """
    while True:
        raw = input("\nSelect input file number (1-5) or 'Other' for custom files: ").strip()
        if raw in ("1", "2", "3", "4", "5"):
            n = int(raw)
            return (
                f"{INPUT_DIR}/spot_{n}.parquet",
                f"{INPUT_DIR}/FCR_{n}.parquet",
                str(n),
            )
        if raw.lower() == "other":
            fcr_name  = input("  Enter FCR file name (.parquet or .csv): ").strip()
            spot_name = input("  Enter spot file name (.parquet or .csv): ").strip()
            if not fcr_name or not spot_name:
                print("  Both file names are required.")
                continue
            label = input("  Enter output file label (trailing suffix on result file): ").strip()
            if not label:
                label = Path(spot_name).stem
            return (
                f"{INPUT_DIR}/{spot_name}",
                f"{INPUT_DIR}/{fcr_name}",
                label,
            )
        print("  Invalid input, please enter a number from 1 to 5 or 'Other'.")


def _read_price_file(path: str) -> pd.DataFrame:
    """Load a price file as a DataFrame, dispatching on extension."""
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{ext}' for {path}. Use .parquet or .csv.")


def _optimize_single_path(args: tuple) -> list[dict]:
    """Top-level worker for parallel execution (must be picklable)."""
    path_col, da_values, fcr_values, years, storage_cfg = args
    print(f"  Optimizing {path_col} ({years} years)...")
    storage_cls, params = storage_cfg
    storage = storage_cls(**params)
    prices = pd.DataFrame({"DA": da_values, "FCR": fcr_values})
    yearly_data, yearly_revenues = _optimize_yearly(storage, prices, path_col, days_in_year=365, years_in_horizon=years)
    rows = []
    for d_row, r_row in zip(yearly_data.itertuples(index=False), yearly_revenues.itertuples(index=False)):
        rows.append({
            "path_id":      path_col,
            "year":         d_row.year,
            "rev_DA":       r_row.rev_DA,
            "rev_FCR":      r_row.rev_FCR,
            "rev_total":    r_row.rev_total,
            "bid_DA_buy":   d_row.bid_DA_buy,
            "bid_DA_sell":  d_row.bid_DA_sell,
            "bid_FCR":      d_row.bid_FCR,
            "avg_DA_price": d_row.avg_DA_price,
            "avg_FCR_price":d_row.avg_FCR_price,
            "SoC_end_frac": d_row.SoC_end_frac,
            "min_SoC":      d_row.min_SoC,
            "max_SoC":      d_row.max_SoC,
        })
    return rows


def _optimize_all_paths(storage: Storage, spot_file: str, fcr_file: str) -> pd.DataFrame:
    """
    Load the given spot and FCR parquet files, pair matching path columns, and
    optimize all paths in parallel.
    Returns a DataFrame with one row per (path_id, year).
    """
    storage_cfg = (
        type(storage),
        {
            "capacity":         storage.capacity,
            "power":            storage.power,
            "efficiency":       storage.efficiency,
            "initial_soc_frac": storage.initial_soc / storage.capacity,
        },
    )

    print(f"Loading spot prices from {spot_file}...")
    df_spot = _read_price_file(spot_file)

    print(f"Loading FCR prices from {fcr_file}...")
    df_fcr = _read_price_file(fcr_file)

    path_cols = [c for c in df_spot.columns if c != "timestamp"]

    n_hours      = len(df_spot)
    NT_year      = 365 * NT
    years        = min(n_hours // NT_year, MAX_YEARS)
    n_hours_used = years * NT_year

    args_list = []
    for path_col in path_cols:
        da_values  = df_spot[path_col].to_numpy()[:n_hours_used]
        # Use matching FCR path if available, otherwise fall back to path_0
        fcr_col    = path_col if path_col in df_fcr.columns else df_fcr.columns[1]
        fcr_values = df_fcr[fcr_col].to_numpy()[:n_hours_used]
        args_list.append((path_col, da_values, fcr_values, years, storage_cfg))

    n_workers = min(len(path_cols), os.cpu_count())
    print(f"Running {len(path_cols)} paths on {n_workers} workers...")
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results_per_path = pool.map(_optimize_single_path, args_list)

    rows = [row for path_rows in results_per_path for row in path_rows]
    return pd.DataFrame(rows)


def run_single_path_test(storage: Storage = None, spot_file: str = None, fcr_file: str = None, label: str = "1") -> None:
    """Load only the first path, optimize it, and save the result."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if storage is None:
        storage = Pump()

    if spot_file is None:
        spot_file = f"{INPUT_DIR}/spot_{label}.parquet"
    if fcr_file is None:
        fcr_file = f"{INPUT_DIR}/FCR_{label}.parquet"

    df_spot = _read_price_file(spot_file)
    df_fcr  = _read_price_file(fcr_file)

    path_col = [c for c in df_spot.columns if c != "timestamp"][0]
    fcr_col  = path_col if path_col in df_fcr.columns else df_fcr.columns[1]

    n_hours = len(df_spot)
    NT_year = 365 * NT
    years   = min(n_hours // NT_year, MAX_YEARS)

    storage_cfg = (
        type(storage),
        {
            "capacity":         storage.capacity,
            "power":            storage.power,
            "efficiency":       storage.efficiency,
            "initial_soc_frac": storage.initial_soc / storage.capacity,
        },
    )

    da_values  = df_spot[path_col].to_numpy()[:years * NT_year]
    fcr_values = df_fcr[fcr_col ].to_numpy()[:years * NT_year]

    print(f"Running single path test: {path_col} ({years} years) with {type(storage).__name__}")
    rows = _optimize_single_path((path_col, da_values, fcr_values, years, storage_cfg))

    result      = pd.DataFrame(rows)
    storage_cls = type(storage).__name__
    output_file = OUTPUT_DIR / f"{storage_cls}_test_single_path_{label}.csv"
    result.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


def main():
    import sys
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    storages                       = _prompt_storage()
    spot_file, fcr_file, label     = _prompt_input_files()

    t_start = time.perf_counter()

    if "--test" in sys.argv:
        for storage in storages:
            run_single_path_test(storage=storage, spot_file=spot_file, fcr_file=fcr_file, label=label)
        elapsed = int(time.perf_counter() - t_start)
        print(f"\nTotal runtime: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")
        return

    for storage in storages:
        storage_cls = type(storage).__name__
        output_file = OUTPUT_DIR / f"{storage_cls}_optimized_results_{label}.csv"

        print(f"\n=== Running {storage_cls} ===")
        results = _optimize_all_paths(storage, spot_file, fcr_file)
        results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

    elapsed = int(time.perf_counter() - t_start)
    print(f"\nTotal runtime: {elapsed // 3600:02d}:{(elapsed % 3600) // 60:02d}:{elapsed % 60:02d}")


if __name__ == "__main__":
    main()