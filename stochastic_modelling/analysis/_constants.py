"""Shared constants for thesis result scripts."""
import os

import pandas as pd

DIR        = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(DIR)
REPO_ROOT  = os.path.dirname(ROOT)

# When STOCH_SCENARIO is set (by run_all.py), per-scenario outputs go to
# data/<scenario>/ and figures/<scenario>/ so multiple pairs don't overwrite.
_SCENARIO = os.environ.get("STOCH_SCENARIO", "").strip()
DATA_OUT  = os.path.join(DIR, "data",    _SCENARIO) if _SCENARIO else os.path.join(DIR, "data")
FIG_OUT   = os.path.join(DIR, "figures", _SCENARIO) if _SCENARIO else os.path.join(DIR, "figures")

# Unit conversions for HISTORICAL data only.
# The new simulation parquet files are already stored in thesis units.
SPOT_SCALE = 10.0   # øre/kWh -> NOK/MWh
FCR_SCALE  = 6.5    # raw simulation/EUR -> NOK/MW

# Historical data
DA_DIR     = os.path.join(REPO_ROOT, "data", "prices", "DA", "DA_cleaned_hourly")
FCR_DIR    = os.path.join(REPO_ROOT, "data", "prices", "FCR", "FCR_cleaned")
SPOT_AREA  = "NO2"

# Simulation parquet files (values already in NOK/MWh and NOK/MW/h).
# STOCH_INPUT_DIR / STOCH_SPOT_FILE / STOCH_FCR_FILE override the defaults so
# run_all.py can point at parquets stored outside the repo.
_SIM_DIR     = os.environ.get("STOCH_INPUT_DIR", os.path.join(DIR, "data"))
SPOT_PARQUET = os.path.join(_SIM_DIR, os.environ.get("STOCH_SPOT_FILE", "spot_1.parquet"))
FCR_PARQUET  = os.path.join(_SIM_DIR, os.environ.get("STOCH_FCR_FILE",  "FCR_1.parquet"))

# Simulation geometry
N_PATHS            = 2000
N_SIM_YEARS        = 70
SPOT_START_YEAR    = 2026
FCR_START_YEAR     = 2026
SPOT_END_YEAR      = SPOT_START_YEAR + N_SIM_YEARS - 1   # 2095
FCR_END_YEAR       = FCR_START_YEAR + N_SIM_YEARS - 1    # 2095

# Units shown in plots
UNIT_SPOT = "NOK/MWh"
UNIT_FCR  = "NOK/MW/h"


def sim_parquet_path(market: str) -> str:
    return SPOT_PARQUET if market == "spot" else FCR_PARQUET


def load_sim_parquet(market: str, columns=None) -> pd.DataFrame:
    """Load the simulation parquet for "spot" or "fcr".

    Returns a DataFrame indexed by timestamp with columns path_0..path_{N-1}.
    Values are already in thesis units (no further scaling required).
    """
    return pd.read_parquet(sim_parquet_path(market), columns=columns)
