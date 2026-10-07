"""
LSMC Engine — computational building blocks for the real options analysis.

This module contains all utility functions needed by the investment decision algorithm:
- Laguerre polynomial basis functions for least-squares regression
- GBM simulation for BESS capital cost paths
- Electricity price path loading from optimiser CSVs
- Revenue estimation (analytical approximation and MILP training scaffold)
- After-tax cash flow computation with Norwegian declining-balance depreciation

All monetary values are in NOK.
"""

# All monetary values are in NOK unless explicitly noted.

from __future__ import annotations

import logging
import sys
from itertools import product as iproduct
from pathlib import Path
from typing import NamedTuple

import numpy as np

from roa.config import Config

# Repository root — used to resolve sibling package paths
_REPO_ROOT = Path(__file__).parent.parent

_log = logging.getLogger(__name__)


# =============================================================================
# === OPTIMIZER DATA LOADER ===
# =============================================================================

class OptimizerData(NamedTuple):
    """Pre-computed annual revenues and mean prices for all simulation paths.

    All arrays have shape (N, T) where N = cfg.N and T = cfg.T_max.
    Axis 0 indexes Monte Carlo paths (path_0 … path_{N-1}).
    Axis 1 indexes years within the evaluation horizon (year 0 … T_max-1).
    All monetary values are in NOK.
    """
    bess_rev_total:  np.ndarray   # BESS annual total revenue            [NOK/year]
    bess_bid_DA_buy: np.ndarray   # BESS annual DA buy volume             [MWh/year]
    psh_rev_total:   np.ndarray   # PSH annual total revenue              [NOK/year]
    psh_rev_DA:      np.ndarray   # PSH annual DA revenue                 [NOK/year]
    psh_bid_DA_buy:  np.ndarray   # PSH annual DA buy (pumped) volume     [MWh/year]
    psh_bid_DA_sell: np.ndarray   # PSH annual DA sell (turbined) volume  [MWh/year]
    mean_DA:         np.ndarray   # Annual mean DA spot price             [NOK/MWh]
    mean_FCR:        np.ndarray   # Annual mean FCR price                 [NOK/MW]


def load_optimizer_data(
    cfg: Config,
    data_dir: Path | str = "roa/input_files",
) -> OptimizerData:
    """Load optimizer outputs into (N, T) arrays with schema auto-detection.

    Reads ``BESS_optimized_results_combined.csv`` and
    ``PSH_optimized_results_combined.csv`` from *data_dir*.

    Two schemas are supported:

      - **Embedded mode**: both BESS and PSH files contain
        ``avg_DA_price`` and ``avg_FCR_price`` columns. The mean prices
        are read directly from each file (and verified bit-identical
        across the two files; mismatch raises ValueError).

      - **Legacy mode**: neither file contains the embedded price
        columns. Mean prices are read from a sibling
        ``yearly_mean_prices.csv`` whose ``path_number`` /
        ``year_number`` columns are translated to ``path_id`` / ``year``.

    Mixed-mode files (one embedded, one legacy) raise ValueError.

    Shape inference:
      - N_data and T_data are read from the file (distinct path count
        and year count). cfg.N and cfg.T_max are populated in place.
      - cfg.N is None → set to N_data; equal to N_data → kept;
        less than N_data → deterministic random subsample
        (rng seed = getattr(cfg, 'seed', 42)); greater → ValueError.
      - cfg.T_max is None → set to T_data; equal → kept;
        less than T_data → first T_max years kept; greater → ValueError.

    After population the loader calls cfg.adjust_T_def_for_horizon() and
    cfg.validate_against_data().

    Parameters
    ----------
    cfg :
        Config instance. ``cfg.N`` and ``cfg.T_max`` may be None
        (auto-infer) or pre-set.
    data_dir :
        Directory that contains the input CSVs. Relative paths are
        resolved from the repository root.

    Returns
    -------
    OptimizerData
        Seven float64 arrays of shape ``(cfg.N, cfg.T_max)``.
    """
    import pandas as pd

    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = _REPO_ROOT / data_dir

    bess_path   = data_dir / "BESS_optimized_results_combined.csv"
    psh_path    = data_dir / "PSH_optimized_results_combined.csv"
    prices_path = data_dir / "yearly_mean_prices.csv"

    # 1. Required-file presence
    for p in (bess_path, psh_path):
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")

    # 2. Schema detection from headers
    bess_cols = pd.read_csv(bess_path, nrows=0).columns.tolist()
    psh_cols  = pd.read_csv(psh_path,  nrows=0).columns.tolist()
    bess_emb = ("avg_DA_price" in bess_cols) and ("avg_FCR_price" in bess_cols)
    psh_emb  = ("avg_DA_price" in psh_cols)  and ("avg_FCR_price" in psh_cols)

    if bess_emb != psh_emb:
        raise ValueError(
            f"Mixed-mode input files: BESS embedded={bess_emb}, "
            f"PSH embedded={psh_emb}. Both files must use the same schema."
        )

    if bess_emb:
        data, N_data, T_data = _load_embedded(cfg, bess_path, psh_path)
        mode_label = "embedded"
    else:
        if not prices_path.exists():
            raise FileNotFoundError(
                f"Legacy mode requires {prices_path} but the file was not found."
            )
        data, N_data, T_data = _load_legacy(cfg, bess_path, psh_path, prices_path)
        mode_label = "legacy"

    # 3. Apply cfg.N / cfg.T_max policy and mutate cfg in place
    seed = getattr(cfg, "seed", 42)
    data = _resolve_shape(cfg, data, N_data, T_data, seed)

    # 4. Adjust T_def to the loaded horizon, then run deferred validation
    cfg.adjust_T_def_for_horizon()
    cfg.validate_against_data()

    print(f"[loader] Mode: {mode_label}. N={cfg.N}, T_max={cfg.T_max} (from data).")

    # 5. Numerical sanity (warnings only — preserved from previous behavior)
    if np.any(data.bess_rev_total < 0):
        _log.warning("bess_rev_total contains negative values (min=%.2f).", data.bess_rev_total.min())
    if np.any(data.psh_rev_total < 0):
        _log.warning("psh_rev_total contains negative values (min=%.2f).", data.psh_rev_total.min())
    if data.psh_rev_DA.shape != data.psh_rev_total.shape:
        raise ValueError(
            f"psh_rev_DA shape {data.psh_rev_DA.shape} does not match "
            f"psh_rev_total shape {data.psh_rev_total.shape}."
        )
    if np.isnan(data.psh_rev_DA).any():
        raise ValueError("psh_rev_DA contains NaN values.")
    if np.any(data.psh_rev_DA < 0):
        raise ValueError(f"psh_rev_DA contains negative values (min={data.psh_rev_DA.min():.2f}).")
    if np.any(data.psh_rev_DA > data.psh_rev_total + 1e-6):
        diff = float((data.psh_rev_DA - data.psh_rev_total).max())
        raise ValueError(f"psh_rev_DA exceeds psh_rev_total beyond tolerance (max diff={diff:.6f}).")
    if np.any(data.mean_DA <= 0):
        _log.warning("mean_DA contains non-positive values.")
    if np.any(data.mean_FCR <= 0):
        _log.warning("mean_FCR contains non-positive values.")

    return data


def _load_embedded(
    cfg: Config,
    bess_path: Path,
    psh_path: Path,
) -> tuple[OptimizerData, int, int]:
    """Load both files in embedded-price mode and verify cross-file consistency."""
    import pandas as pd

    cols_b = ["path_id", "year", "rev_total", "bid_DA_buy", "avg_DA_price", "avg_FCR_price"]
    cols_p = ["path_id", "year", "rev_total", "rev_DA", "bid_DA_buy", "bid_DA_sell",
              "avg_DA_price", "avg_FCR_price"]
    df_b = pd.read_csv(bess_path, usecols=cols_b)
    df_p = pd.read_csv(psh_path,  usecols=cols_p)

    for label, df in [("BESS", df_b), ("PSH", df_p)]:
        n_dup = df.duplicated(subset=["path_id", "year"]).sum()
        if n_dup > 0:
            raise ValueError(f"{label} file has {n_dup} duplicate (path_id, year) rows.")

    # Path-major sort
    for df in (df_b, df_p):
        df["_path_index"] = df["path_id"].str.removeprefix("path_").astype(int)
        df.sort_values(["_path_index", "year"], inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Infer (N, T) from BESS file; verify PSH agrees
    N_data = int(df_b["_path_index"].nunique())
    T_data = int(df_b["year"].nunique())
    expected_rows = N_data * T_data

    for label, df in [("BESS", df_b), ("PSH", df_p)]:
        if len(df) != expected_rows:
            raise ValueError(
                f"{label} file has {len(df)} rows; expected {expected_rows} "
                f"(N_data={N_data} × T_data={T_data})."
            )
        if int(df["_path_index"].nunique()) != N_data:
            raise ValueError(f"{label} path count mismatch (got {df['_path_index'].nunique()}, expected {N_data}).")
        if int(df["year"].nunique()) != T_data:
            raise ValueError(f"{label} year count mismatch (got {df['year'].nunique()}, expected {T_data}).")

    def _arr(df, col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=np.float64).reshape(N_data, T_data)

    bess_mean_DA  = _arr(df_b, "avg_DA_price")
    psh_mean_DA   = _arr(df_p, "avg_DA_price")
    bess_mean_FCR = _arr(df_b, "avg_FCR_price")
    psh_mean_FCR  = _arr(df_p, "avg_FCR_price")

    if not np.array_equal(bess_mean_DA, psh_mean_DA):
        n_mis = int(np.sum(bess_mean_DA != psh_mean_DA))
        raise ValueError(
            f"Embedded-mode cross-file price mismatch: avg_DA_price differs in "
            f"{n_mis} of {bess_mean_DA.size} (path, year) cells. Partner-side bug?"
        )
    if not np.array_equal(bess_mean_FCR, psh_mean_FCR):
        n_mis = int(np.sum(bess_mean_FCR != psh_mean_FCR))
        raise ValueError(
            f"Embedded-mode cross-file price mismatch: avg_FCR_price differs in "
            f"{n_mis} of {bess_mean_FCR.size} (path, year) cells. Partner-side bug?"
        )

    data = OptimizerData(
        bess_rev_total  = _arr(df_b, "rev_total"),
        bess_bid_DA_buy = _arr(df_b, "bid_DA_buy"),
        psh_rev_total   = _arr(df_p, "rev_total"),
        psh_rev_DA      = _arr(df_p, "rev_DA"),
        psh_bid_DA_buy  = _arr(df_p, "bid_DA_buy"),
        psh_bid_DA_sell = _arr(df_p, "bid_DA_sell"),
        mean_DA         = bess_mean_DA,
        mean_FCR        = bess_mean_FCR,
    )
    assert data.psh_rev_DA.shape == data.psh_rev_total.shape
    assert data.psh_bid_DA_sell.shape == data.psh_rev_total.shape
    return data, N_data, T_data


def _load_legacy(
    cfg: Config,
    bess_path: Path,
    psh_path: Path,
    prices_path: Path,
) -> tuple[OptimizerData, int, int]:
    """Load BESS / PSH revenue files plus separate yearly_mean_prices.csv."""
    import pandas as pd

    df_bess = pd.read_csv(bess_path, usecols=["path_id", "year", "rev_total", "bid_DA_buy"])
    df_bess = df_bess.rename(columns={"rev_total": "bess_rev_total", "bid_DA_buy": "bess_bid_DA_buy"})

    psh_cols = pd.read_csv(psh_path, nrows=0).columns.tolist()
    if "bid_DA_sell" not in psh_cols:
        raise NotImplementedError(
            f"Legacy PSH file at {psh_path} is missing 'bid_DA_sell'. "
            "The path-dependent PSH OPEX model (production-scaled components) "
            "requires gross turbined volume per (path, year). Re-run the "
            "dispatch optimizer with bid_DA_sell included, or use the "
            "embedded-mode CSV that already contains it."
        )
    if "rev_DA" not in psh_cols:
        if cfg.grunnrente_mode != "none":
            raise NotImplementedError(
                "Legacy PSH files without rev_DA are unsupported because the "
                "resource-rent base requires DA revenue. "
                "Do not substitute rev_total for rev_DA."
            )
        _log.warning(
            "Legacy PSH file has no rev_DA column; setting psh_rev_DA to zero "
            "because grunnrente_mode='none'."
        )
        df_psh = pd.read_csv(
            psh_path,
            usecols=["path_id", "year", "rev_total", "bid_DA_buy", "bid_DA_sell"],
        )
        df_psh = df_psh.rename(columns={
            "rev_total":   "psh_rev_total",
            "bid_DA_buy":  "psh_bid_DA_buy",
            "bid_DA_sell": "psh_bid_DA_sell",
        })
        df_psh["psh_rev_DA"] = 0.0
    else:
        df_psh = pd.read_csv(
            psh_path,
            usecols=["path_id", "year", "rev_total", "rev_DA", "bid_DA_buy", "bid_DA_sell"],
        )
        df_psh = df_psh.rename(columns={
            "rev_total":   "psh_rev_total",
            "rev_DA":      "psh_rev_DA",
            "bid_DA_buy":  "psh_bid_DA_buy",
            "bid_DA_sell": "psh_bid_DA_sell",
        })

    df_prices = pd.read_csv(
        prices_path,
        usecols=["path_number", "year_number", "avg_spot_price", "avg_FCR_price"],
    )
    df_prices["path_id"] = "path_" + df_prices["path_number"].astype(str)
    df_prices["year"]    = df_prices["year_number"] - 2026
    df_prices = df_prices.drop(columns=["path_number", "year_number"])
    df_prices = df_prices.rename(columns={"avg_spot_price": "mean_DA", "avg_FCR_price": "mean_FCR"})

    for label, df in [("BESS", df_bess), ("PSH", df_psh), ("prices", df_prices)]:
        n_dup = df.duplicated(subset=["path_id", "year"]).sum()
        if n_dup > 0:
            raise ValueError(f"{label} has {n_dup} duplicate (path_id, year) rows.")

    df = (
        df_bess
        .merge(df_psh,    on=["path_id", "year"], how="inner")
        .merge(df_prices, on=["path_id", "year"], how="inner")
    )
    df["_path_index"] = df["path_id"].str.removeprefix("path_").astype(int)
    df = df.sort_values(["_path_index", "year"]).reset_index(drop=True)

    N_data = int(df["_path_index"].nunique())
    T_data = int(df["year"].nunique())
    if len(df) != N_data * T_data:
        raise ValueError(
            f"Legacy join produced {len(df)} rows; expected {N_data * T_data} "
            f"(N_data={N_data} × T_data={T_data}). Check path/year coverage across files."
        )

    def _arr(col: str) -> np.ndarray:
        return df[col].to_numpy(dtype=np.float64).reshape(N_data, T_data)

    data = OptimizerData(
        bess_rev_total  = _arr("bess_rev_total"),
        bess_bid_DA_buy = _arr("bess_bid_DA_buy"),
        psh_rev_total   = _arr("psh_rev_total"),
        psh_rev_DA      = _arr("psh_rev_DA"),
        psh_bid_DA_buy  = _arr("psh_bid_DA_buy"),
        psh_bid_DA_sell = _arr("psh_bid_DA_sell"),
        mean_DA         = _arr("mean_DA"),
        mean_FCR        = _arr("mean_FCR"),
    )
    return data, N_data, T_data


def _resolve_shape(
    cfg: Config,
    data: OptimizerData,
    N_data: int,
    T_data: int,
    seed: int,
) -> OptimizerData:
    """Apply cfg.N / cfg.T_max policy: auto-infer, subsample, truncate, or error.

    Mutates cfg in place: cfg.N and cfg.T_max receive the effective values.
    """
    # ── N policy ──
    if cfg.N is None:
        N_eff = N_data
    elif cfg.N == N_data:
        N_eff = N_data
    elif cfg.N < N_data:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(N_data, cfg.N, replace=False))
        data = OptimizerData(
            bess_rev_total  = data.bess_rev_total[idx],
            bess_bid_DA_buy = data.bess_bid_DA_buy[idx],
            psh_rev_total   = data.psh_rev_total[idx],
            psh_rev_DA      = data.psh_rev_DA[idx],
            psh_bid_DA_buy  = data.psh_bid_DA_buy[idx],
            psh_bid_DA_sell = data.psh_bid_DA_sell[idx],
            mean_DA         = data.mean_DA[idx],
            mean_FCR        = data.mean_FCR[idx],
        )
        N_eff = cfg.N
        print(f"[loader] Subsampled to N={N_eff} from N_data={N_data}.")
    elif getattr(cfg, "bootstrap_upsample", False):
        # Bootstrap with replacement when cfg.N > N_data. Diagnostic-only path
        # for the convergence study; duplicated paths bias R² upward and
        # underestimate seed-to-seed SE relative to a true cfg.N independent
        # sample. Each row in the CSV produced by such a run carries the
        # bootstrap=True flag.
        rng = np.random.default_rng(seed)
        idx = rng.choice(N_data, cfg.N, replace=True)
        data = OptimizerData(
            bess_rev_total  = data.bess_rev_total[idx],
            bess_bid_DA_buy = data.bess_bid_DA_buy[idx],
            psh_rev_total   = data.psh_rev_total[idx],
            psh_rev_DA      = data.psh_rev_DA[idx],
            psh_bid_DA_buy  = data.psh_bid_DA_buy[idx],
            psh_bid_DA_sell = data.psh_bid_DA_sell[idx],
            mean_DA         = data.mean_DA[idx],
            mean_FCR        = data.mean_FCR[idx],
        )
        N_eff = cfg.N
        print(f"[loader] BOOTSTRAP-UPSAMPLED with replacement to N={N_eff} "
              f"from N_data={N_data}. R² and SE are biased.")
    else:
        raise ValueError(
            f"cfg.N={cfg.N} exceeds available paths in data (N_data={N_data}). "
            "Set cfg.bootstrap_upsample=True for a diagnostic bootstrap path."
        )

    # ── T_max policy ──
    if cfg.T_max is None:
        T_eff = T_data
    elif cfg.T_max == T_data:
        T_eff = T_data
    elif cfg.T_max < T_data:
        T_eff = cfg.T_max
        data = OptimizerData(
            bess_rev_total  = data.bess_rev_total[:, :T_eff],
            bess_bid_DA_buy = data.bess_bid_DA_buy[:, :T_eff],
            psh_rev_total   = data.psh_rev_total[:, :T_eff],
            psh_rev_DA      = data.psh_rev_DA[:, :T_eff],
            psh_bid_DA_buy  = data.psh_bid_DA_buy[:, :T_eff],
            psh_bid_DA_sell = data.psh_bid_DA_sell[:, :T_eff],
            mean_DA         = data.mean_DA[:, :T_eff],
            mean_FCR        = data.mean_FCR[:, :T_eff],
        )
        print(f"[loader] Truncated to T_max={T_eff} from T_data={T_data}.")
    else:
        raise ValueError(
            f"cfg.T_max={cfg.T_max} exceeds available years in data (T_data={T_data})."
        )

    cfg.N     = N_eff
    cfg.T_max = T_eff
    return data


# =============================================================================
# === LAGUERRE BASIS FUNCTIONS AND REGRESSION ===
# =============================================================================
#
# This section implements the weighted Laguerre polynomial regression basis used
# by the Longstaff-Schwartz (2001) Monte Carlo method, as extended to multi-
# dimensional state spaces by Areal, Rodrigues & Armada (2008). At each backward-
# induction step, a least-squares regression of discounted future values onto these
# basis functions produces an estimated continuation value for every simulated path,
# allowing the algorithm to decide whether to exercise now or wait.


def normalize_state(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Map state variable x to [0, ∞) by dividing by its median.

    The Laguerre weight exp(-x/2) collapses to zero for large x and diverges
    near zero, so normalisation keeps values in a numerically safe range.
    Dividing by the median is robust against outliers in the price/capex paths.

    Parameters
    ----------
    x : 1D array of non-negative state values.

    Returns
    -------
    x_norm : normalised array (median ≈ 1 after normalisation)
    scale  : the median used, so the caller can invert or reuse it.
    """
    scale = float(np.median(x))
    if scale == 0.0:
        # Fallback: avoid division by zero when all paths are at zero
        return x.astype(float), 1.0
    return x / scale, scale


def weighted_laguerre(x: np.ndarray, degree: int) -> np.ndarray:
    """Compute weighted Laguerre polynomial basis up to `degree`.

    Evaluates the first (degree+1) weighted Laguerre functions:
        phi_k(x) = L_k(x) * exp(-x/2)

    where L_k is the k-th Laguerre polynomial:
        L_0(x) = 1
        L_1(x) = 1 - x
        L_2(x) = 1 - 2x + x^2/2
        L_3(x) = 1 - 3x + 3x^2/2 - x^3/6
        L_4(x) = 1 - 4x + 3x^2 - (2/3)x^3 + x^4/24

    The caller is responsible for normalising x to be non-negative (see
    normalize_state). Negative inputs will produce a misleading weight.

    Parameters
    ----------
    x      : 1D array of shape (n,) with non-negative values.
    degree : maximum polynomial degree (inclusive). Supports 0-4.

    Returns
    -------
    Phi : 2D array of shape (n, degree+1), column k contains phi_k(x).

    Raises
    ------
    ValueError if degree > 4 (higher-order terms not implemented).
    """
    if degree > 4:
        raise ValueError(f"degree must be <= 4, got {degree}.")

    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    w = np.exp(-x / 2.0)            # common exponential weight

    Phi = np.empty((n, degree + 1), dtype=float)

    if degree >= 0:
        Phi[:, 0] = w                                                       # L_0 = 1
    if degree >= 1:
        Phi[:, 1] = w * (1.0 - x)                                           # L_1
    if degree >= 2:
        Phi[:, 2] = w * (1.0 - 2.0 * x + 0.5 * x**2)                       # L_2
    if degree >= 3:
        Phi[:, 3] = w * (1.0 - 3.0 * x + 1.5 * x**2 - x**3 / 6.0)           # L_3
    if degree >= 4:
        Phi[:, 4] = w * (1.0 - 4.0 * x + 3.0 * x**2
                         - (2.0 / 3.0) * x**3 + x**4 / 24.0)                # L_4

    return Phi


def build_basis_matrix(
    state_vars: list[np.ndarray],
    degree: int,
) -> np.ndarray:
    """Build the full LSMC regression matrix from one or more state variables.

    For a single state variable, the basis is [phi_0, phi_1, ..., phi_degree].
    For two state variables (x1, x2), the basis consists of all products
        phi_j(x1) * phi_k(x2)   where  j + k <= degree,
    ordered by total degree then by j. The constant term (j=0, k=0) is
    always included as the first column.

    Each state variable is independently normalised before basis evaluation
    so that the exponential weighting operates in a numerically safe range.

    Parameters
    ----------
    state_vars : list of 1D arrays, each of length n_samples. One array per
                 state variable (e.g., [price_paths[:, t], capex_paths[:, t]]).
    degree     : maximum total polynomial degree (inclusive).

    Returns
    -------
    B : 2D array of shape (n_samples, n_basis_functions).

    Raises
    ------
    ValueError if state_vars is empty or arrays have inconsistent lengths.
    """
    if len(state_vars) == 0:
        raise ValueError("state_vars must contain at least one array.")

    n = len(state_vars[0])
    for i, sv in enumerate(state_vars[1:], start=1):
        if len(sv) != n:
            raise ValueError(
                f"state_vars[{i}] has length {len(sv)}, expected {n}."
            )

    # Normalise and compute per-variable Laguerre matrices
    phi_list = []
    for sv in state_vars:
        x_norm, _ = normalize_state(np.asarray(sv, dtype=float))
        phi_list.append(weighted_laguerre(x_norm, degree))

    if len(phi_list) == 1:
        # Single state variable — basis is just the Laguerre columns
        return phi_list[0]

    # Multiple state variables — tensor product with total-degree constraint
    columns = []
    if len(phi_list) == 2:
        # Common 2-variable case: all (j, k) with j+k <= degree
        phi0, phi1 = phi_list
        for total in range(degree + 1):
            for j in range(total + 1):
                k = total - j
                columns.append(phi0[:, j] * phi1[:, k])
    else:
        # General M-variable case: iterate over all index tuples summing to <= degree
        n_vars = len(phi_list)
        for indices in iproduct(range(degree + 1), repeat=n_vars):
            if sum(indices) <= degree:
                col = phi_list[0][:, indices[0]]
                for v in range(1, n_vars):
                    col = col * phi_list[v][:, indices[v]]
                columns.append(col)

    return np.column_stack(columns)


def fit_continuation(
    basis_matrix: np.ndarray,
    future_values: np.ndarray,
    itm_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Estimate continuation values via OLS on the basis matrix.

    Optionally restricts the fit to in-the-money paths (Longstaff-Schwartz
    filter) but returns fitted values for ALL paths so backward induction
    can compare exercise vs. continuation on every path.

    Parameters
    ----------
    basis_matrix  : shape (n_samples, n_basis_functions)
    future_values : shape (n_samples,) -- discounted values one step ahead
    itm_mask      : boolean array of shape (n_samples,); if provided, only
                    paths where itm_mask is True are used for fitting.

    Returns
    -------
    continuation : 1D array of shape (n_samples,) -- fitted values for all paths.
    """
    if itm_mask is not None:
        B_fit = basis_matrix[itm_mask]
        y_fit = future_values[itm_mask]
    else:
        B_fit = basis_matrix
        y_fit = future_values

    coeffs, _, _, _ = np.linalg.lstsq(B_fit, y_fit, rcond=None)
    assert not np.any(np.isnan(coeffs)), "C2f: NaN in LSMC regression coefficients"
    return basis_matrix @ coeffs


# =============================================================================
# === BESS CAPEX SIMULATION (GBM) ===
# =============================================================================
#
# This section simulates stochastic paths for BESS (Battery Energy Storage System)
# capital costs using Geometric Brownian Motion (GBM). The GBM model captures the
# empirically observed declining trend and year-to-year uncertainty in battery
# costs. Each simulated path represents one possible trajectory of NOK/MWh costs
# over the investment horizon, which directly determines whether investing in BESS
# is profitable in a given Monte Carlo scenario.


def simulate_bess_capex(config: Config, seed: int = 42) -> np.ndarray:
    """Simulate GBM paths for BESS unit CAPEX (NOK/MWh).

    Uses the standard log-normal GBM discretisation:

        K[i, t+1] = K[i, t] * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    where Z ~ N(0,1) i.i.d.  Column 0 of the output is the deterministic
    initial value config.gbm_K0 (all paths start there).

    Parameters
    ----------
    config : Config instance supplying gbm_K0, gbm_mu, gbm_sigma, dt,
             T_max, and N.
    seed   : numpy random seed for reproducibility.

    Returns
    -------
    K : array of shape (N, T_max + 1)
        K[:, 0]  = config.gbm_K0  (initial value, NOK/MWh, no randomness)
        K[:, t]  = BESS CAPEX at year t (NOK/MWh)
    """
    rng = np.random.default_rng(seed)

    mu    = config.gbm_mu
    sigma = config.gbm_sigma
    dt    = config.dt
    K0    = config.gbm_K0          # NOK/MWh -- derived from Aa Energi quote
    N     = config.N
    steps = config.T_max            # number of increments; output has T_max+1 columns

    drift = (mu - 0.5 * sigma**2) * dt
    vol   = sigma * np.sqrt(dt)

    # Draw all standard normals at once for efficiency
    Z = rng.standard_normal((N, steps))

    # Log-increments, then cumulative sum to get the log-price path
    log_increments = drift + vol * Z                   # shape (N, steps)
    log_path = np.cumsum(log_increments, axis=1)       # shape (N, steps)

    # Prepend column of zeros (t=0 -> K0 * exp(0) = K0)
    log_path = np.hstack([np.zeros((N, 1)), log_path]) # shape (N, T_max+1)

    return K0 * np.exp(log_path)


# =============================================================================
# === OPTIMIZER-OUTPUT CASH FLOW INPUTS ===
# =============================================================================
#
# Revenue and charged-energy matrices come straight from the dispatch optimizer
# output (OptimizerData), with linear capacity scaling applied for BESS only.
# The optimizer runs at the 4.6 MW / 10 MWh BESS reference unit
# (cfg.capex_bess_baseline_mwh); PSH values are passed through unscaled.
# All arrays returned have shape (N, T_max).


def precompute_revenue_matrix(
    cfg: Config,
    data: OptimizerData,
    technology: str,
    q: float | None,
) -> np.ndarray:
    """Annual gross revenue (NOK) by direct lookup of optimizer output.

    Parameters
    ----------
    cfg        : Config — used for BESS reference capacity.
    data       : OptimizerData — provides per-(path, year) revenue arrays.
    technology : "BESS" or "PSH".
    q          : BESS capacity (MWh); scaled by q / cfg.capex_bess_baseline_mwh.
                 Pass None for PSH.

    Returns
    -------
    (N, T_max) float64 array. Optimizer values are returned as-produced; no clipping.
    """
    tech = technology.upper()
    if tech == "BESS":
        if q is None:
            raise ValueError("BESS revenue lookup requires q (capacity in MWh).")
        return data.bess_rev_total * (q / cfg.capex_bess_baseline_mwh)
    if tech == "PSH":
        return data.psh_rev_total
    raise ValueError(f"Unknown technology {technology!r}; expected 'BESS' or 'PSH'.")


def precompute_psh_da_revenue_matrix(data: OptimizerData) -> np.ndarray:
    """Annual PSH DA revenue (NOK) by direct lookup; used for the grunnrente base."""
    return data.psh_rev_DA


def precompute_charged_energy_matrix(
    cfg: Config,
    data: OptimizerData,
    technology: str,
    q: float | None,
) -> np.ndarray:
    """Annual DA-buy charged energy (MWh/year) by direct lookup of optimizer output.

    Parameters
    ----------
    cfg        : Config — used for BESS reference capacity.
    data       : OptimizerData.
    technology : "BESS" or "PSH".
    q          : BESS capacity (MWh); scaled by q / cfg.capex_bess_baseline_mwh.
                 Pass None for PSH.

    Returns
    -------
    (N, T_max) float64 array. Optimizer values are returned as-produced; no clipping.
    """
    tech = technology.upper()
    if tech == "BESS":
        if q is None:
            raise ValueError("BESS charged-energy lookup requires q (capacity in MWh).")
        return data.bess_bid_DA_buy * (q / cfg.capex_bess_baseline_mwh)
    if tech == "PSH":
        return data.psh_bid_DA_buy
    raise ValueError(f"Unknown technology {technology!r}; expected 'BESS' or 'PSH'.")


def build_opex_matrix(
    cfg: Config,
    charge_matrix: np.ndarray,
    technology: str,
    q: float | None,
    production_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Build the annual OPEX matrix (NOK/year).

    BESS:
        opex_bess_per_mwh_year * q  +  elavgift_nok_per_mwh_bess * charge_matrix

    PSH:
        psh_capacity_opex_nok                              # constant capacity-scaled
      + elavgift_nok_per_mwh_psh   * charge_matrix         # per-(i,t) on bid_DA_buy
      + psh_production_opex_per_mwh * production_matrix    # per-(i,t) on bid_DA_sell

    Parameters
    ----------
    cfg               : Config.
    charge_matrix     : (N, T_max) annual charged energy (MWh/year) from
                        precompute_charged_energy_matrix. For BESS this is
                        bid_DA_buy at the BESS reference unit, scaled by q;
                        for PSH this is the PSH bid_DA_buy from the dispatch optimizer.
    technology        : "BESS" or "PSH".
    q                 : BESS capacity (MWh). Ignored for PSH.
    production_matrix : (N, T_max) annual gross turbined volume (MWh/year),
                        i.e. bid_DA_sell from the dispatch optimizer output.
                        Required for PSH — applied to the production-scaled
                        OPEX components (property tax, fixed grid, system
                        grid, concession fees) at psh_production_opex_per_mwh
                        NOK/MWh. Ignored for BESS.

    Returns
    -------
    (N, T_max) float64 array of annual OPEX (NOK/year).
    """
    if technology.upper() == "BESS":
        elavgift_component = cfg.elavgift_nok_per_mwh_bess * charge_matrix
        fixed = cfg.opex_bess_per_mwh_year * q
        return fixed + elavgift_component

    if production_matrix is None:
        raise ValueError(
            "PSH OPEX requires production_matrix (bid_DA_sell) for the "
            "production-scaled OPEX components (property tax, fixed grid, "
            "system grid, concession fees)."
        )
    elavgift_component  = cfg.elavgift_nok_per_mwh_psh   * charge_matrix
    production_component = cfg.psh_production_opex_per_mwh * production_matrix
    fixed = cfg.psh_capacity_opex_nok
    return fixed + elavgift_component + production_component


# =============================================================================
# === AFTER-TAX CASH FLOWS ===
# =============================================================================
#
# This section computes after-tax net cash flows (NCF) for storage investments
# under Norwegian tax rules. The key feature is saldoavskrivning (declining-balance
# depreciation): each year the investor deducts a fixed fraction of the remaining
# book value, generating a tax shield that reduces the effective cost of capital.
# Functions in this section produce the per-path, per-year NCF arrays consumed by
# Phase 1 (BESS compound chain) and Phase 2 (joint optimisation). All values in NOK.


def compute_ncf(
    revenue: np.ndarray,
    opex_annual: np.ndarray | float,
    capex: float | np.ndarray,
    config: Config,
    operating_years: int,
    dep_charge_schedule: np.ndarray | None = None,
) -> np.ndarray:
    """Compute after-tax net cash flows for a single investment.

    For each path i and operating year k (0-indexed):
        taxable_income_ik = revenue[i, k] - opex_annual[i, k]
        dep_charge_ik     = (caller-provided schedule, or default decay × capex)
        tax_ik            = tax_rate * max(taxable_income_ik - dep_charge_ik, 0)
        ncf[i, k]         = taxable_income_ik - tax_ik

    No negative tax: losses generate zero tax (no carry-forward in this model).

    Parameters
    ----------
    revenue             : array of shape (N, T) -- gross annual revenue (NOK).
    opex_annual         : scalar, shape (T,), or shape (N, T) -- annual OPEX (NOK).
                          When shape (N, T), each path has its own OPEX trajectory
                          (e.g. from build_opex_matrix). numpy broadcasts all three forms.
    capex               : scalar or shape (N,) -- CAPEX at exercise (NOK).
                          Used only when dep_charge_schedule is None.
    config              : Config instance; uses tax_rate; uses dep_rate when
                          dep_charge_schedule is None.
    operating_years     : number of years this asset operates (T_bess or T_psh).
                          Must be <= revenue.shape[1].
    dep_charge_schedule : optional shape (operating_years,) or (N, operating_years)
                          array of annual depreciation charges (NOK). When
                          provided, overrides the default declining-balance
                          schedule. Used by PSH for the component-wise §18-6
                          linear schedule (see build_psh_depreciation_schedule).
                          The BESS path passes None and uses the default
                          declining-balance schedule per §14-43 group d.

    Returns
    -------
    ncf : array of shape (N, operating_years) -- after-tax NCF per path per year (NOK).
    """
    revenue = np.asarray(revenue, dtype=float)

    # Slice to operating window
    rev = revenue[:, :operating_years]                     # (N, L)

    # OPEX broadcast to match the operating window
    opex = np.asarray(opex_annual, dtype=float)
    if opex.ndim == 0:
        opex = float(opex)                                  # scalar stays scalar
    elif opex.ndim == 1:
        opex = opex[:operating_years]                       # (L,)
    else:
        opex = opex[:, :operating_years]                   # (N, L)

    taxable_income = rev - opex                             # (N, L)

    if dep_charge_schedule is None:
        # Default: declining-balance schedule from config.dep_rate × capex (BESS).
        k = np.arange(operating_years, dtype=float)
        decay = config.dep_rate * (1.0 - config.dep_rate) ** k  # (L,)

        capex_arr = np.asarray(capex, dtype=float)
        if capex_arr.ndim == 0:
            dep_charge = float(capex_arr) * decay               # (L,)
        else:
            dep_charge = capex_arr[:, np.newaxis] * decay[np.newaxis, :]  # (N, L)
    else:
        # Caller-provided schedule (e.g. PSH §18-6 component-wise linear).
        sched = np.asarray(dep_charge_schedule, dtype=float)
        if sched.ndim == 1:
            if sched.shape != (operating_years,):
                raise ValueError(
                    f"dep_charge_schedule shape {sched.shape}, "
                    f"expected ({operating_years},)."
                )
            dep_charge = sched
        elif sched.ndim == 2:
            if sched.shape != (rev.shape[0], operating_years):
                raise ValueError(
                    f"dep_charge_schedule shape {sched.shape}, "
                    f"expected ({rev.shape[0]}, {operating_years})."
                )
            dep_charge = sched
        else:
            raise ValueError(
                f"dep_charge_schedule must be 1D or 2D, got ndim={sched.ndim}."
            )

    tax = config.tax_rate * np.maximum(taxable_income - dep_charge, 0.0)

    return taxable_income - tax


def build_psh_depreciation_schedule(
    config: Config,
    operating_years: int,
) -> np.ndarray:
    """Per-operating-year linear PSH depreciation charge (NOK), Skatteloven §18-6.

    Component-wise linear depreciation. Each component contributes
    rate × CAPEX_component for as long as k ≤ life_component, then
    drops to zero.

    Parameters
    ----------
    config          : Config instance.
    operating_years : length L of the schedule. Operating year k = k_idx + 1
                      maps to schedule[k_idx] for k_idx ∈ {0, ..., L-1}.

    Returns
    -------
    schedule : (L,) float64 array of NOK/year deductions.

    Default rates and lives (active concept):
      Civil works (bygg):       1.5%/yr × CAPEX_civil for 67 years
      Electromechanical (elek): 2.5%/yr × CAPEX_elek  for 40 years
      Other (andre):            2.5%/yr × CAPEX_andre for 40 years
    At T_psh = 67 all components are fully depreciated (civil: 1.5% × 67 = 100.5%
    → capped at 100%; elek/andre: fully written off at year 40). Terminal civil
    book value is zero; compute_psh_terminal_value returns 0.
    """
    concept = config.psh_concepts[config.psh_concept - 1]
    capex_civil = concept.capex_bygg_nok_per_mw  * concept.psh_mw
    capex_elek  = concept.capex_elek_nok_per_mw  * concept.psh_mw
    capex_andre = concept.capex_andre_nok_per_mw * concept.psh_mw

    schedule = np.zeros(operating_years, dtype=float)
    for k_idx in range(operating_years):
        k = k_idx + 1
        d_k = 0.0
        if k <= config.psh_dep_life_civil:
            d_k += config.psh_dep_rate_civil * capex_civil
        if k <= config.psh_dep_life_elek:
            d_k += config.psh_dep_rate_elek * capex_elek
        if k <= config.psh_dep_life_andre:
            d_k += config.psh_dep_rate_andre * capex_andre
        schedule[k_idx] = d_k
    return schedule


def compute_psh_terminal_value(config: Config) -> float:
    """Year-T_psh terminal cash flow for PSH (NOK), post-tax.

    Sale of the asset at the end of the operating life. The civil-works
    component still has undepreciated tax book value at T_psh under the
    §18-6 schedule (1.5%/yr × 67 yr → 40% remains at k=40). The elek and
    andre components are fully written off and do not contribute.

    At psh_terminal_value_multiplier == 1.0:
        Sale at tax book value; realized gain is zero by construction,
        so post-tax cash flow equals pre-tax. No §5-30 gain-on-disposal tax.
    At multiplier > 1.0:
        gain          = (mult - 1) × book
        tax_on_gain   = tax_rate × max(gain, 0)
        TV_post_tax   = mult × book − tax_on_gain
    """
    concept = config.psh_concepts[config.psh_concept - 1]
    capex_civil = concept.capex_bygg_nok_per_mw * concept.psh_mw

    cum_dep_frac = min(config.T_psh * config.psh_dep_rate_civil, 1.0)
    book = capex_civil * (1.0 - cum_dep_frac)

    mult = config.psh_terminal_value_multiplier
    pre_tax     = mult * book
    gain        = (mult - 1.0) * book
    tax_on_gain = config.tax_rate * max(gain, 0.0)
    return pre_tax - tax_on_gain


def compute_grunnrente_operating_tax_slice(
    config: Config,
    rev_DA_slice: np.ndarray,
    rev_total_slice: np.ndarray,
    opex_total_slice: np.ndarray,
    L: int,
) -> np.ndarray:
    """Annual operating grunnrente tax for PSH.

    CAPEX is not deducted here. The post-2021 immediate investment deduction is
    handled separately at exercise time by compute_grunnrente_capex_credit().
    """
    rev_DA = np.asarray(rev_DA_slice, dtype=float)[:, :L]
    if config.grunnrente_mode == "none":
        return np.zeros_like(rev_DA)

    if config.psh_opex_allocation_for_grunnrente != "revenue_share":
        raise ValueError(
            "Only revenue_share OPEX allocation is implemented for grunnrente."
        )

    rev_total = np.asarray(rev_total_slice, dtype=float)[:, :L]
    opex_total = np.asarray(opex_total_slice, dtype=float)[:, :L]

    share = np.divide(
        rev_DA,
        rev_total,
        out=np.zeros_like(rev_DA),
        where=np.abs(rev_total) > 1e-12,
    )
    share = np.clip(share, 0.0, 1.0)
    opex_DA = opex_total * share
    gm_DA = rev_DA - opex_DA

    if config.grunnrente_mode == "symmetric":
        taxable = gm_DA
    elif config.grunnrente_mode == "truncated":
        taxable = np.maximum(gm_DA, 0.0)
    else:
        raise ValueError(f"Unknown grunnrente_mode: {config.grunnrente_mode!r}")

    return config.grunnrente_effective_rate * taxable


def compute_grunnrente_capex_credit(config: Config, capex_psh: float) -> float:
    """Exercise-time CAPEX tax credit for PSH, discount factor 1.

    Truncated mode treats negative grunnrente as zero and intentionally omits
    carry-forward/refund. Symmetric mode values the negative GR tax shield
    immediately at exercise.
    """
    if config.grunnrente_mode == "symmetric":
        return config.grunnrente_effective_rate * capex_psh
    if config.grunnrente_mode in {"none", "truncated"}:
        return 0.0
    raise ValueError(f"Unknown grunnrente_mode: {config.grunnrente_mode!r}")


def compute_opnpv(
    ncf: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Compute the NPV of operating cash flows.

    Cash flows are discounted as if the first operating year's cash flow
    arrives one period after exercise:

        OpNPV_i = sum_{k=0}^{L-1}  delta^{k+1} * ncf[i, k]

    Parameters
    ----------
    ncf    : array of shape (N, L) -- after-tax NCFs for L operating years (NOK).
    config : Config instance; uses delta.

    Returns
    -------
    opnpv : 1D array of shape (N,) -- NPV for each path (NOK).
    """
    ncf = np.asarray(ncf, dtype=float)
    L = ncf.shape[1]
    k = np.arange(1, L + 1, dtype=float)                  # [1, 2, ..., L]
    discount = config.delta ** k                            # (L,)
    return ncf @ discount                                   # (N,)


def compute_psh_exercise_payoff(
    config: Config,
    revenue_slice: np.ndarray,
    revenue_DA_slice: np.ndarray,
    opex_slice: np.ndarray,
    year_idx: int,
) -> np.ndarray:
    """Compute the immediate payoff from exercising the PSH deferral option.

        T_operating   = revenue_slice.shape[1]   (caller pre-slices to correct length)
        total_capex   = config.capex_psh         (property, reads active concept, NOK)
        dep_schedule  = build_psh_depreciation_schedule(config, T_operating)  (§18-6)
        ncf           = compute_ncf(..., dep_charge_schedule=dep_schedule)
        gr_tax        = compute_grunnrente_operating_tax_slice(...)
        ncf_after_gr  = ncf − gr_tax
        if T_operating == T_psh:
            ncf_after_gr[:, -1] += compute_psh_terminal_value(config)  # year-T_psh sale
        opnpv         = compute_opnpv(ncf_after_gr, config)
        payoff[i]     = -total_capex + gr_capex_credit + opnpv[i]  (NOK)

    Parameters
    ----------
    config        : Config instance; reads capex_psh.
    revenue_slice : array of shape (N, L) -- gross annual total revenues (NOK).
    revenue_DA_slice : array of shape (N, L) -- gross annual DA revenues (NOK).
    opex_slice    : array of shape (N, L) -- annual OPEX from build_opex_matrix (NOK).
    year_idx      : int -- year of exercise (used for logging/debugging only).

    Returns
    -------
    payoff : 1D array of shape (N,) -- net payoff per path (NOK).
    """
    T_operating = revenue_slice.shape[1]
    total_capex = config.capex_psh                          # scalar NOK

    dep_schedule = build_psh_depreciation_schedule(config, T_operating)

    ncf_before_gr = compute_ncf(
        revenue=revenue_slice,
        opex_annual=opex_slice,
        capex=total_capex,
        config=config,
        operating_years=T_operating,
        dep_charge_schedule=dep_schedule,
    )
    gr_tax_op_slice = compute_grunnrente_operating_tax_slice(
        config,
        revenue_DA_slice,
        revenue_slice,
        opex_slice,
        T_operating,
    )
    ncf_after_gr = ncf_before_gr - gr_tax_op_slice

    # Year-T_psh terminal cash flow: sale of undepreciated civil-works book.
    # Sale at tax book value; realized gain is zero by construction (mult=1.0),
    # so post-tax cash flow equals pre-tax. Only paid for the full operating life;
    # truncated final cycles receive zero residual value per the methodology.
    if T_operating == config.T_psh:
        tv_post_tax = compute_psh_terminal_value(config)
        ncf_after_gr = ncf_after_gr.copy()
        ncf_after_gr[:, -1] = ncf_after_gr[:, -1] + tv_post_tax

    opnpv = compute_opnpv(ncf_after_gr, config)
    gr_capex_credit = compute_grunnrente_capex_credit(config, total_capex)
    return -total_capex + gr_capex_credit + opnpv


def compute_psh_cashflow_components(
    config: Config,
    revenue_slice: np.ndarray,
    revenue_DA_slice: np.ndarray,
    opex_slice: np.ndarray,
    year_idx: int,
) -> dict:
    """Per-path PSH cash-flow components, the undiscounted twin of compute_psh_exercise_payoff.

    Reuses the EXACT same GM / §18-6 depreciation / ordinary-tax / grunnrente
    logic as compute_psh_exercise_payoff, but returns the raw NOK cash-flow
    schedule instead of the discounted scalar, so that static invest-now metrics
    (NPV/IRR/ROI/PI/MIRR) can be built on it.

    The discounting convention matches compute_psh_exercise_payoff: operating
    year k_idx (0-indexed in the NCF array) is the calendar period k = k_idx + 1.
    CAPEX and the grunnrente CAPEX credit fall in calendar period 0. The terminal
    civil-works sale (only for a full T_psh life) is folded into F[T_psh].

    Parameters
    ----------
    config           : Config instance; reads capex_psh, delta, T_psh.
    revenue_slice    : (N, L) gross annual total revenues (NOK).
    revenue_DA_slice : (N, L) gross annual DA revenues (NOK).
    opex_slice       : (N, L) annual OPEX from build_opex_matrix (NOK).
    year_idx         : exercise year (passed through to compute_psh_exercise_payoff
                       for the consistency assertion; otherwise diagnostic only).

    Returns
    -------
    dict with two (N, L+1) float64 arrays (L = revenue_slice.shape[1]):
        "I" : capital OUTFLOWS. I[:, 0] = C^PSH (total CAPEX), else 0.
        "F" : after-tax operating INFLOWS. F[:, 0] = C^{GR,CAPEX} (the grunnrente
              CAPEX credit; 0 unless grunnrente_mode == "symmetric").
              F[:, k] = NCF^PSH_{i, year_idx, k} for k = 1..L (terminal civil-works
              book sale folded into F[:, L] when L == T_psh).

    The function asserts sum_k delta^k (F[k] - I[k]) == compute_psh_exercise_payoff(...)
    per path (rtol < 1e-6).
    """
    T_operating = revenue_slice.shape[1]
    N = revenue_slice.shape[0]
    total_capex = config.capex_psh                          # scalar NOK

    dep_schedule = build_psh_depreciation_schedule(config, T_operating)

    ncf_before_gr = compute_ncf(
        revenue=revenue_slice,
        opex_annual=opex_slice,
        capex=total_capex,
        config=config,
        operating_years=T_operating,
        dep_charge_schedule=dep_schedule,
    )
    gr_tax_op_slice = compute_grunnrente_operating_tax_slice(
        config,
        revenue_DA_slice,
        revenue_slice,
        opex_slice,
        T_operating,
    )
    ncf_after_gr = ncf_before_gr - gr_tax_op_slice

    # Year-T_psh terminal cash flow folded into the final operating year, exactly
    # as in compute_psh_exercise_payoff. Only for a full operating life.
    if T_operating == config.T_psh:
        tv_post_tax = compute_psh_terminal_value(config)
        ncf_after_gr = ncf_after_gr.copy()
        ncf_after_gr[:, -1] = ncf_after_gr[:, -1] + tv_post_tax

    gr_capex_credit = compute_grunnrente_capex_credit(config, total_capex)

    I = np.zeros((N, T_operating + 1), dtype=float)
    F = np.zeros((N, T_operating + 1), dtype=float)
    I[:, 0] = total_capex
    F[:, 0] = gr_capex_credit
    F[:, 1:] = ncf_after_gr                                  # period k=1..T maps to ncf[:, k-1]

    # VERIFY: discounted net stream reproduces the scalar payoff per path (rel < 1e-9).
    disc = config.delta ** np.arange(T_operating + 1, dtype=float)
    npv_components = (F - I) @ disc
    payoff_ref = compute_psh_exercise_payoff(
        config, revenue_slice, revenue_DA_slice, opex_slice, year_idx,
    )
    rel = np.abs(npv_components - payoff_ref) / np.maximum(np.abs(payoff_ref), 1.0)
    assert np.all(rel < 1e-9), (
        "compute_psh_cashflow_components: discounted net stream does not match "
        f"compute_psh_exercise_payoff (max rel diff {float(rel.max()):.3e})."
    )

    return {"I": I, "F": F}
