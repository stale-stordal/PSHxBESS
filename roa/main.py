"""roa/main.py — K-seed averaged ROA pipeline (production headline).

Default: K=30 BESS-CAPEX seeds, N=10,000 paths, canonical Concept-1 PSH.

Single-seed quick check: ``python -m roa.main --K 1``.

The K=30 averaging is the production headline V_0^stop reported in the
thesis (§5.3). Mathea's ``run_roa_sensitivity.py`` script imports
``run_pipeline`` from this module as a single-seed entry point and
monkeypatches ``_INPUT_DIR`` / ``load_optimizer_data`` — both are preserved
at module level for that purpose.

All monetary values in NOK.

Usage
-----
    python -m roa.main                        # K=30 production headline
    python -m roa.main --K 1                  # single-seed quick check (~30s)
    python -m roa.main --K 30 --out runs/X    # custom output dir
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from roa.config import (
    Config,
    ELAVGIFT_SCENARIO_VALUES,
    SCENARIO_DEFINITIONS,
)
from roa.investment import (
    compute_psh_payoffs,
    phase1_bess_compound,
    phase2_joint_optimization,
)
from roa.lsmc import (
    OptimizerData,
    build_opex_matrix,
    build_psh_depreciation_schedule,
    compute_grunnrente_capex_credit,
    compute_grunnrente_operating_tax_slice,
    compute_opnpv,
    compute_psh_terminal_value,
    load_optimizer_data,
    precompute_charged_energy_matrix,
    precompute_psh_da_revenue_matrix,
    precompute_revenue_matrix,
    simulate_bess_capex,
)

# Module-level attributes monkeypatched by
# roa/input_files/sensitivity_price_modelling/run_roa_sensitivity.py.
# Do not rename or delete without coordinating with Mathea's script.
_REPO_ROOT = Path(__file__).parent.parent
_INPUT_DIR = _REPO_ROOT / "roa" / "input_files"
_DEFAULT_OUT_DIR = _REPO_ROOT / "roa" / "output_files"


# ---------------------------------------------------------------------------
# Monetary formatting (inlined from the deleted roa/formatting.py)
# ---------------------------------------------------------------------------

def fmt_nok(x: float) -> str:
    return f"{x:,.0f} NOK"


def fmt_mnok(x: float) -> str:
    return f"{x / 1e6:.3g} MNOK"


def fmt_bnok(x: float) -> str:
    return f"{x / 1e9:.4g} BNOK"


def fmt_money(x: float) -> str:
    """Auto-select NOK / MNOK / BNOK by magnitude of |x|."""
    if x is None:
        return "n/a"
    ax = abs(x)
    if ax >= 1e9:
        return fmt_bnok(x)
    if ax >= 1e6:
        return fmt_mnok(x)
    return fmt_nok(x)


# ---------------------------------------------------------------------------
# Per-scenario Config variant
# ---------------------------------------------------------------------------

def _make_scenario_cfg(
    base: Config,
    *,
    elavgift_psh: float,
    grunnrente_mode: str,
) -> Config:
    """Per-scenario Config: BESS elavgift held at "full"; PSH elavgift + GR mode set."""
    cfg = copy.deepcopy(base)
    cfg.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    cfg.elavgift_nok_per_mwh_psh = elavgift_psh
    cfg.grunnrente_mode = grunnrente_mode
    cfg.__post_init__()
    return cfg


# ---------------------------------------------------------------------------
# Stats and diagnostic helpers
# ---------------------------------------------------------------------------

def _array_summary(x: np.ndarray) -> dict:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "count": int(x.size),
            "finite_count": 0,
            "all_finite": False,
            "mean": None, "median": None, "p5": None, "p95": None,
            "min": None, "max": None, "share_positive": 0.0,
        }
    return {
        "count": int(x.size),
        "finite_count": int(finite.size),
        "all_finite": bool(finite.size == x.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p5": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "share_positive": float((finite > 0).mean()),
    }


def _payoff_diagnostics(cfg: Config, Pi_bess: np.ndarray, Pi_psh: np.ndarray) -> dict:
    out = {}
    for t in (0, cfg.T_def):
        out[str(t)] = {
            "Pi_BESS": _array_summary(Pi_bess[:, t]),
            "Pi_PSH":  _array_summary(Pi_psh[:, t]),
        }
    return out


def _cashflow_decomposition(
    cfg: Config,
    revenue_bess_q10: np.ndarray,
    charge_bess_q10: np.ndarray,
    revenue_psh: np.ndarray,
    charge_psh: np.ndarray,
    psh_production: np.ndarray,
) -> dict:
    bess_el = cfg.elavgift_nok_per_mwh_bess
    psh_el = cfg.elavgift_nok_per_mwh_psh

    bess_revenue = float(revenue_bess_q10.mean())
    bess_charge = float(charge_bess_q10.mean())
    bess_fixed = float(cfg.opex_bess_per_mwh_year * cfg.capex_bess_baseline_mwh)
    bess_el_component = float(bess_el * bess_charge)
    bess_total_opex = bess_fixed + bess_el_component
    bess_gross_margin = bess_revenue - bess_total_opex
    bess_d1 = float(cfg.dep_rate * cfg.gbm_K0 * cfg.capex_bess_baseline_mwh)

    psh_revenue = float(revenue_psh.mean())
    psh_charge = float(charge_psh.mean())
    psh_production_mean = float(psh_production.mean())
    psh_capacity_opex = float(cfg.psh_capacity_opex_nok)
    psh_el_component = float(psh_el * psh_charge)
    psh_production_component = float(cfg.psh_production_opex_per_mwh * psh_production_mean)
    psh_total_opex = psh_capacity_opex + psh_el_component + psh_production_component
    psh_gross_margin = psh_revenue - psh_total_opex
    psh_d1 = float(build_psh_depreciation_schedule(cfg, cfg.T_psh)[0])

    return {
        "BESS_q10": {
            "revenue": bess_revenue,
            "charged_energy": bess_charge,
            "fixed_om": bess_fixed,
            "elavgift_component": bess_el_component,
            "total_opex": bess_total_opex,
            "gross_margin": bess_gross_margin,
            "D1": bess_d1,
            "year1_tax_base": float(max(bess_gross_margin - bess_d1, 0.0)),
        },
        "PSH": {
            "revenue": psh_revenue,
            "charged_energy": psh_charge,
            "gross_turbined": psh_production_mean,
            "fixed_opex": psh_capacity_opex,
            "elavgift_component": psh_el_component,
            "production_component": psh_production_component,
            "total_opex": psh_total_opex,
            "gross_margin": psh_gross_margin,
            "D1": psh_d1,
            "year1_tax_base": float(max(psh_gross_margin - psh_d1, 0.0)),
        },
    }


def _psh_tax_decomposition(
    cfg: Config,
    revenue_psh: np.ndarray,
    revenue_psh_DA: np.ndarray,
    opex_psh: np.ndarray,
) -> dict:
    L = cfg.T_psh
    rev_total = revenue_psh[:, :L]
    rev_DA = revenue_psh_DA[:, :L]
    rev_FCR = rev_total - rev_DA
    opex_total = opex_psh[:, :L]

    share = np.divide(
        rev_DA, rev_total,
        out=np.zeros_like(rev_DA),
        where=np.abs(rev_total) > 1e-12,
    )
    share = np.clip(share, 0.0, 1.0)
    opex_DA = opex_total * share
    gm_total = rev_total - opex_total
    gm_DA = rev_DA - opex_DA

    dep_charge = build_psh_depreciation_schedule(cfg, L)
    ordinary_tax_base = np.maximum(gm_total - dep_charge[np.newaxis, :], 0.0)
    ordinary_tax = cfg.tax_rate * ordinary_tax_base

    gr_tax_op = compute_grunnrente_operating_tax_slice(
        cfg, rev_DA, rev_total, opex_total, L,
    )
    gr_capex_credit = compute_grunnrente_capex_credit(cfg, cfg.capex_psh)
    pv_operating_gr_tax = compute_opnpv(gr_tax_op, cfg)

    return {
        "rev_DA": float(rev_DA.mean()),
        "rev_FCR": float(rev_FCR.mean()),
        "rev_total": float(rev_total.mean()),
        "OPEX_total": float(opex_total.mean()),
        "OPEX_DA": float(opex_DA.mean()),
        "GM_total": float(gm_total.mean()),
        "GM_DA": float(gm_DA.mean()),
        "ordinary_tax_base": float(ordinary_tax_base.mean()),
        "ordinary_tax": float(ordinary_tax.mean()),
        "operating_GR_tax": float(gr_tax_op.mean()),
        "operating_GR_tax_min": float(gr_tax_op.min()),
        "GR_CAPEX_credit": float(gr_capex_credit),
        "PV_operating_GR_tax_mean": float(pv_operating_gr_tax.mean()),
        "net_GR_effect_on_Pi0": float(gr_capex_credit - pv_operating_gr_tax.mean()),
    }


def _build_runs_block(
    cfg: Config,
    Pi_bess: np.ndarray,
    Pi_psh: np.ndarray,
    p2: dict,
) -> tuple[dict, np.ndarray]:
    """Build the runs["joint"] stats block + return per-path X_i for K-seed accumulation."""
    N = cfg.N
    T_def = cfg.T_def
    tau = p2["exercise_times"]
    tech = p2["technology_choices"]
    exercised = tau >= 0
    safe_tau = np.maximum(tau, 0)
    E = np.maximum(Pi_bess[:, : T_def + 1], Pi_psh)
    payoff_at_stop = np.where(exercised, E[np.arange(N), safe_tau], 0.0)
    X_i = (cfg.delta ** safe_tau) * payoff_at_stop * exercised

    V0_stop = float(X_i.mean())
    V0_rec  = float(p2["V0_recursion"])
    rel_gap = abs(V0_rec - V0_stop) / max(abs(V0_stop), 1e-12)
    SE      = float(X_i.std(ddof=1) / np.sqrt(N))
    ci_lo   = V0_stop - 1.96 * SE
    ci_hi   = V0_stop + 1.96 * SE
    exercise_hist = [int((tau == t).sum()) for t in range(T_def + 1)]
    no_ex_count   = int((tau < 0).sum())
    bess_count = int((tech == 1).sum())
    psh_count  = int((tech == 2).sum())
    bess_by_year = [int(((tau == t) & (tech == 1)).sum()) for t in range(T_def + 1)]
    psh_by_year  = [int(((tau == t) & (tech == 2)).sum()) for t in range(T_def + 1)]

    n_ex = int(exercised.sum())
    if n_ex > 0:
        p_bess = float((tech[exercised] == 1).mean())
        p_psh  = float((tech[exercised] == 2).mean())
    else:
        p_bess = p_psh = 0.0

    runs_joint = {
        "V0_stoptime":           V0_stop,
        "V0_recursion":          V0_rec,
        "rel_gap":               rel_gap,
        "SE":                    SE,
        "CI95_low":              ci_lo,
        "CI95_high":             ci_hi,
        "exercise_rate":         float(exercised.mean()),
        "p_bess_given_ex":       p_bess,
        "p_psh_given_ex":        p_psh,
        "exercise_hist":         exercise_hist,
        "no_exercise_count":     no_ex_count,
        "bess_chosen_count":     bess_count,
        "psh_chosen_count":      psh_count,
        "bess_by_exercise_year": bess_by_year,
        "psh_by_exercise_year":  psh_by_year,
    }
    return runs_joint, X_i


# ---------------------------------------------------------------------------
# Symmetric grunnrente identity check (single-seed)
# ---------------------------------------------------------------------------

def _grunnrente_identity_check(
    cfg_sym: Config,
    revenue_psh: np.ndarray,
    revenue_psh_DA: np.ndarray,
    opex_psh_sym: np.ndarray,
    Pi_psh_sym_t0: np.ndarray,
    Pi_psh_none_t0: np.ndarray,
) -> dict:
    """Sample-path check that Pi_PSH_sym − Pi_PSH_none ≡ GR_credit − PV(op_GR_tax) at t=0.

    Raises RuntimeError on identity violation. Returns the check dict on success.
    """
    L = cfg_sym.T_psh
    gr_tax_op = compute_grunnrente_operating_tax_slice(
        cfg_sym,
        revenue_psh_DA[:, :L],
        revenue_psh[:, :L],
        opex_psh_sym[:, :L],
        L,
    )
    pv_operating_gr = compute_opnpv(gr_tax_op, cfg_sym)
    gr_credit = compute_grunnrente_capex_credit(cfg_sym, cfg_sym.capex_psh)
    lhs = Pi_psh_sym_t0 - Pi_psh_none_t0
    rhs = gr_credit - pv_operating_gr
    sample_idx = np.arange(min(10, cfg_sym.N))
    identity_diff = lhs[sample_idx] - rhs[sample_idx]
    identity_tol = np.maximum(
        1.0,
        1e-6 * np.maximum(np.abs(lhs[sample_idx]), np.abs(rhs[sample_idx])),
    )
    if np.any(np.abs(identity_diff) > identity_tol):
        raise RuntimeError("Symmetric grunnrente identity check failed for sample paths at t=0.")
    return {
        "sample_paths": sample_idx.astype(int).tolist(),
        "max_abs_diff": float(np.max(np.abs(identity_diff))),
        "GR_CAPEX_credit": float(gr_credit),
        "PV_operating_GR_tax_mean": float(pv_operating_gr.mean()),
        "passed": True,
    }


# ---------------------------------------------------------------------------
# Per-seed kernel: one Phase 1 + five Phase 2 sweeps with full decomposition payload
# ---------------------------------------------------------------------------

def _run_one_seed(
    cfg_base: Config,
    seed: int,
    data: OptimizerData,
    revenue_bess: dict,
    charge_bess: dict,
    opex_bess: dict,
    revenue_psh: np.ndarray,
    revenue_psh_DA: np.ndarray,
    charge_psh: np.ndarray,
    price_state: np.ndarray,
    fcr_state: np.ndarray,
) -> dict:
    """Run all five scenarios at one BESS-CAPEX seed.

    Returns
    -------
    dict with keys "scenarios" (mapping name → rich record) and "Pi_bess" (the
    Phase-1 Pi^BESS matrix shared across scenarios), plus the per-path X_i
    arrays accumulated for the K-seed pathwise SE.
    """
    capex_paths = simulate_bess_capex(cfg_base, seed)

    p1 = phase1_bess_compound(
        cfg_base, revenue_bess, opex_bess, capex_paths,
        price_state, fcr_paths=fcr_state,
    )
    Pi_bess_full = p1["Pi_bess"]
    inf_psh = np.full((cfg_base.N, cfg_base.T_def + 1), -np.inf, dtype=float)
    bess_q10 = int(cfg_base.capex_bess_baseline_mwh)   # 10

    out_scenarios: dict[str, dict] = {}
    Pi_psh_t0_by_scenario: dict[str, np.ndarray] = {}
    opex_psh_by_scenario: dict[str, np.ndarray] = {}

    for scenario_name, elavgift_key, gr_mode in SCENARIO_DEFINITIONS:
        cfg_s = _make_scenario_cfg(
            cfg_base,
            elavgift_psh=ELAVGIFT_SCENARIO_VALUES[elavgift_key],
            grunnrente_mode=gr_mode,
        )
        opex_psh = build_opex_matrix(
            cfg_s, charge_psh, "PSH", None,
            production_matrix=data.psh_bid_DA_sell,
        )
        Pi_psh = compute_psh_payoffs(
            cfg_s, revenue_psh, price_state, opex_psh, revenue_psh_DA,
        )

        # Joint Phase 2 — production headline
        p2_joint = phase2_joint_optimization(
            cfg_s, Pi_bess_full, Pi_psh, price_state, capex_paths, fcr_state,
        )
        runs_joint, X_joint = _build_runs_block(cfg_s, Pi_bess_full, Pi_psh, p2_joint)

        # BESS-only Phase 2 — invariance sentinel
        p2_bess = phase2_joint_optimization(
            cfg_s, Pi_bess_full, inf_psh, price_state, capex_paths, fcr_state,
        )
        runs_bess, X_bess = _build_runs_block(cfg_s, Pi_bess_full, inf_psh, p2_bess)

        cashflow = _cashflow_decomposition(
            cfg_s,
            revenue_bess[bess_q10], charge_bess[bess_q10],
            revenue_psh, charge_psh, data.psh_bid_DA_sell,
        )
        psh_tax = _psh_tax_decomposition(cfg_s, revenue_psh, revenue_psh_DA, opex_psh)
        payoff_diag = _payoff_diagnostics(cfg_s, Pi_bess_full, Pi_psh)

        out_scenarios[scenario_name] = {
            "panel": "A_elavgift" if gr_mode == "none" else "B_grunnrente",
            "elavgift_nok_per_mwh_bess": cfg_s.elavgift_nok_per_mwh_bess,
            "elavgift_nok_per_mwh_psh":  cfg_s.elavgift_nok_per_mwh_psh,
            "grunnrente_mode": gr_mode,
            "runs": {"joint": runs_joint, "bess_only": runs_bess},
            "cashflow_decomposition": cashflow,
            "psh_tax_decomposition":  psh_tax,
            "payoff_diagnostics":     payoff_diag,
            "Pi_BESS_paths_t0": Pi_bess_full[:, 0],
            "Pi_PSH_paths_t0":  Pi_psh[:, 0],
            "_X_joint": X_joint,
            "_X_bess":  X_bess,
        }
        Pi_psh_t0_by_scenario[scenario_name] = Pi_psh[:, 0]
        opex_psh_by_scenario[scenario_name] = opex_psh

    # Symmetric grunnrente identity check at t=0 (canonical).
    if "full_GRsymmetric" in out_scenarios and "full_GRnone" in out_scenarios:
        cfg_sym = _make_scenario_cfg(
            cfg_base,
            elavgift_psh=ELAVGIFT_SCENARIO_VALUES["full"],
            grunnrente_mode="symmetric",
        )
        identity = _grunnrente_identity_check(
            cfg_sym,
            revenue_psh, revenue_psh_DA,
            opex_psh_by_scenario["full_GRsymmetric"],
            Pi_psh_t0_by_scenario["full_GRsymmetric"],
            Pi_psh_t0_by_scenario["full_GRnone"],
        )
        out_scenarios["full_GRsymmetric"]["grunnrente_identity_check_t0"] = identity

    return {"seed": seed, "scenarios": out_scenarios}


# ---------------------------------------------------------------------------
# Top-level K-seed orchestration
# ---------------------------------------------------------------------------

@dataclass
class HeadlineConfig:
    K: int = 30
    seed_start: int = 42
    out_dir: Path = field(default_factory=lambda: _DEFAULT_OUT_DIR)
    N_override: int | None = None
    allow_nonproduction_shape: bool = False
    psh_concept: int = 1


def _build_shared_matrices(cfg_base: Config) -> dict:
    """Load optimiser data + build all scenario-invariant matrices once."""
    # `load_optimizer_data` is looked up on the module global at call time so
    # that monkeypatching in roa/input_files/sensitivity_price_modelling/...
    # takes effect.
    data = load_optimizer_data(cfg_base)

    bess_capacities = list(cfg_base.q_bess_grid)
    revenue_bess = {q: precompute_revenue_matrix(cfg_base, data, "BESS", q) for q in bess_capacities}
    charge_bess  = {q: precompute_charged_energy_matrix(cfg_base, data, "BESS", q) for q in bess_capacities}
    opex_bess    = {q: build_opex_matrix(cfg_base, charge_bess[q], "BESS", q) for q in bess_capacities}
    revenue_psh    = precompute_revenue_matrix(cfg_base, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh     = precompute_charged_energy_matrix(cfg_base, data, "PSH", None)
    if np.isnan(revenue_psh_DA).any():
        raise RuntimeError("PSH DA revenue contains NaN.")
    return {
        "data": data,
        "revenue_bess": revenue_bess,
        "charge_bess":  charge_bess,
        "opex_bess":    opex_bess,
        "revenue_psh":  revenue_psh,
        "revenue_psh_DA": revenue_psh_DA,
        "charge_psh":   charge_psh,
        "price_state":  data.mean_DA,
        "fcr_state":    data.mean_FCR,
    }


def _build_base_cfg(spec: HeadlineConfig) -> Config:
    cfg_base = Config()
    cfg_base.psh_concept = spec.psh_concept
    if spec.N_override is not None:
        cfg_base.N = spec.N_override
    cfg_base.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    cfg_base.elavgift_nok_per_mwh_psh = ELAVGIFT_SCENARIO_VALUES["full"]
    return cfg_base


def _check_canonical_shape(cfg: Config, spec: HeadlineConfig) -> bool:
    is_canonical = (cfg.N == 10000 and cfg.T_max == 80 and cfg.T_def == 10)
    if not is_canonical and not spec.allow_nonproduction_shape:
        raise RuntimeError(
            "Input validation failed: expected canonical shape "
            f"(N=10000, T_max=80, T_def=10); got N={cfg.N}, "
            f"T_max={cfg.T_max}, T_def={cfg.T_def}. "
            "Pass --allow-nonproduction-shape to bypass."
        )
    if not is_canonical:
        print(
            f"[main] WARNING: nonproduction shape "
            f"(N={cfg.N}, T_max={cfg.T_max}, T_def={cfg.T_def})."
        )
    return is_canonical


def run_kseeds(spec: HeadlineConfig | None = None, cfg_base: Config | None = None) -> dict:
    """Run K independent BESS-CAPEX seeds × five scenarios.

    When ``cfg_base`` is supplied it is used verbatim (e.g. by ``run_pipeline``,
    which accepts a user-provided Config); otherwise a fresh Config is built
    from ``spec``.
    """
    if spec is None:
        spec = HeadlineConfig()
    t0 = time.perf_counter()

    if cfg_base is None:
        cfg_base = _build_base_cfg(spec)

    shared = _build_shared_matrices(cfg_base)
    _check_canonical_shape(cfg_base, spec)

    seeds = list(range(spec.seed_start, spec.seed_start + spec.K))
    print(
        f"[main] Running K={spec.K} seeds {seeds[0]}..{seeds[-1]} "
        f"× {len(SCENARIO_DEFINITIONS)} scenarios "
        f"(N={cfg_base.N}, T_max={cfg_base.T_max}, T_def={cfg_base.T_def}, "
        f"psh_concept={cfg_base.psh_concept})."
    )

    per_seed_dicts: list[dict] = []
    phase1_invocations = 0
    for k_idx, seed in enumerate(seeds, start=1):
        t_seed = time.perf_counter()
        per_seed = _run_one_seed(
            cfg_base, seed, shared["data"],
            shared["revenue_bess"], shared["charge_bess"], shared["opex_bess"],
            shared["revenue_psh"], shared["revenue_psh_DA"], shared["charge_psh"],
            shared["price_state"], shared["fcr_state"],
        )
        phase1_invocations += 1
        per_seed_dicts.append(per_seed)
        print(f"  [{k_idx:>2d}/{spec.K}] seed={seed:<5d}  "
              f"({time.perf_counter() - t_seed:.1f}s)")

    # ---------- Aggregate per scenario (K-seed averaging) ----------
    K = spec.K
    summary: dict[str, dict] = {}
    canonical_scenarios: dict[str, dict] = {}

    for scenario_name, _, _ in SCENARIO_DEFINITIONS:
        recs = [ps["scenarios"][scenario_name] for ps in per_seed_dicts]

        V0s = np.array([r["runs"]["joint"]["V0_stoptime"] for r in recs], dtype=float)
        V0s_b = np.array([r["runs"]["bess_only"]["V0_stoptime"] for r in recs], dtype=float)
        ex_rates = np.array([r["runs"]["joint"]["exercise_rate"] for r in recs], dtype=float)
        p_pshs = np.array([r["runs"]["joint"]["p_psh_given_ex"] for r in recs], dtype=float)
        no_ex_counts = np.array([r["runs"]["joint"]["no_exercise_count"] for r in recs])
        X_stack = np.concatenate([r["_X_joint"] for r in recs])

        V0_mean = float(V0s.mean())
        V0_seed_sd = float(V0s.std(ddof=1)) if K > 1 else 0.0
        V0_seed_se = V0_seed_sd / float(np.sqrt(K)) if K > 1 else 0.0
        V0_pathwise_se = float(X_stack.std(ddof=1) / np.sqrt(X_stack.size))
        ci_lo = V0_mean - 1.96 * V0_seed_se
        ci_hi = V0_mean + 1.96 * V0_seed_se

        summary[scenario_name] = {
            "V0_stoptime_kseed_mean":           V0_mean,
            "V0_stoptime_kseed_sd":             V0_seed_sd,
            "V0_stoptime_kseed_se":             V0_seed_se,
            "V0_stoptime_kseed_ci95_low":       ci_lo,
            "V0_stoptime_kseed_ci95_high":      ci_hi,
            "V0_stoptime_pathwise_se":          V0_pathwise_se,
            "V0_stoptime_bess_only_kseed_mean": float(V0s_b.mean()),
            "exercise_rate_kseed_mean":         float(ex_rates.mean()),
            "p_psh_given_exercise_kseed_mean":  float(p_pshs.mean()),
            "mean_no_exercise_count":           float(no_ex_counts.mean()),
            "per_seed_V0_stoptime":             V0s.tolist(),
            "per_seed_V0_stoptime_bess_only":   V0s_b.tolist(),
            "per_seed_exercise_rate":           ex_rates.tolist(),
            "per_seed_p_psh_given_exercise":    p_pshs.tolist(),
        }

        # Canonical (seed_start) record carries the rich per-scenario payload
        # that Mathea's script and roa.plots both consume.
        canonical = per_seed_dicts[0]["scenarios"][scenario_name]
        canonical_scenarios[scenario_name] = canonical

    # ---------- Sanity check: cross-scenario BESS-only invariance ----------
    bess_means = np.array([
        summary[s[0]]["V0_stoptime_bess_only_kseed_mean"]
        for s in SCENARIO_DEFINITIONS
    ])
    sc_kavg_spread = float(bess_means.max() - bess_means.min())
    sc_kavg_tol = 1e-3
    sc_kavg_pass = sc_kavg_spread <= sc_kavg_tol

    # Single-seed BESS-only spread at seed_start (C8 from old main).
    bess_singleseed = np.array([
        canonical_scenarios[s[0]]["runs"]["bess_only"]["V0_stoptime"]
        for s in SCENARIO_DEFINITIONS
    ])
    sc_single_spread = float(bess_singleseed.max() - bess_singleseed.min())
    sc_single_tol = 1e3  # NOK
    sc_single_pass = sc_single_spread <= sc_single_tol

    # Phase 1 invocation count.
    phase1_pass = (phase1_invocations == K)

    elapsed = time.perf_counter() - t0

    headline_means = np.array([
        summary[s[0]]["V0_stoptime_kseed_mean"]
        for s in SCENARIO_DEFINITIONS
    ])
    headline_spread = float(headline_means.max() - headline_means.min())

    metadata = {
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
        "loader_mode":         "embedded",   # current loader path
        "K":                   K,
        "seed_start":          spec.seed_start,
        "seeds":               seeds,
        "N":                   cfg_base.N,
        "N_eff":               cfg_base.N,
        "T_max":               cfg_base.T_max,
        "T_max_eff":           cfg_base.T_max,
        "T_def":               cfg_base.T_def,
        "T_def_eff":           cfg_base.T_def,
        "T_psh":               cfg_base.T_psh,
        "T_bess":              cfg_base.T_bess,
        "psh_concept":         cfg_base.psh_concept,
        "capex_psh":           cfg_base.capex_psh,
        "r":                   cfg_base.r,
        "tax_rate":            cfg_base.tax_rate,
        "dep_rate":            cfg_base.dep_rate,
        "grunnrente_effective_rate": cfg_base.grunnrente_effective_rate,
        "phase1_invocations":  phase1_invocations,
        "elapsed_seconds":     elapsed,
    }

    return {
        "metadata":                       metadata,
        "scenarios":                      canonical_scenarios,
        "summary":                        summary,
        "headline_cross_scenario_spread": headline_spread,
        "sanity_checks": {
            "C_kavg_bess_only_invariance": {
                "passed": bool(sc_kavg_pass),
                "spread": sc_kavg_spread,
                "tol":    sc_kavg_tol,
            },
            "C8_single_seed_bess_only_invariance": {
                "passed": bool(sc_single_pass),
                "spread": sc_single_spread,
                "tol":    sc_single_tol,
            },
            "C_phase1_reuse": {
                "passed": bool(phase1_pass),
                "phase1_invocations": phase1_invocations,
                "expected": K,
            },
        },
    }


# ---------------------------------------------------------------------------
# Backward-compat single-seed entry point used by
# roa/input_files/sensitivity_price_modelling/run_roa_sensitivity.py
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: Config | None = None,
    seed: int = 42,
    allow_nonproduction_shape: bool = True,
    run_modes: list[str] | None = None,
) -> dict:
    """K=1 wrapper that returns the rich per-scenario dict in Mathea-compatible shape.

    Parameters
    ----------
    cfg :
        Optional Config; if None, defaults are used. Honored verbatim, so the
        caller controls psh_concept and any other override (e.g. N).
    seed :
        BESS-CAPEX GBM seed (default 42).
    allow_nonproduction_shape :
        Forwarded to the canonical-shape guard. Defaults to True for
        compatibility with run_roa_sensitivity.py, which uses N=1000.
    run_modes :
        Accepted for backwards compatibility; ignored. ``run_pipeline`` always
        runs joint + BESS-only inside the per-seed kernel.

    Returns
    -------
    dict
        ``{"metadata": ..., "scenarios": {scenario_name: rich_record, ...}}``.
        Each ``rich_record`` has the keys ``panel``, ``elavgift_nok_per_mwh_bess``,
        ``elavgift_nok_per_mwh_psh``, ``grunnrente_mode``, ``runs``,
        ``cashflow_decomposition``, ``psh_tax_decomposition`` —
        see run_roa_sensitivity.py for the exact contract.
    """
    spec = HeadlineConfig(
        K=1,
        seed_start=seed,
        out_dir=_DEFAULT_OUT_DIR,
        allow_nonproduction_shape=allow_nonproduction_shape,
    )
    cfg_in = cfg if cfg is not None else Config()
    # The canonical-shape guard reads N/T_max from cfg AFTER load_optimizer_data
    # mutates them; pre-populate elavgift to "full" so any cfg field reads are
    # consistent before _make_scenario_cfg fires.
    if not hasattr(cfg_in, "elavgift_nok_per_mwh_bess") or cfg_in.elavgift_nok_per_mwh_bess is None:
        cfg_in.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    if not hasattr(cfg_in, "elavgift_nok_per_mwh_psh") or cfg_in.elavgift_nok_per_mwh_psh is None:
        cfg_in.elavgift_nok_per_mwh_psh = ELAVGIFT_SCENARIO_VALUES["full"]

    results = run_kseeds(spec, cfg_base=cfg_in)

    # Strip the per-seed accumulators that are not part of Mathea's contract.
    for name in list(results["scenarios"].keys()):
        rec = results["scenarios"][name]
        rec.pop("_X_joint", None)
        rec.pop("_X_bess", None)
        # Convert numpy arrays to lists for JSON-friendliness (Mathea writes
        # the dict to disk via json.dump).
        if isinstance(rec.get("Pi_BESS_paths_t0"), np.ndarray):
            rec["Pi_BESS_paths_t0"] = rec["Pi_BESS_paths_t0"].tolist()
        if isinstance(rec.get("Pi_PSH_paths_t0"), np.ndarray):
            rec["Pi_PSH_paths_t0"] = rec["Pi_PSH_paths_t0"].tolist()

    return results


# ---------------------------------------------------------------------------
# Print helpers (ported from old main.py)
# ---------------------------------------------------------------------------

def _print_results_table(results: dict) -> None:
    print()
    print("=" * 104)
    print("  ROA Results by Scenario (single-seed canonical: seed_start)")
    print("=" * 104)
    header = (
        f"  {'Scenario':17s} {'Mode':10s} {'V0_stop':>12s} {'V0_rec':>12s} "
        f"{'SE':>11s} {'CI95':>27s} {'ex%':>7s} {'no ex':>7s} "
        f"{'BESS|ex':>9s} {'PSH|ex':>9s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for scenario, sr in results["scenarios"].items():
        for mode in ("joint", "bess_only"):
            if mode not in sr["runs"]:
                continue
            s = sr["runs"][mode]
            ci_str = f"[{fmt_money(s['CI95_low'])}, {fmt_money(s['CI95_high'])}]"
            print(f"  {scenario:17s} {mode:10s} {fmt_money(s['V0_stoptime']):>12s} "
                  f"{fmt_money(s['V0_recursion']):>12s} {fmt_money(s['SE']):>11s} "
                  f"{ci_str:>27s} {s['exercise_rate']:>7.1%} "
                  f"{s['no_exercise_count']:>7d} {s['p_bess_given_ex']:>9.1%} "
                  f"{s['p_psh_given_ex']:>9.1%}")
    print("=" * 104)


def _print_cashflow_decomposition(results: dict) -> None:
    print()
    print("=" * 134)
    print("  Cash-flow / OPEX Decomposition")
    print("=" * 134)
    header = (
        f"  {'Scenario':17s} {'Tech':8s} {'Revenue':>12s} {'Charge':>12s} "
        f"{'Fixed':>12s} {'Elavgift':>12s} {'Production':>12s} {'OPEX':>12s} "
        f"{'GM':>12s} {'D1':>12s} {'Tax base':>12s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for scenario, sr in results["scenarios"].items():
        for tech_key, label in (("BESS_q10", "BESS q10"), ("PSH", "PSH")):
            d = sr["cashflow_decomposition"][tech_key]
            fixed_key = "fixed_om" if tech_key == "BESS_q10" else "fixed_opex"
            production = d.get("production_component", 0.0)
            production_str = fmt_money(production) if tech_key == "PSH" else "—"
            print(
                f"  {scenario:17s} {label:8s} {fmt_money(d['revenue']):>12s} "
                f"{d['charged_energy']:>12,.0f} {fmt_money(d[fixed_key]):>12s} "
                f"{fmt_money(d['elavgift_component']):>12s} {production_str:>12s} "
                f"{fmt_money(d['total_opex']):>12s} "
                f"{fmt_money(d['gross_margin']):>12s} {fmt_money(d['D1']):>12s} "
                f"{fmt_money(d['year1_tax_base']):>12s}"
            )
    print("=" * 134)


def _print_psh_tax_decomposition(results: dict) -> None:
    print()
    print("=" * 140)
    print("  PSH Tax / Grunnrente Decomposition at t=0 Operating Window")
    print("=" * 140)
    header = (
        f"  {'Scenario':17s} {'GR':>9s} {'rev_DA':>11s} {'rev_FCR':>11s} "
        f"{'rev_total':>11s} {'OPEX':>11s} {'OPEX_DA':>11s} {'GM_total':>11s} "
        f"{'GM_DA':>11s} {'ord tax':>11s} {'op GR tax':>11s} {'GR credit':>11s} "
        f"{'net GR Pi0':>11s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for scenario, sr in results["scenarios"].items():
        d = sr["psh_tax_decomposition"]
        print(
            f"  {scenario:17s} {sr['grunnrente_mode']:>9s} "
            f"{fmt_money(d['rev_DA']):>11s} {fmt_money(d['rev_FCR']):>11s} "
            f"{fmt_money(d['rev_total']):>11s} {fmt_money(d['OPEX_total']):>11s} "
            f"{fmt_money(d['OPEX_DA']):>11s} {fmt_money(d['GM_total']):>11s} "
            f"{fmt_money(d['GM_DA']):>11s} {fmt_money(d['ordinary_tax']):>11s} "
            f"{fmt_money(d['operating_GR_tax']):>11s} {fmt_money(d['GR_CAPEX_credit']):>11s} "
            f"{fmt_money(d['net_GR_effect_on_Pi0']):>11s}"
        )
    print("=" * 140)


def _print_payoff_diagnostics(results: dict) -> None:
    print()
    print("=" * 120)
    print("  Payoff Diagnostics (single-seed canonical)")
    print("=" * 120)
    header = (
        f"  {'Scenario':17s} {'t':>3s} {'Array':8s} {'mean':>12s} {'median':>12s} "
        f"{'p5':>12s} {'p95':>12s} {'min':>12s} {'max':>12s} {'pos%':>8s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for scenario, sr in results["scenarios"].items():
        for t, diag_by_array in sr["payoff_diagnostics"].items():
            for name in ("Pi_BESS", "Pi_PSH"):
                d = diag_by_array[name]
                print(
                    f"  {scenario:17s} {int(t):>3d} {name:8s} "
                    f"{fmt_money(d['mean']):>12s} {fmt_money(d['median']):>12s} "
                    f"{fmt_money(d['p5']):>12s} {fmt_money(d['p95']):>12s} "
                    f"{fmt_money(d['min']):>12s} {fmt_money(d['max']):>12s} "
                    f"{d['share_positive']:>8.1%}"
                )
    print("=" * 120)


def _print_methodology_sanity_check(results: dict, cfg: Config) -> None:
    """Sanity prints anchored on the canonical full_GRnone scenario."""
    sr = results["scenarios"].get("full_GRnone")
    if sr is None:
        return
    cf_psh = sr["cashflow_decomposition"]["PSH"]
    print()
    print("=" * 60)
    print("  Methodology Sanity Check (full_GRnone scenario)")
    print("=" * 60)

    sched = build_psh_depreciation_schedule(cfg, cfg.T_psh)
    concept = cfg.psh_concepts[cfg.psh_concept - 1]
    capex_civil = concept.capex_bygg_nok_per_mw * concept.psh_mw
    capex_elek  = concept.capex_elek_nok_per_mw * concept.psh_mw
    capex_andre = concept.capex_andre_nok_per_mw * concept.psh_mw
    print(f"  PSH CAPEX components (NOK):")
    print(f"    civil works (bygg):       {fmt_money(capex_civil)}")
    print(f"    electromechanical (elek): {fmt_money(capex_elek)}")
    print(f"    other (andre):            {fmt_money(capex_andre)}")
    print(f"    total:                    {fmt_money(cfg.capex_psh)}")
    print(f"  PSH §18-6 depreciation profile (NOK/yr):")
    print(f"    D_k at k=1:   {fmt_money(sched[0])}")
    print(f"    D_k at k=20:  {fmt_money(sched[19])}")
    print(f"    D_k at k=40:  {fmt_money(sched[39])}")

    prop_tax_per_mwh = (
        cfg.opex_psh_property_tax_rate
        * cfg.opex_psh_property_tax_nok_per_kwh
        * 1_000
    )
    bid_sell_mean = float(cf_psh["gross_turbined"])
    print(f"  Property-tax rate: {prop_tax_per_mwh:.4f} NOK/MWh")
    print(f"  Gross turbined volume (mean): {bid_sell_mean:,.0f} MWh/yr")

    print(f"  PSH OPEX decomposition (mean):")
    print(f"    Capacity-scaled (O&M + rehab):    {fmt_money(cf_psh['fixed_opex'])}")
    print(f"    Elavgift × bid_DA_buy:            {fmt_money(cf_psh['elavgift_component'])}")
    print(f"    Production × bid_DA_sell:         {fmt_money(cf_psh['production_component'])}")
    print(f"    Total OPEX:                       {fmt_money(cf_psh['total_opex'])}")

    tv = compute_psh_terminal_value(cfg)
    cum_dep_frac = min(cfg.T_psh * cfg.psh_dep_rate_civil, 1.0)
    book_civil = capex_civil * (1.0 - cum_dep_frac)
    print(f"  Year-T_psh terminal value:")
    print(f"    civil book at k=T_psh: {fmt_money(book_civil)}")
    print(f"    multiplier:            {cfg.psh_terminal_value_multiplier:.2f}")
    print(f"    post-tax terminal CF:  {fmt_money(tv)}")
    print("=" * 60)


def _regression_checks(results: dict, cfg: Config) -> None:
    """Mechanical PASS/FAIL checks. Ports C1, C2, C5, C6, C7, C8, C9 from old main."""
    print()
    print("=" * 60)
    print("  Mechanical validation checks")
    print("=" * 60)

    md = results["metadata"]
    scenarios = results["scenarios"]
    sanity = results["sanity_checks"]

    def _row(label: str, ok: bool, detail: str = "") -> None:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}{(' — ' + detail) if detail else ''}")

    tol = 1e3

    _row("C1 loader_mode = embedded; N=10000; T_max=80",
         md["loader_mode"] == "embedded" and md["N_eff"] == 10000 and md["T_max_eff"] == 80,
         f"got mode={md['loader_mode']}, N={md['N_eff']}, T_max={md['T_max_eff']}")

    _row("C2 T_def = 10 (full deferral window)",
         md["T_def_eff"] == 10,
         f"T_def_eff={md['T_def_eff']}")

    v0_checks = []
    for _, sr in scenarios.items():
        for r in sr["runs"].values():
            v0_checks.append(np.isfinite(r["V0_stoptime"]) and np.isfinite(r["V0_recursion"]))
    _row("C5 all V0 values finite", all(v0_checks),
         f"checked={len(v0_checks)} run values")

    pi_checks = []
    for sr in scenarios.values():
        for diag_by_array in sr["payoff_diagnostics"].values():
            pi_checks.append(diag_by_array["Pi_BESS"]["all_finite"])
            pi_checks.append(diag_by_array["Pi_PSH"]["all_finite"])
    _row("C6 Pi_BESS and Pi_PSH diagnostics finite", all(pi_checks),
         f"checked={len(pi_checks)} arrays")

    # C7: joint >= bess_only (single-seed canonical here; with -tol slack).
    c7_checks = []
    c7_details = []
    for scenario, sr in scenarios.items():
        runs = sr["runs"]
        if "joint" in runs and "bess_only" in runs:
            diff = runs["joint"]["V0_stoptime"] - runs["bess_only"]["V0_stoptime"]
            c7_checks.append(diff >= -tol)
            c7_details.append(f"{scenario} J-B={fmt_money(diff)}")
    _row("C7 joint value dominates BESS-only (single-seed)",
         all(c7_checks) if c7_checks else True,
         "; ".join(c7_details) + f"; tol={fmt_money(tol)}")

    # C8: single-seed BESS-only invariance across scenarios.
    sc8 = sanity["C8_single_seed_bess_only_invariance"]
    _row("C8 BESS-only invariant across scenarios (single seed)",
         sc8["passed"],
         f"spread={fmt_money(sc8['spread'])}; tol={fmt_money(sc8['tol'])}")

    # K-averaged BESS-only invariance (additional check for K>1).
    sc_k = sanity["C_kavg_bess_only_invariance"]
    if md["K"] > 1:
        _row(f"C8' BESS-only invariant across scenarios (K={md['K']}-averaged)",
             sc_k["passed"],
             f"spread={sc_k['spread']:.3e} NOK; tol={sc_k['tol']:.0e} NOK")

    # C9: symmetric GR identity.
    identity = scenarios.get("full_GRsymmetric", {}).get("grunnrente_identity_check_t0")
    _row("C9 symmetric GR identity confirms CAPEX credit",
         bool(identity and identity.get("passed")),
         f"max_abs_diff={identity['max_abs_diff']:.6g}" if identity else "not available")

    sc_p1 = sanity["C_phase1_reuse"]
    _row(f"Phase 1 invoked exactly K={md['K']} times",
         sc_p1["passed"],
         f"invocations={sc_p1['phase1_invocations']}")


def _print_kseed_summary(results: dict) -> None:
    """K-seed summary block (ported from headline_kseeds._print_summary)."""
    if results["metadata"]["K"] <= 1:
        return
    summary = results["summary"]
    meta = results["metadata"]
    print()
    print("=" * 76)
    print(f"  K-seed Headline V_0^stop Summary  (K={meta['K']}, "
          f"seeds {meta['seeds'][0]}..{meta['seeds'][-1]})")
    print("=" * 76)
    for scenario_name, _, _ in SCENARIO_DEFINITIONS:
        s = summary[scenario_name]
        print(
            f"  {scenario_name:18s}: V_0^stop = "
            f"{s['V0_stoptime_kseed_mean']/1e6:8.3f} MNOK   "
            f"(seed SE = {s['V0_stoptime_kseed_se']/1e6:6.4f},  "
            f"pathwise SE = {s['V0_stoptime_pathwise_se']/1e6:6.4f})"
        )
    print(
        f"\n  Cross-scenario spread in seed-averaged V_0^stop: "
        f"{results['headline_cross_scenario_spread']:.3e} NOK"
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_outputs(results: dict, out_dir: Path) -> dict[str, Path]:
    """Write headline JSON, per-seed CSV, and main_pipeline JSON (rich, single-seed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: dict[str, Path] = {}

    headline_path = out_dir / f"headline_{ts}.json"
    headline_payload = {
        "metadata":                       results["metadata"],
        "summary":                        results["summary"],
        "headline_cross_scenario_spread": results["headline_cross_scenario_spread"],
        "sanity_checks":                  results["sanity_checks"],
    }
    with headline_path.open("w") as f:
        json.dump(headline_payload, f, indent=2)
    paths["headline_json"] = headline_path

    # Rich single-seed payload, compatible with roa.plots.loader.
    main_path = out_dir / f"main_pipeline_{ts}.json"
    main_payload = {
        "metadata": results["metadata"],
        "scenarios": {
            name: {
                "panel":                      rec["panel"],
                "elavgift_nok_per_mwh_bess":  rec["elavgift_nok_per_mwh_bess"],
                "elavgift_nok_per_mwh_psh":   rec["elavgift_nok_per_mwh_psh"],
                "grunnrente_mode":            rec["grunnrente_mode"],
                "runs":                       rec["runs"],
                "payoff_diagnostics":         rec["payoff_diagnostics"],
                "cashflow_decomposition":     rec["cashflow_decomposition"],
                "psh_tax_decomposition":      rec["psh_tax_decomposition"],
                "grunnrente_identity_check_t0": rec.get("grunnrente_identity_check_t0"),
                "Pi_BESS_paths_t0": (
                    rec["Pi_BESS_paths_t0"].tolist()
                    if isinstance(rec.get("Pi_BESS_paths_t0"), np.ndarray)
                    else rec.get("Pi_BESS_paths_t0", [])
                ),
                "Pi_PSH_paths_t0": (
                    rec["Pi_PSH_paths_t0"].tolist()
                    if isinstance(rec.get("Pi_PSH_paths_t0"), np.ndarray)
                    else rec.get("Pi_PSH_paths_t0", [])
                ),
            }
            for name, rec in results["scenarios"].items()
        },
    }
    with main_path.open("w") as f:
        json.dump(main_payload, f, indent=2)
    paths["main_pipeline_json"] = main_path

    # Per-seed CSV.
    csv_path = out_dir / f"per_seed_{ts}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "seed",
            "V0_stoptime", "V0_stoptime_bess_only",
            "exercise_rate", "p_psh_given_exercise",
        ])
        for scenario_name, _, _ in SCENARIO_DEFINITIONS:
            s = results["summary"][scenario_name]
            for i, seed in enumerate(results["metadata"]["seeds"]):
                w.writerow([
                    scenario_name, seed,
                    f"{s['per_seed_V0_stoptime'][i]:.6f}",
                    f"{s['per_seed_V0_stoptime_bess_only'][i]:.6f}",
                    f"{s['per_seed_exercise_rate'][i]:.6f}",
                    f"{s['per_seed_p_psh_given_exercise'][i]:.6f}",
                ])
    paths["per_seed_csv"] = csv_path

    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="K-seed averaged ROA pipeline (NOK).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--K", type=int, default=30,
                   help="Number of independent BESS-CAPEX seeds (default 30; "
                        "use --K 1 for single-seed quick check).")
    p.add_argument("--seed-start", type=int, default=42,
                   help="First seed.")
    p.add_argument("--out", type=str, default=None,
                   help="Output directory. Defaults to roa/output_files/.")
    p.add_argument("--psh_concept", type=int, default=1, choices=[1, 2, 3],
                   help="Active PSH concept (1=100 MW, 2=1000 MW, 3=1000 MW low-cost).")
    p.add_argument("--N", type=int, default=None,
                   help="Override sample size (requires --allow-nonproduction-shape).")
    p.add_argument("--allow-nonproduction-shape", action="store_true",
                   help="Bypass the canonical N=10000 / T_max=80 / T_def=10 guard.")
    return p.parse_args(argv)


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = Path(args.out) if args.out is not None else _DEFAULT_OUT_DIR
    spec = HeadlineConfig(
        K=args.K,
        seed_start=args.seed_start,
        out_dir=out_dir,
        N_override=args.N,
        allow_nonproduction_shape=args.allow_nonproduction_shape,
        psh_concept=args.psh_concept,
    )

    print()
    print("=" * 60)
    print("  ROA Pipeline — K-seed Production Headline (NOK)")
    print("=" * 60)
    print(f"  K          : {spec.K}")
    print(f"  seed_start : {spec.seed_start}")
    print(f"  out_dir    : {out_dir}")
    print(f"  psh_concept: {spec.psh_concept}")
    if spec.N_override is not None:
        print(f"  N override : {spec.N_override}")
    if spec.allow_nonproduction_shape:
        print(f"  shape      : nonproduction")
    print("=" * 60)
    print()

    cfg_base = _build_base_cfg(spec)
    results = run_kseeds(spec, cfg_base=cfg_base)
    paths = _save_outputs(results, out_dir)

    _print_results_table(results)
    _print_cashflow_decomposition(results)
    _print_psh_tax_decomposition(results)
    _print_payoff_diagnostics(results)
    _print_methodology_sanity_check(results, cfg_base)
    _print_kseed_summary(results)
    _regression_checks(results, cfg_base)

    print()
    for label, path in paths.items():
        print(f"  Saved {label}: {path}")
    return results


if __name__ == "__main__":
    main()
