"""PSH CAPEX sensitivity sweep.

Reuses the main pipeline's BESS Phase 1 and revenue / charge / OPEX matrices.
Re-runs only Phase 2 (joint mutually exclusive optimisation) for a sweep of
PSH CAPEX values.

Phase 1 is independent of PSH CAPEX, so it is computed once. Revenue and
charged-energy matrices come directly from optimizer output; this is a
partial-equilibrium analysis that isolates the CAPEX channel only.
"""

from __future__ import annotations

import copy
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from roa.config import (
    Config,
    PshConcept,
    ELAVGIFT_SCENARIO_VALUES,
    SCENARIO_DEFINITIONS,
)
from roa.investment import (
    compute_psh_payoffs,
    phase1_bess_compound,
    phase2_joint_optimization,
)
from roa.lsmc import (
    build_opex_matrix,
    load_optimizer_data,
    precompute_charged_energy_matrix,
    precompute_psh_da_revenue_matrix,
    precompute_revenue_matrix,
    simulate_bess_capex,
)
from roa.sensitivity import BREAKEVEN_REL_EPS


_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output_files"


def override_psh_capex(cfg: Config, capex_target_nok: float) -> Config:
    """Return a deep copy of cfg with the active PSH concept's total CAPEX set to target.

    The override collapses all CAPEX components onto the bygg field for simplicity;
    only the total matters downstream because compute_psh_exercise_payoff reads
    config.capex_psh = active_concept.total_nok().
    """
    new_cfg = copy.deepcopy(cfg)
    idx = new_cfg.psh_concept - 1
    mw = new_cfg.psh_concepts[idx].psh_mw
    per_mw = float(capex_target_nok) / mw
    new_cfg.psh_concepts[idx] = PshConcept(
        capex_bygg_nok_per_mw=per_mw,
        capex_elek_nok_per_mw=0.0,
        capex_andre_nok_per_mw=0.0,
        psh_mw=mw,
    )
    return new_cfg


def _make_scenario_cfg(
    base: Config, *, elavgift_psh: float, grunnrente_mode: str,
) -> Config:
    cfg = copy.deepcopy(base)
    cfg.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    cfg.elavgift_nok_per_mwh_psh = elavgift_psh
    cfg.grunnrente_mode = grunnrente_mode
    cfg.__post_init__()
    return cfg


def _v0_stoptime(p2: dict, Pi_bess: np.ndarray, Pi_psh: np.ndarray, cfg: Config):
    tau = p2["exercise_times"]
    exercised = tau >= 0
    safe_tau = np.maximum(tau, 0)
    E_joint = np.maximum(Pi_bess[:, : cfg.T_def + 1], Pi_psh)
    payoff_at_stop = np.where(exercised, E_joint[np.arange(cfg.N), safe_tau], 0.0)
    X = (cfg.delta ** safe_tau) * payoff_at_stop * exercised
    v0 = float(X.mean())
    se = float(X.std(ddof=1) / np.sqrt(cfg.N))
    return v0, se, X, exercised, p2["technology_choices"]


def _interpolate_threshold_capex(
    capex_arr: np.ndarray,
    series: np.ndarray,
    threshold: float,
) -> float | None:
    """Largest CAPEX at which `series` crosses `threshold` as CAPEX decreases.

    Returns the CAPEX above which series <= threshold and below which
    series > threshold. None if no such crossing in range, or if series is
    already above threshold at the maximum CAPEX (out-of-range).
    """
    order = np.argsort(capex_arr)
    x = capex_arr[order]
    y = series[order]
    above = y > threshold
    if not above.any():
        return None
    if above[-1]:
        return None
    j_rev = int(np.argmax(above[::-1]))
    j = len(x) - 1 - j_rev
    if j + 1 >= len(x):
        return float(x[j])
    x0, x1 = x[j], x[j + 1]
    y0, y1 = y[j] - threshold, y[j + 1] - threshold
    if y1 == y0:
        return float(x1)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def _summarise_thresholds(
    sweep: dict, bess_only_v0: float, rel_eps: float = BREAKEVEN_REL_EPS,
) -> dict:
    """Compute breakeven and crossover CAPEX (BNOK) per scenario.

    breakeven : largest CAPEX where V0_stoptime > bess_only_v0 * (1 + rel_eps).
    crossover : largest CAPEX where p_psh_given_ex > 0.5.
    Both via downward-in-CAPEX scan with linear interpolation.
    """
    capex_keys = sorted(sweep.keys(), key=float)
    capex_arr = np.array([float(k) for k in capex_keys])
    scenario_names = list(sweep[capex_keys[0]].keys())

    out: dict[str, dict[str, float | None]] = {}
    floor_v0 = float(bess_only_v0) * (1.0 + rel_eps)
    for name in scenario_names:
        v0_series = np.array([sweep[k][name]["V0_stoptime"] for k in capex_keys])
        psh_share_series = np.array([sweep[k][name]["p_psh_given_ex"] for k in capex_keys])
        out[name] = {
            "breakeven_capex_bnok": _interpolate_threshold_capex(capex_arr, v0_series, floor_v0),
            "crossover_capex_bnok": _interpolate_threshold_capex(capex_arr, psh_share_series, 0.5),
        }
    return out


def run_capex_sweep(
    capex_values_bnok: np.ndarray | list[float],
    output_path: Path | str | None = None,
    seed: int = 42,
) -> dict:
    capex_values_bnok = np.asarray(list(capex_values_bnok), dtype=float)

    cfg = Config()

    print("[sweep] Loading optimizer data...", flush=True)
    data = load_optimizer_data(cfg)

    price_state = data.mean_DA
    capex_paths = simulate_bess_capex(cfg, seed)

    bess_capacities = list(cfg.q_bess_grid)
    revenue_bess = {q: precompute_revenue_matrix(cfg, data, "BESS", q)
                    for q in bess_capacities}
    charge_bess  = {q: precompute_charged_energy_matrix(cfg, data, "BESS", q)
                    for q in bess_capacities}
    opex_bess    = {q: build_opex_matrix(cfg, charge_bess[q], "BESS", q)
                    for q in bess_capacities}
    revenue_psh    = precompute_revenue_matrix(cfg, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh     = precompute_charged_energy_matrix(cfg, data, "PSH", None)

    if np.isnan(revenue_psh_DA).any():
        raise RuntimeError("PSH DA revenue contains NaN.")

    print("[sweep] Phase 1: BESS compound chain...", flush=True)
    t0 = time.perf_counter()
    p1 = phase1_bess_compound(
        cfg, revenue_bess, opex_bess, capex_paths, price_state,
        fcr_paths=data.mean_FCR,
    )
    print(f"  ({time.perf_counter() - t0:.1f}s)")
    Pi_bess_full = p1["Pi_bess"]

    # BESS-only reference V0 (CAPEX-independent)
    inf_psh = np.full((cfg.N, cfg.T_def + 1), -np.inf, dtype=float)
    p2_bess = phase2_joint_optimization(
        cfg, Pi_bess_full, inf_psh, price_state, capex_paths,
        fcr_paths=data.mean_FCR,
    )
    bess_only_v0, _, _, _, _ = _v0_stoptime(p2_bess, Pi_bess_full, inf_psh, cfg)
    print(f"[sweep] BESS-only V0 reference: {bess_only_v0:,.2f} NOK")

    sweep: dict[str, dict] = {}
    t_sweep = time.perf_counter()
    for capex_bnok in capex_values_bnok:
        capex_nok = float(capex_bnok) * 1e9
        cfg_capex = override_psh_capex(cfg, capex_nok)
        per_capex: dict[str, dict] = {}
        for scenario_name, elavgift_key, gr_mode in SCENARIO_DEFINITIONS:
            cfg_s = _make_scenario_cfg(
                cfg_capex,
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
            p2 = phase2_joint_optimization(
                cfg_s, Pi_bess_full, Pi_psh, price_state, capex_paths,
                fcr_paths=data.mean_FCR,
            )
            v0, se, _, exercised, tech = _v0_stoptime(p2, Pi_bess_full, Pi_psh, cfg_s)
            n_ex = int(exercised.sum())
            n_psh = int(((tech == 2) & exercised).sum())
            n_bess = int(((tech == 1) & exercised).sum())
            psh_share = (n_psh / n_ex) if n_ex > 0 else 0.0
            bess_share = (n_bess / n_ex) if n_ex > 0 else 0.0
            per_capex[scenario_name] = {
                "V0_stoptime": v0,
                "SE": se,
                "p_psh_given_ex": psh_share,
                "p_bess_given_ex": bess_share,
                "exercise_rate": n_ex / cfg_s.N,
                "n_psh_chosen": n_psh,
                "n_bess_chosen": n_bess,
                "Pi_PSH_paths_t0": Pi_psh[:, 0].tolist(),
            }
        sweep[f"{capex_bnok:.1f}"] = per_capex
        print(f"[sweep] CAPEX = {capex_bnok:5.1f} BNOK done.")
    print(f"[sweep] Total sweep time: {time.perf_counter() - t_sweep:.1f}s")

    breakeven_summary = _summarise_thresholds(sweep, bess_only_v0)

    out = {
        "metadata": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "seed": seed,
            "psh_concept": cfg.psh_concept,
            "capex_values_bnok": [float(c) for c in capex_values_bnok],
            "BESS_only_V0_stoptime": bess_only_v0,
            "N": cfg.N,
            "T_max": cfg.T_max,
            "T_def": cfg.T_def,
            "T_psh": cfg.T_psh,
            "T_bess": cfg.T_bess,
            "scenario_defs": [
                {"name": n, "elavgift": e, "grunnrente_mode": g}
                for n, e, g in SCENARIO_DEFINITIONS
            ],
        },
        "breakeven_summary": breakeven_summary,
        "sweep": sweep,
    }

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[sweep] Saved {output_path}")
    return out


def main() -> None:
    capex_values = np.arange(3.0, 18.0 + 0.5, 0.5)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = _OUTPUT_DIR / f"psh_capex_sweep_{timestamp}.json"
    run_capex_sweep(capex_values, out_path)


if __name__ == "__main__":
    main()
