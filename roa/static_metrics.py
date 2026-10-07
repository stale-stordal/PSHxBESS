"""
Static baseline metrics (WS1 v0.5) — additive diagnostic module.

Headline = a STATIC NO-FLEX baseline: NPV / ROI / PI at t=0, no operating
flexibility (no waiting, exiting, resizing, or re-optimisation).

  - BESS: forced reinvestment at the 10 MWh baseline unit every T_bess years
    over the PSH-matched horizon H = T_psh (mechanical replay_bess_noflex).
  - PSH:  the one-shot investment exercised now, for all five scenarios in
    config.SCENARIO_DEFINITIONS.

The cross-path MEAN cash-flow stream feeds the metrics. In the mean, the no-flex
BESS NPV lies below the invest-now-plus-flexibility value mean(Pi_bess[:, 0]); the
flexibility strip must be comfortably positive. Per-path dominance is NOT asserted
(Pi_bess is an LSMC value estimate, not a per-path optimum).

Usage
-----
    cd masteroppgave
    python -m roa.static_metrics
"""

from __future__ import annotations

import sys
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
)
from roa.lsmc import (
    build_opex_matrix,
    compute_ncf,
    compute_psh_cashflow_components,
    compute_psh_terminal_value,
    load_optimizer_data,
    precompute_charged_energy_matrix,
    precompute_psh_da_revenue_matrix,
    precompute_revenue_matrix,
    simulate_bess_capex,
)
from roa.main import _make_scenario_cfg

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INPUT_DIR = _REPO_ROOT / "roa" / "input_files"
_RESULTS_DIR = _REPO_ROOT / "results"

_SEED = 42       # canonical BESS-CAPEX GBM seed (= main.py seed_start)
_Q0 = 10.0       # BESS baseline unit (MWh); the no-flex forced capacity


# =============================================================================
# === STEP 2: NO-FLEX BESS MECHANICAL REPLAY (the WS1 baseline) ===
# =============================================================================

def replay_bess_noflex(
    config: Config,
    revenue_bess: dict[float, np.ndarray],
    opex_bess: dict[float, np.ndarray],
    capex_paths: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mechanical no-flex BESS cash-flow schedule over H = T_psh years.

    Forced reinvestment at q0 = 10 MWh every T_bess years, with the final cycle
    truncated to fit H. NO continuation value, NO waiting/exiting/resizing, NO
    reading of the optimal policy. This is the static baseline, deliberately
    stripped of all operating flexibility.

    Returns
    -------
    dict with two (N, H+1) float64 arrays:
        "I" : capital outflows, NOK (stochastic CAPEX at each reinvest year).
        "F" : after-tax operating inflows, NOK.
    """
    N = capex_paths.shape[0]
    H = config.T_psh
    T_bess = config.T_bess

    I = np.zeros((N, H + 1), dtype=float)
    F = np.zeros((N, H + 1), dtype=float)

    for t in range(0, H, T_bess):                     # 0, 10, 20, 30, 40, 50, 60
        L = min(T_bess, H - t)                         # t=60 -> L=7 (truncated)
        capex_t = capex_paths[:, t] * _Q0              # (N,) NOK
        ncf = compute_ncf(
            revenue_bess[_Q0][:, t:t + L],
            opex_bess[_Q0][:, t:t + L],
            capex_t,
            config,
            L,
        )                                              # (N, L), undiscounted
        I[:, t]          += capex_t
        F[:, t + 1:t + L + 1] += ncf

    return {"I": I, "F": F}


# =============================================================================
# === STEP 3: STATIC METRICS ON THE CROSS-PATH MEAN STREAM (no IRR/MIRR) ===
# =============================================================================

def compute_static_metrics(Ibar: np.ndarray, Fbar: np.ndarray, config: Config) -> dict:
    """NPV / ROI / ROI_disc / PI on a single mean cash-flow stream (NOK). No IRR/MIRR."""
    delta = config.delta
    Ibar = np.asarray(Ibar, dtype=float)
    Fbar = np.asarray(Fbar, dtype=float)
    net = Fbar - Ibar
    disc = delta ** np.arange(net.shape[0], dtype=float)

    npv = float(np.sum(disc * net))
    pv_in = float(np.sum(disc * Fbar))
    pv_out = float(np.sum(disc * Ibar))
    sum_in = float(np.sum(Fbar))
    sum_out = float(np.sum(Ibar))

    roi = float(np.sum(net) / sum_out) if sum_out != 0.0 else None
    pi = pv_in / pv_out if pv_out != 0.0 else None
    # Discounted ROI = NPV / PV(CAPEX) = present-value return on deployed capital.
    roi_disc = (pv_in - pv_out) / pv_out if pv_out != 0.0 else None

    # Inline identities: roi_disc == PI - 1, plus definition round-trips.
    if roi_disc is not None:
        assert abs(roi_disc - (pi - 1.0)) < 1e-9, (
            f"ROI_disc != PI - 1 ({roi_disc:.12g} vs {pi - 1.0:.12g})."
        )
        assert abs(pv_out * (1.0 + roi_disc) - pv_in) < 1e-6, (
            "ROI_disc definition round-trip failed: pv_out*(1+roi_disc) != pv_in."
        )
        assert abs(npv - roi_disc * pv_out) < 1e-6, (
            "NPV != ROI_disc * PV(CAPEX)."
        )

    return {
        "NPV": npv,
        "ROI": roi,
        "ROI_disc": roi_disc,
        "PI": pi,
        "pv_in": pv_in,
        "pv_out": pv_out,
        "sum_in": sum_in,
        "sum_out": sum_out,
    }


# =============================================================================
# === DATA / CONFIRMATION ===
# =============================================================================

def _load_concept1(cfg: Config) -> tuple:
    """Load the canonical concept-1 optimiser CSVs and build all required matrices.

    Reads the lowered-kappa recalibrated dataset directly from
    roa/input_files/{BESS,PSH}_optimized_results_combined.csv (the canonical
    thesis dataset). The old *_concept1.csv files (old kappa) are no longer used.
    """
    data = load_optimizer_data(cfg, data_dir=_INPUT_DIR)

    revenue_bess = {q: precompute_revenue_matrix(cfg, data, "BESS", q) for q in cfg.q_bess_grid}
    charge_bess  = {q: precompute_charged_energy_matrix(cfg, data, "BESS", q) for q in cfg.q_bess_grid}
    opex_bess    = {q: build_opex_matrix(cfg, charge_bess[q], "BESS", q) for q in cfg.q_bess_grid}
    revenue_psh    = precompute_revenue_matrix(cfg, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh     = precompute_charged_energy_matrix(cfg, data, "PSH", None)
    return (data, revenue_bess, charge_bess, opex_bess,
            revenue_psh, revenue_psh_DA, charge_psh)


def _confirm_first(cfg: Config) -> None:
    """Print the WS1 v0.5 pre-conditions; stop hard if any is wrong."""
    terminal = compute_psh_terminal_value(cfg)
    H_bess = cfg.T_psh
    checks = [
        ("10 in cfg.q_bess_grid",        10 in cfg.q_bess_grid, f"grid={list(cfg.q_bess_grid)}"),
        ("q0 = 10.0 MWh (baseline)",     _Q0 == 10.0,           f"q0={_Q0}"),
        ("cfg.T_psh == 67",              cfg.T_psh == 67,        f"T_psh={cfg.T_psh}"),
        ("terminal value == 0.0",        abs(terminal) < 1e-9,   f"terminal={terminal:.6g} NOK"),
        ("cfg.T_max == 80 (from data)",  cfg.T_max == 80,        f"T_max={cfg.T_max}"),
    ]
    print("=" * 72)
    print("  WS1 v0.5 — CONFIRM FIRST")
    print("=" * 72)
    all_ok = True
    for label, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'BAD'}] {label:30s} ({detail})")
        all_ok = all_ok and ok
    print(f"  H_bess (BESS horizon) := cfg.T_psh = {H_bess}")
    print("=" * 72)
    if not all_ok:
        print("CONFIRM FIRST failed — halting before any metric is computed.")
        sys.exit(1)


# =============================================================================
# === DRIVER ===
# =============================================================================

def run() -> list[dict]:
    cfg_base = Config()
    cfg_base.psh_concept = 1
    cfg_base.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    cfg_base.elavgift_nok_per_mwh_psh = ELAVGIFT_SCENARIO_VALUES["full"]

    (data, revenue_bess, charge_bess, opex_bess,
     revenue_psh, revenue_psh_DA, charge_psh) = _load_concept1(cfg_base)

    _confirm_first(cfg_base)

    price_state = data.mean_DA
    fcr_state = data.mean_FCR
    capex_paths = simulate_bess_capex(cfg_base, _SEED)
    delta = cfg_base.delta

    # ---- STEP 2/3: BESS no-flex baseline (scenario-invariant) ----
    nf = replay_bess_noflex(cfg_base, revenue_bess, opex_bess, capex_paths)
    I_nf, F_nf = nf["I"], nf["F"]
    m_bess = compute_static_metrics(I_nf.mean(axis=0), F_nf.mean(axis=0), cfg_base)

    # ---- STEP 4: ordering invariant against invest-now + flexibility ----
    disc = delta ** np.arange(I_nf.shape[1], dtype=float)
    npv_noflex_i = (F_nf - I_nf) @ disc                       # (N,)

    p1 = phase1_bess_compound(
        cfg_base, revenue_bess, opex_bess, capex_paths, price_state, fcr_paths=fcr_state,
    )
    pi_bess_t0 = p1["Pi_bess"][:, 0]                          # (N,) invest-now + flexibility
    flex_strip = pi_bess_t0 - npv_noflex_i                    # (N,) >= 0 in the mean
    strip_mean = float(flex_strip.mean())
    strip_p5 = float(np.percentile(flex_strip, 5))
    pi_bess_mean = float(pi_bess_t0.mean())
    noflex_mean = float(npv_noflex_i.mean())

    # Per-path diagnostics only: Pi_bess is an LSMC value estimate (stopping
    # decisions on fitted continuations), so it does NOT dominate the feasible
    # no-flex policy path-by-path. The economically meaningful invariant is at
    # the mean: mean(Pi_bess) must exceed mean(NPV_noflex) comfortably.
    n_viol = int(np.sum(npv_noflex_i > pi_bess_t0 + 1.0))
    frac_viol = n_viol / npv_noflex_i.shape[0]
    comfortable = strip_mean > 0.01 * abs(pi_bess_mean)       # not ~0 / near-equality
    assert comfortable and strip_mean > 0.0, (
        f"Flexibility strip FAILED at the mean level: mean strip "
        f"{strip_mean/1e6:.3f} MNOK is not comfortably > 0 "
        f"(mean Pi_bess {pi_bess_mean/1e6:.3f} vs no-flex {noflex_mean/1e6:.3f} MNOK)."
    )

    print()
    print("=" * 72)
    print("  STEP 4 — flexibility strip (Pi_bess[:,0] - NPV_noflex)")
    print("=" * 72)
    print(f"  mean(NPV_noflex)                           : {noflex_mean/1e6:,.3f} MNOK")
    print(f"  mean(Pi_bess[:,0])                         : {pi_bess_mean/1e6:,.3f} MNOK")
    print("    ^ invest-now + operating flexibility (diagnostic, NOT the base)")
    print(f"  mean flexibility strip                     : {strip_mean/1e6:,.3f} MNOK"
          f"  ({'OK, comfortably > 0' if comfortable else 'WARNING: near-equality'})")
    print(f"  P5   flexibility strip                     : {strip_p5/1e6:,.3f} MNOK")
    print(f"  per-path ordering violations (diagnostic)  : {n_viol}/{npv_noflex_i.shape[0]} "
          f"({frac_viol:.1%})")
    print("    ^ expected for an LSMC estimate; not a per-path lower bound")
    print("=" * 72)

    rows: list[dict] = [{
        "technology": "BESS",
        "scenario": "noflex_67y",
        **m_bess,
        "pi_bess_diag_MNOK": pi_bess_mean / 1e6,
    }]

    # ---- STEP 5: PSH one-shot at t=0, all five scenarios ----
    for scenario_name, elavgift_key, gr_mode in SCENARIO_DEFINITIONS:
        cfg_s = _make_scenario_cfg(
            cfg_base,
            elavgift_psh=ELAVGIFT_SCENARIO_VALUES[elavgift_key],
            grunnrente_mode=gr_mode,
        )
        opex_psh = build_opex_matrix(
            cfg_s, charge_psh, "PSH", None, production_matrix=data.psh_bid_DA_sell,
        )
        L = cfg_s.psh_lifetime(0)
        comp = compute_psh_cashflow_components(
            cfg_s,
            revenue_psh[:, 0:L],
            revenue_psh_DA[:, 0:L],
            opex_psh[:, 0:L],
            year_idx=0,
        )
        m_psh = compute_static_metrics(comp["I"].mean(axis=0), comp["F"].mean(axis=0), cfg_s)

        # cross-check: PSH NPV == compute_psh_payoffs(...)[:, 0].mean(), rel < 1e-9
        Pi_psh = compute_psh_payoffs(cfg_s, revenue_psh, price_state, opex_psh, revenue_psh_DA)
        pi_psh_t0 = float(Pi_psh[:, 0].mean())
        rel_psh = abs(m_psh["NPV"] - pi_psh_t0) / max(abs(pi_psh_t0), 1.0)
        assert rel_psh < 1e-9, (
            f"PSH NPV cross-check failed ({scenario_name}): NPV={m_psh['NPV']:.6g} "
            f"vs Pi_psh[:,0].mean()={pi_psh_t0:.6g} (rel={rel_psh:.3e})."
        )

        rows.append({
            "technology": "PSH",
            "scenario": scenario_name,
            **m_psh,
            "pi_bess_diag_MNOK": None,
        })

    return rows


# =============================================================================
# === OUTPUT ===
# =============================================================================

def _write_and_print(rows: list[dict]) -> Path:
    import csv

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / "static_metrics.csv"
    cols = ["technology", "scenario", "NPV_MNOK", "ROI_pct", "ROI_disc_pct",
            "PI", "pi_bess_diag_MNOK"]

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["technology"],
                r["scenario"],
                f"{r['NPV'] / 1e6:.6f}",
                "" if r["ROI"] is None else f"{r['ROI'] * 100:.6f}",
                "" if r["ROI_disc"] is None else f"{r['ROI_disc'] * 100:.6f}",
                "" if r["PI"] is None else f"{r['PI']:.6f}",
                "" if r["pi_bess_diag_MNOK"] is None else f"{r['pi_bess_diag_MNOK']:.6f}",
            ])

    # Per-row aggregate verification block for hand-checking the discounting gap.
    print()
    print("=" * 82)
    print("  Per-row aggregate verification (NOK aggregates -> MNOK)")
    print("=" * 82)
    for r in rows:
        roi = "n/a" if r["ROI"] is None else f"{r['ROI'] * 100:.3f}%"
        roi_d = "n/a" if r["ROI_disc"] is None else f"{r['ROI_disc'] * 100:.3f}%"
        pi = "n/a" if r["PI"] is None else f"{r['PI']:.4f}"
        print(f"  {r['technology']} / {r['scenario']}")
        print(f"      sum_I  (undisc)   : {r['sum_out'] / 1e6:>12,.3f} MNOK")
        print(f"      sum_F  (undisc)   : {r['sum_in'] / 1e6:>12,.3f} MNOK")
        print(f"      PV_CAPEX = pv_out : {r['pv_out'] / 1e6:>12,.3f} MNOK")
        print(f"      PV_inflows = pv_in: {r['pv_in'] / 1e6:>12,.3f} MNOK")
        print(f"      NPV               : {r['NPV'] / 1e6:>12,.3f} MNOK")
        print(f"      ROI      (undisc) : {roi:>13s}")
        print(f"      ROI_disc          : {roi_d:>13s}")
        print(f"      PI                : {pi:>13s}")
    print("=" * 82)

    print()
    print("=" * 94)
    print("  Static no-flex baseline metrics at t=0  (Concept 1, seed 42)")
    print("=" * 94)
    header = (f"  {'Tech':5s} {'Scenario':18s} {'NPV [MNOK]':>12s} {'ROI [%]':>10s} "
              f"{'ROI_disc [%]':>13s} {'PI':>8s} {'pi_bess_diag [MNOK]':>20s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        roi = "n/a" if r["ROI"] is None else f"{r['ROI'] * 100:.2f}"
        roi_d = "n/a" if r["ROI_disc"] is None else f"{r['ROI_disc'] * 100:.2f}"
        pi = "n/a" if r["PI"] is None else f"{r['PI']:.3f}"
        diag = "" if r["pi_bess_diag_MNOK"] is None else f"{r['pi_bess_diag_MNOK']:,.3f}"
        print(f"  {r['technology']:5s} {r['scenario']:18s} "
              f"{r['NPV'] / 1e6:>12,.3f} {roi:>10s} {roi_d:>13s} {pi:>8s} {diag:>20s}")
    print("=" * 94)
    print(f"  Saved: {out_path}")
    return out_path


def main() -> None:
    rows = run()
    _write_and_print(rows)


if __name__ == "__main__":
    main()
