"""
Investment Decision Algorithm -- real options valuation via LSMC.

Phase 1: Solves the BESS compound reinvestment chain backward from the terminal
         date. At each year, the investor can reinvest at any capacity on a discrete
         grid, wait one year, or exit permanently. The output is V_bess[i,t] (total
         compound option value) and Pi_bess[i,t] (value of entering the BESS chain).

Phase 2: Solves the joint mutually exclusive investment decision following
         Areal, Rodrigues, and Armada (2008). At each deferral date, the investor
         compares the best immediate payoff across BESS and PSH against a single
         estimated continuation value. Technology choice and exercise timing are
         optimized simultaneously.

All monetary values are in NOK.
"""

# All monetary values are in NOK unless explicitly noted.

from __future__ import annotations

import math
import time

import numpy as np

from roa.config import Config
from roa.lsmc import (
    build_basis_matrix,
    fit_continuation,
    normalize_state,
    compute_ncf,
    compute_opnpv,
    compute_psh_exercise_payoff,
)


# =============================================================================
# === PHASE 1: BESS COMPOUND REINVESTMENT CHAIN ===
# =============================================================================
#
# Phase 1 implements the BESS compound option valuation described in Areal,
# Rodrigues & Armada (2008). The investor manages a chain of BESS investment
# cycles over the full horizon T_max. At every year t the investor chooses:
#
#   * Reinvest at capacity q in Q_bess_grid (q > 0):
#         pay CAPEX, operate for T_bess years, then re-face the decision
#   * Wait (idle one year, zero cost/revenue, option preserved)
#   * Exit permanently (value = 0)
#
# State variables: annual average spot price P[i, t] and BESS unit CAPEX K[i, t].
#
# The backward induction follows the Longstaff-Schwartz discipline:
#   - LSMC regression gives the *estimated* wait value (for decision-making).
#   - The *stored* value is always the actual discounted future value,
#     not the fitted approximation.


def _compute_reinvest_values(
    config: Config,
    revenue_bess: dict[float, np.ndarray],
    opex_bess: dict[float, np.ndarray],
    capex_paths: np.ndarray,
    t: int,
    V_future: np.ndarray,
    has_continuation: bool,
) -> tuple[np.ndarray, list[float]]:
    """Compute the reinvestment value for each nonzero capacity at year t.

    Parameters
    ----------
    config           : Config instance.
    revenue_bess     : dict {q -> (N, T_max)} of annual gross revenues (NOK).
    opex_bess        : dict {q -> (N, T_max)} of annual OPEX (NOK).
    capex_paths      : (N, T_max+1) BESS unit CAPEX (NOK/MWh).
    t                : current decision year.
    V_future         : (N,) -- V_bess[:, t + T_bess] if has_continuation, else zeros.
    has_continuation : whether this cycle earns a continuation value at its end.
                       True only when the cycle is a full T_bess cycle and its
                       end index is strictly before T_max.

    Returns
    -------
    reinvest_vals : (N, num_q) -- value per path per capacity (NOK). -inf if infeasible.
    bess_qs       : list of capacities in the same column order.
    """
    bess_qs = list(config.q_bess_grid)
    N = capex_paths.shape[0]
    T_bess = config.T_bess
    delta = config.delta

    L = config.bess_cycle_length(t)          # available operating years from t inside horizon

    reinvest_vals = np.full((N, len(bess_qs)), -np.inf, dtype=float)

    if L <= 0:
        return reinvest_vals, bess_qs
    assert t + L <= config.T_max, f"C2a BESS: t={t}, L={L}, t+L={t+L} > T_max={config.T_max}"

    # Continuation discount: value of the chain at t+T_bess discounted back T_bess steps.
    # Added only for full, non-terminal cycles (gated by has_continuation).
    if has_continuation:
        continuation = (delta ** T_bess) * V_future    # (N,)
    else:
        continuation = 0.0                             # truncated or terminal cycle

    for qi, q in enumerate(bess_qs):
        rev_slice  = revenue_bess[q][:, t : t + L]    # (N, L)
        opex_slice = opex_bess[q][:, t : t + L]       # (N, L) from build_opex_matrix
        total_capex = capex_paths[:, t] * q            # (N,) NOK

        ncf = compute_ncf(rev_slice, opex_slice, total_capex, config, L)
        opnpv = compute_opnpv(ncf, config)             # (N,)

        reinvest_vals[:, qi] = -total_capex + opnpv + continuation

    return reinvest_vals, bess_qs


_BASIS_VAR_NAMES_DEFAULT = ("DA", "FCR", "K")


def _select_state_arrays(
    basis_vars: tuple[str, ...],
    price_paths: np.ndarray,
    fcr_paths: np.ndarray,
    capex_paths: np.ndarray,
) -> list[np.ndarray]:
    """Map basis_var names to the corresponding state-variable arrays."""
    available = {"DA": price_paths, "FCR": fcr_paths, "K": capex_paths}
    out: list[np.ndarray] = []
    for v in basis_vars:
        if v not in available:
            raise ValueError(f"Unknown basis var {v!r}; allowed: {sorted(available)}.")
        out.append(available[v])
    return out


def phase1_bess_compound(
    config: Config,
    revenue_bess: dict[float, np.ndarray],
    opex_bess: dict[float, np.ndarray],
    capex_paths: np.ndarray,
    price_paths: np.ndarray,
    fcr_paths: np.ndarray,
    basis_vars: tuple[str, ...] = _BASIS_VAR_NAMES_DEFAULT,
    basis_degree: int | None = None,
) -> dict:
    """BESS compound reinvestment chain backward induction.

    Parameters
    ----------
    config        : Config instance.
    revenue_bess  : dict mapping capacity q (float, MWh) -> array (N, T_max) (NOK).
                    Keys are the nonzero elements of config.q_bess_grid.
    opex_bess     : dict mapping capacity q -> (N, T_max) annual OPEX (NOK/year).
    capex_paths   : (N, T_max+1) -- BESS unit CAPEX (NOK/MWh); column 0 = year 0.
    price_paths   : (N, T_max) -- annual average spot price (NOK/MWh).
    fcr_paths     : (N, T_max) -- annual average FCR price (NOK/MW); third
                    state variable for the LSMC continuation regression.

    Returns
    -------
    dict with keys:
        'V_bess'      : (N, T_max+1) -- total compound option value at each node.
                        V_bess[i, t] = value of optimally managing BESS from year t
                        onward on path i, stored as the ACTUAL (not fitted) value.
        'Pi_bess'     : (N, T_max+1) -- best "enter the chain now" value at each node.
                        Pi_bess[i, t] = max over q of (-CAPEX_q + OpNPV_q + delta^T_bess * V[t+T_bess]).
                        This is what Phase 2 uses as the BESS component of joint value.
        'optimal_q'   : (N, T_max+1) -- capacity chosen at each node (MWh).
                        0 = exit, -1 = wait, positive = reinvest at that MWh.
        'exercise_map': (N, T_max+1) int8 -- action label.
                        1 = reinvest, 2 = wait, 3 = exit, 0 = terminal/no decision.
        '_elapsed_s'  : float -- wall-clock runtime in seconds.
    """
    t0_wall = time.perf_counter()

    N = capex_paths.shape[0]
    T_max = config.T_max
    delta = config.delta
    bess_qs = np.asarray(config.q_bess_grid, dtype=float)

    # Output arrays — all initialised to 0 / terminal values.
    # V_bess[:, T_max] = 0.0 is the explicit terminal boundary condition.
    V_bess      = np.zeros((N, T_max + 1), dtype=float)
    Pi_bess     = np.zeros((N, T_max + 1), dtype=float)
    optimal_q   = np.zeros((N, T_max + 1), dtype=float)
    exercise_map = np.zeros((N, T_max + 1), dtype=np.int8)

    # Minimum ITM paths needed for a stable lstsq fit.
    # Basis: weighted Laguerre tensor product, total degree <= d in
    # n_state_vars variables.  M = C(d + n_state_vars, n_state_vars).
    # Production default: d=3, n=2 (DA, K) -> M=10.
    d = config.laguerre_degree if basis_degree is None else basis_degree
    n_state_vars = len(basis_vars)
    if not (2 <= n_state_vars <= 3):
        raise ValueError(
            f"basis_vars must have 2 or 3 entries, got {basis_vars!r}"
        )
    state_pool = _select_state_arrays(
        basis_vars, price_paths, fcr_paths, capex_paths
    )
    M = math.comb(d + n_state_vars, n_state_vars)
    min_itm = M + 1
    convergence_ratio = M ** 3 / config.N
    print(
        f"[Phase 1] Basis size M={M} (degree={d}, {n_state_vars} state vars "
        f"{tuple(basis_vars)}), N={config.N:,}, M³/N={convergence_ratio:.4f}"
    )
    phase1_r2_t10: float | None = None

    assert V_bess.shape == (N, T_max + 1)
    assert np.all(V_bess[:, T_max] == 0.0)   # explicit terminal boundary

    # -----------------------------------------------------------------------
    # Backward induction: t = T_max-1 down to 0
    # -----------------------------------------------------------------------
    for t in range(T_max - 1, -1, -1):

        # Wait value: actual discounted future value (used for storage)
        actual_wait = delta * V_bess[:, t + 1]          # (N,)

        # Admissibility gate: skip reinvestment branch when no revenue year fits.
        # bess_admissible(t) is False only at t = T_max - 1.
        if not config.bess_admissible(t):
            V_bess[:, t] = np.maximum(0.0, actual_wait)
            continue

        # Continuation gate: True only for full T_bess cycles ending before T_max.
        has_continuation = config.bess_has_continuation(t)
        V_future = (
            V_bess[:, t + config.T_bess] if has_continuation
            else np.zeros(N, dtype=float)
        )

        # --- 1. Reinvestment values for all capacities ---
        reinvest_vals, _ = _compute_reinvest_values(
            config, revenue_bess, opex_bess, capex_paths, t, V_future, has_continuation
        )                                               # (N, num_q)

        # Best reinvestment: argmax over capacities, elementwise over paths
        best_q_idx = np.argmax(reinvest_vals, axis=1)  # (N,)
        best_reinvest = reinvest_vals[np.arange(N), best_q_idx]  # (N,) NOK
        best_q_vals = bess_qs[best_q_idx]              # (N,)

        # Mask for paths where reinvestment is at least feasible (finite value)
        reinvest_feasible = np.isfinite(best_reinvest)

        # Pi_bess: best entry-value regardless of whether we reinvest this step
        Pi_bess[:, t] = np.where(reinvest_feasible, best_reinvest, 0.0)

        # --- 2. Wait value via LSMC regression ---
        # State: weighted-Laguerre tensor in basis_vars with total-degree
        # truncation; each component is median-normalised over the full N
        # paths at date t inside build_basis_matrix.
        basis = build_basis_matrix(
            [arr[:, t] for arr in state_pool],
            d,
        )                                               # (N, M)

        # ITM filter: standard Longstaff-Schwartz — condition the regression
        # sample on the immediate-exercise payoff being positive, not on the
        # realised continuation target. Symmetric with Phase 2's filter on E.
        itm_mask = best_reinvest > 0.0
        n_itm = int(itm_mask.sum())
        if n_itm >= min_itm:
            wait_fitted = fit_continuation(basis, actual_wait, itm_mask)
        else:
            # Too few ITM paths -- use raw target (no regression distortion)
            wait_fitted = actual_wait.copy()

        # Diagnostic at t=10: basis size and in-sample R² on ITM paths
        if t == 10 and n_itm >= min_itm:
            y_true = actual_wait[itm_mask]
            y_pred = wait_fitted[itm_mask]
            ss_res = float(((y_true - y_pred) ** 2).sum())
            ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
            r2_t10 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
            phase1_r2_t10 = r2_t10
            print(
                f"  [Phase 1 t=10] n_basis={basis.shape[1]}, "
                f"n_itm={n_itm}, R²={r2_t10:.4f}"
            )

        # --- 3. Decision: max(exit=0, wait_fitted, best_reinvest) ---
        # Use FITTED wait for the decision comparison only.
        eff_reinvest = np.where(reinvest_feasible, best_reinvest, -np.inf)

        reinvest_wins = (eff_reinvest >= wait_fitted) & (eff_reinvest >= 0.0)
        wait_wins     = (~reinvest_wins) & (wait_fitted > 0.0)
        # exit_wins = everything else (implicit: value = 0)

        # --- 4. Store ACTUAL values (not fitted) ---
        V_bess[:, t] = np.where(
            reinvest_wins, best_reinvest,
            np.where(wait_wins, actual_wait, 0.0)
        )

        optimal_q[:, t] = np.where(
            reinvest_wins, best_q_vals,
            np.where(wait_wins, -1.0, 0.0)
        )

        exercise_map[:, t] = np.where(
            reinvest_wins, np.int8(1),
            np.where(wait_wins, np.int8(2), np.int8(3))
        )

    assert np.all(np.isfinite(V_bess[:, 0]))

    elapsed = time.perf_counter() - t0_wall

    return {
        "V_bess":       V_bess,
        "Pi_bess":      Pi_bess,
        "optimal_q":    optimal_q,
        "exercise_map": exercise_map,
        "phase1_r2_t10": phase1_r2_t10,
        "M":            M,
        "_elapsed_s":   elapsed,      # diagnostic: wall-clock time
    }


# =============================================================================
# === PHASE 2: JOINT MUTUALLY EXCLUSIVE OPTIMIZATION (Areal et al. 2008) ===
# =============================================================================
#
# Phase 2 implements the Areal, Rodrigues & Armada (2008) LSMC framework for
# an American option over two mutually exclusive technologies: BESS and PSH.
#
# The investor holds a deferral option expiring at T_def. At each annual exercise
# date t in {0, 1, ..., T_def} she compares:
#
#   * Exercise BESS  -> payoff = Pi_bess[i, t]  (from Phase 1; embeds all future cycles)
#   * Exercise PSH   -> payoff = Pi_psh[i, t]   (single investment, T_psh operating years)
#   * Wait           -> estimated continuation value via LSMC on (price, capex)
#
# Key insight (Areal et al.): the maximum immediate payoff across BOTH technologies
# is compared against ONE estimated continuation value. Technology choice and timing
# are optimized simultaneously in a single backward sweep.
#
# State vector: bivariate (P̄, K) — same basis as Phase 1, M=10 terms.


def compute_psh_payoffs(
    config: Config,
    revenue_psh: np.ndarray,
    price_paths: np.ndarray,
    psh_opex_matrix: np.ndarray,
    revenue_psh_DA: np.ndarray | None = None,
) -> np.ndarray:
    """Compute PSH exercise payoff at each possible exercise date.

    Returns -inf for inadmissible deferral dates so that PSH is excluded
    from the joint max at those dates without treating it as zero payoff.

    Parameters
    ----------
    config          : Config instance.
    revenue_psh     : (N, T_max) -- annual gross PSH revenue per path per year (NOK).
    price_paths     : (N, T_max+1) -- annual avg spot price (NOK/MWh). Not used directly;
                      kept for API symmetry and future spot-price adjustments.
    psh_opex_matrix : (N, T_max) -- annual PSH OPEX from build_opex_matrix (NOK/year).
    revenue_psh_DA  : (N, T_max) -- annual gross PSH DA revenue for grunnrente base.
                      Required when grunnrente_mode != "none".

    Returns
    -------
    Pi_psh : (N, T_def+1) -- PSH exercise payoff (NOK) at each deferral date t.
             -inf where psh_admissible(t) is False.
    """
    N = revenue_psh.shape[0]
    T_def = config.T_def
    Pi_psh = np.empty((N, T_def + 1), dtype=float)
    if revenue_psh_DA is None:
        if config.grunnrente_mode != "none":
            raise ValueError("revenue_psh_DA is required when grunnrente_mode is not 'none'.")
        revenue_psh_DA = np.zeros_like(revenue_psh)

    for t in range(T_def + 1):
        if not config.psh_admissible(t):
            # PSH project can't complete within the horizon: exclude from max.
            Pi_psh[:, t] = -np.inf
            continue

        L = config.psh_lifetime(t)                            # respects truncation flag
        assert t + L <= config.T_max, f"C2a PSH: t={t}, L={L}, t+L={t+L} > T_max={config.T_max}"
        revenue_slice    = revenue_psh[:, t : t + L]          # (N, L)
        revenue_DA_slice = revenue_psh_DA[:, t : t + L]       # (N, L)
        opex_slice       = psh_opex_matrix[:, t : t + L]      # (N, L)

        Pi_psh[:, t] = compute_psh_exercise_payoff(
            config,
            revenue_slice,
            revenue_DA_slice,
            opex_slice,
            year_idx=t,
        )

    return Pi_psh


def phase2_joint_optimization(
    config: Config,
    Pi_bess: np.ndarray,
    Pi_psh: np.ndarray,
    price_paths: np.ndarray,
    capex_paths: np.ndarray,
    fcr_paths: np.ndarray,
    basis_vars: tuple[str, ...] = _BASIS_VAR_NAMES_DEFAULT,
    basis_degree: int | None = None,
) -> dict:
    """Areal et al. (2008) joint mutually exclusive LSMC.

    Parameters
    ----------
    config      : Config instance.
    Pi_bess     : (N, T_max+1) -- BESS entry value from Phase 1 (NOK).
                  Only columns 0..T_def are read.
    Pi_psh      : (N, T_def+1) -- PSH exercise payoff from compute_psh_payoffs (NOK).
                  -inf at inadmissible deferral dates.
    price_paths : (N, T_max) -- annual avg spot price (NOK/MWh).
    capex_paths : (N, T_max+1) -- BESS unit CAPEX (NOK/MWh); state variable for
                  the LSMC continuation regression alongside price_paths and fcr_paths.
    fcr_paths   : (N, T_max) -- annual avg FCR price (NOK/MW); third state
                  variable for the LSMC continuation regression.

    Returns
    -------
    dict with keys:
        'option_value'      : float  -- V_0 (stopping-time estimator, NOK).
        'V'                 : (N,)   -- realized option value per path at t=0 (NOK).
        'V_table'           : (N, T_def+1) -- full realized value array.
        'exercise_times'    : (N,) int -- optimal exercise year; -1 = never exercised.
        'technology_choices': (N,) int -- 0=never, 1=BESS, 2=PSH.
        'exercise_rate'     : float -- fraction of paths that exercise.
        'bess_rate'         : float -- fraction choosing BESS (among exercisers).
        'psh_rate'          : float -- fraction choosing PSH (among exercisers).
        'V0_stoptime'       : float -- V_0 via stopping-time formula (NOK).
        'V0_recursion'      : float -- V_0 via V[:,0].mean() cross-check (NOK).
        'rel_gap_stoptime'  : float -- |recursion - stoptime| / stoptime.
        'exercise_hist'     : dict {t -> count} -- exercise count per deferral date.
        '_elapsed_s'        : float -- wall-clock runtime in seconds.
    """
    t0_wall = time.perf_counter()

    N = price_paths.shape[0]
    T_def = config.T_def
    delta = config.delta
    d = config.laguerre_degree if basis_degree is None else basis_degree

    # Basis: weighted Laguerre tensor with total-degree truncation in
    # n_state_vars variables. M = C(d + n_state_vars, n_state_vars).
    n_state_vars = len(basis_vars)
    if not (2 <= n_state_vars <= 3):
        raise ValueError(
            f"basis_vars must have 2 or 3 entries, got {basis_vars!r}"
        )
    state_pool = _select_state_arrays(
        basis_vars, price_paths, fcr_paths, capex_paths
    )
    M = math.comb(d + n_state_vars, n_state_vars)
    min_itm = M + 1
    convergence_ratio = M ** 3 / config.N
    print(
        f"[Phase 2] Basis size M={M} (degree={d}, {n_state_vars} state vars "
        f"{tuple(basis_vars)}), N={N:,}, M³/N={convergence_ratio:.4f}"
    )
    phase2_r2_t5: float | None = None

    # Slice Pi_bess to the deferral window
    Pi_bess_window = Pi_bess[:, : T_def + 1]             # (N, T_def+1)
    Pi_psh_window  = Pi_psh                               # (N, T_def+1)

    # Joint exercise payoff: max over technologies (−inf where both inadmissible)
    E = np.maximum(Pi_bess_window, Pi_psh_window)         # (N, T_def+1)
    # Tie-break: BESS wins when equal (measure-zero event on continuous paths)
    bess_wins = Pi_bess_window >= Pi_psh_window            # (N, T_def+1)

    assert Pi_bess_window.shape == (N, T_def + 1)
    assert Pi_psh_window.shape  == (N, T_def + 1)
    # Pi_bess >= 0 by construction, so E is always finite under any config.
    assert np.all(np.isfinite(E)), "E has -inf entries; unexpected under default config"

    # V array: realized value at each (path, deferral date).
    V = np.full((N, T_def + 1), np.nan)
    tau_star = np.full(N, -1, dtype=int)    # optimal exercise time; -1 = no exercise

    # -----------------------------------------------------------------------
    # Terminal date t = T_def
    # -----------------------------------------------------------------------
    V[:, T_def] = np.maximum(E[:, T_def], 0.0)
    exercised_term = E[:, T_def] > 0.0
    tau_star[exercised_term] = T_def

    # -----------------------------------------------------------------------
    # Backward induction: t = T_def - 1 down to 0
    # -----------------------------------------------------------------------
    for t in range(T_def - 1, -1, -1):

        wait_realized = delta * V[:, t + 1]              # (N,) actual discounted future
        exercise_now  = E[:, t]                          # (N,) joint payoff

        # ITM filter: paths where exercising is strictly positive
        itm = exercise_now > 0.0
        n_itm = int(itm.sum())

        if n_itm >= min_itm:
            # Per-date median normalisation handled inside
            # build_basis_matrix → normalize_state.
            basis = build_basis_matrix(
                [arr[:, t] for arr in state_pool],
                d,
            )                                            # (N, M)
            cont_hat = fit_continuation(basis, wait_realized, itm_mask=itm)  # (N,)
        else:
            # Too few ITM paths — use realized wait directly (no regression)
            cont_hat = wait_realized

        # Diagnostic at t=5: basis size and in-sample R² on ITM paths
        if t == 5 and n_itm >= min_itm:
            y_true = wait_realized[itm]
            y_pred = cont_hat[itm]
            ss_res = float(((y_true - y_pred) ** 2).sum())
            ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
            r2_t5 = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0
            phase2_r2_t5 = r2_t5
            print(
                f"  [Phase 2 t=5] n_basis={basis.shape[1]}, "
                f"n_itm={n_itm}, R²={r2_t5:.4f}"
            )

        # Decision: exercise if positive and beats estimated continuation
        decide_exercise = itm & (exercise_now >= cont_hat)

        # Store REALIZED values (not fitted)
        V[:, t] = np.where(decide_exercise, exercise_now, wait_realized)

        # Overwrite tau_star with earlier exercise time (backward = earlier wins)
        tau_star = np.where(decide_exercise, t, tau_star)

    assert V.shape == (N, T_def + 1)
    assert np.all(np.isfinite(V[:, 0]))
    assert tau_star.shape == (N,)

    # -----------------------------------------------------------------------
    # Final option value — stopping-time and recursion cross-check
    # -----------------------------------------------------------------------
    exercised_any = tau_star >= 0
    safe_tau = np.maximum(tau_star, 0)                   # avoid negative indexing

    payoff_at_stop = np.where(exercised_any, E[np.arange(N), safe_tau], 0.0)
    disc_factors   = delta ** safe_tau
    V0_stoptime    = float((disc_factors * payoff_at_stop * exercised_any).mean())
    V0_recursion   = float(V[:, 0].mean())
    rel_gap = (
        abs(V0_recursion - V0_stoptime) / max(abs(V0_stoptime), 1e-10)
    )

    # -----------------------------------------------------------------------
    # Decomposition statistics
    # -----------------------------------------------------------------------
    n_ex = int(exercised_any.sum())
    exercise_rate = float(exercised_any.mean())

    if n_ex > 0:
        bess_at_stop = bess_wins[np.arange(N), safe_tau] & exercised_any
        n_bess = int(bess_at_stop.sum())
        n_psh  = n_ex - n_bess
        bess_rate = n_bess / n_ex
        psh_rate  = n_psh  / n_ex
    else:
        n_bess = n_psh = 0
        bess_rate = psh_rate = 0.0

    # technology_choices: 0=never, 1=BESS, 2=PSH (for API compatibility)
    tech_choices = np.zeros(N, dtype=int)
    if n_ex > 0:
        tech_choices[exercised_any] = np.where(
            bess_wins[exercised_any, safe_tau[exercised_any]], 1, 2
        )

    # Exercise-time histogram
    exercise_hist = {t: int((tau_star == t).sum()) for t in range(T_def + 1)}

    elapsed = time.perf_counter() - t0_wall

    return {
        "option_value":       V0_stoptime,
        "V":                  V[:, 0],
        "V_table":            V,
        "exercise_times":     tau_star,
        "technology_choices": tech_choices,
        "exercise_rate":      exercise_rate,
        "bess_rate":          bess_rate,
        "psh_rate":           psh_rate,
        "V0_stoptime":        V0_stoptime,
        "V0_recursion":       V0_recursion,
        "rel_gap_stoptime":   rel_gap,
        "exercise_hist":      exercise_hist,
        "phase2_r2_t5":       phase2_r2_t5,
        "M":                  M,
        "_elapsed_s":         elapsed,
    }
