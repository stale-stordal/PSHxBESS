"""Plotting functions for ROA pipeline output.

Each function takes loaded data and an output path and saves a single PDF.
After the 2026-05 cleanup, only the two thesis figures remain:
``pi_distributions_t0.pdf`` and ``capex_heatmap.pdf``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .loader import PipelineResults, load_results
from .style import (
    COLORS,
    SCENARIO_COLORS,
    SCENARIO_LABELS_SHORT,
    SCENARIO_ORDER,
    SCENARIO_ORDER_BY_PI_PSH_MEAN,
    TECH_COLORS,
    apply_style,
)


def _ordered_scenarios(results: PipelineResults) -> list[str]:
    return [s for s in SCENARIO_ORDER if s in results.scenarios]


def collect_pi_t0_k30(
    K: int = 30,
    seed_start: int = 42,
    psh_concept: int = 1,
) -> dict[str, np.ndarray]:
    """Run Phase 1 across K BESS-CAPEX seeds and stack Π^{BESS}_{i,0} and Π^{PSH}_{i,0}.

    Π^{BESS}_{i,0} depends on the GBM seed (LSMC continuation noise);
    Π^{PSH}_{i,0} is path-deterministic given the tax parameters, so it is
    computed once per scenario. Returns:
        {
          "Pi_bess_stacked":         (K * N,) array in NOK,
          "Pi_psh_<scenario_name>":  (N,)     array in NOK, one entry per scenario.
        }
    """
    # Imports are local to keep the plotting module's top-level imports light.
    from roa.config import Config, ELAVGIFT_SCENARIO_VALUES, SCENARIO_DEFINITIONS
    from roa.investment import compute_psh_payoffs, phase1_bess_compound
    from roa.lsmc import (
        build_opex_matrix,
        load_optimizer_data,
        precompute_charged_energy_matrix,
        precompute_psh_da_revenue_matrix,
        precompute_revenue_matrix,
        simulate_bess_capex,
    )

    cfg = Config()
    cfg.psh_concept = psh_concept

    # Canonical thesis dataset: Concept-1 lowered-kappa combined CSVs.
    data = load_optimizer_data(cfg)

    price_state = data.mean_DA
    fcr_state = data.mean_FCR

    qs = list(cfg.q_bess_grid)
    revenue_bess = {q: precompute_revenue_matrix(cfg, data, "BESS", q) for q in qs}
    charge_bess = {q: precompute_charged_energy_matrix(cfg, data, "BESS", q) for q in qs}
    opex_bess = {q: build_opex_matrix(cfg, charge_bess[q], "BESS", q) for q in qs}
    revenue_psh = precompute_revenue_matrix(cfg, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh = precompute_charged_energy_matrix(cfg, data, "PSH", None)

    # Π^{BESS}_{i,0} across K seeds (stack).
    bess_stacks = []
    for k in range(K):
        seed = seed_start + k
        capex_paths = simulate_bess_capex(cfg, seed)
        p1 = phase1_bess_compound(
            cfg, revenue_bess, opex_bess, capex_paths, price_state,
            fcr_paths=fcr_state,
        )
        bess_stacks.append(p1["Pi_bess"][:, 0].copy())
    Pi_bess_stacked = np.concatenate(bess_stacks)

    # Π^{PSH}_{i,0} per scenario (seed-independent).
    out: dict[str, np.ndarray] = {"Pi_bess_stacked": Pi_bess_stacked}
    for scenario_name, elavgift_key, gr_mode in SCENARIO_DEFINITIONS:
        cfg_s = Config()
        cfg_s.psh_concept = psh_concept
        cfg_s.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
        cfg_s.elavgift_nok_per_mwh_psh = ELAVGIFT_SCENARIO_VALUES[elavgift_key]
        cfg_s.grunnrente_mode = gr_mode
        cfg_s.__post_init__()
        cfg_s.N = cfg.N
        cfg_s.T_max = cfg.T_max
        cfg_s.T_def = cfg.T_def
        opex_psh = build_opex_matrix(
            cfg_s, charge_psh, "PSH", None, production_matrix=data.psh_bid_DA_sell,
        )
        Pi_psh_full = compute_psh_payoffs(
            cfg_s, revenue_psh, price_state, opex_psh, revenue_psh_DA,
        )
        out[f"Pi_psh_{scenario_name}"] = Pi_psh_full[:, 0].copy()
    return out

def plot_pi_distributions_t0(
    results: PipelineResults,
    output_path: str | Path,
    bins: int = 50,
    k30_data: dict[str, np.ndarray] | None = None,
) -> None:
    """Two-panel payoff distribution figure for §5.3.

    Top:    Single BESS Π^{BESS}_{i,0} histogram (scenario-invariant by
            construction; BESS is exempt from the resource rent and its
            electricity tax is fixed at the full rate).
    Bottom: Forest plot of Π^{PSH}_{i,0} per scenario — marker at the mean,
            horizontal whisker spanning p5–p95. Rows ordered from least-
            negative to most-negative mean. A vertical reference at the BESS
            mean (in BNOK on the PSH axis) makes the magnitude gap visible.

    When ``k30_data`` is provided (see ``collect_pi_t0_k30``), the BESS panel
    histograms a K-seed stack and the PSH whiskers read the seed-deterministic
    per-scenario arrays. Otherwise the single-seed arrays in ``results`` are used.
    """
    apply_style()
    scenarios = _ordered_scenarios(results)

    if k30_data is not None:
        pi_bess_mnok = k30_data["Pi_bess_stacked"] / 1e6
        psh_arrays = {s: k30_data[f"Pi_psh_{s}"] / 1e9 for s in scenarios}
    else:
        pi_bess_mnok = results.scenarios[scenarios[0]].Pi_BESS_paths_t0 / 1e6
        psh_arrays = {s: results.scenarios[s].Pi_PSH_paths_t0 / 1e9 for s in scenarios}

    bess_mean_mnok = float(pi_bess_mnok.mean())
    bess_mean_bnok = bess_mean_mnok / 1e3  # converted to BNOK for PSH-axis reference

    psh_stats = {
        s: {
            "mean": float(arr.mean()),
            "p5":   float(np.percentile(arr, 5)),
            "p95":  float(np.percentile(arr, 95)),
        }
        for s, arr in psh_arrays.items()
    }
    # Forest-plot row order: least-negative mean at the top → most-negative at the bottom.
    ordered = sorted(scenarios, key=lambda s: psh_stats[s]["mean"], reverse=True)

    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[2.4, 1.6], hspace=0.35,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])

    # ---- Top: BESS histogram ----
    ax_top.hist(pi_bess_mnok, bins=bins, color=TECH_COLORS["BESS"],
                alpha=0.85, edgecolor="none")
    ax_top.axvline(bess_mean_mnok, color=COLORS.get("highlight", "#C44E52"),
                   lw=1.6, ls="--")
    ax_top.set_xlabel(r"$\Pi^{\mathrm{BESS}}_{i,0}$ (MNOK)", labelpad=6)
    ax_top.set_ylabel("Count")
    ax_top.grid(axis="y")
    ax_top.set_axisbelow(True)

    # Inline mean label next to the dashed line — not a separate legend entry.
    y_lim_top = ax_top.get_ylim()
    ax_top.text(
        bess_mean_mnok, y_lim_top[1] * 0.95,
        f"  Mean = {bess_mean_mnok:,.1f} MNOK",
        color=COLORS.get("highlight", "#C44E52"),
        ha="left", va="top", fontsize=10,
    )

    # Scenario-invariance note sits centered above the BESS panel, on its own
    # line, so it doesn't compete with the mean label or the rising edge of
    # the histogram. (Sample-size detail is reported in the LaTeX caption.)
    ax_top.text(
        0.5, 1.04, "Scenario-invariant by construction",
        transform=ax_top.transAxes, ha="center", va="bottom",
        fontsize=10, color=COLORS.get("text", "#3A3A3A"),
    )

    # ---- Bottom: PSH forest plot ----
    y_positions = np.arange(len(ordered))[::-1].astype(float)  # top row = highest y
    for ypos, s in zip(y_positions, ordered):
        stats = psh_stats[s]
        ax_bot.hlines(ypos, stats["p5"], stats["p95"],
                      color=SCENARIO_COLORS[s], lw=2.4, alpha=0.9)
        ax_bot.plot([stats["p5"], stats["p95"]], [ypos, ypos],
                    marker="|", markersize=10, linestyle="None",
                    color=SCENARIO_COLORS[s])
        ax_bot.plot(stats["mean"], ypos, marker="o", markersize=8,
                    color=SCENARIO_COLORS[s],
                    markeredgecolor=COLORS.get("text", "#3A3A3A"),
                    markeredgewidth=0.8, zorder=4)

    ax_bot.set_yticks(y_positions)
    ax_bot.set_yticklabels([SCENARIO_LABELS_SHORT[s] for s in ordered])
    ax_bot.set_xlabel(r"$\Pi^{\mathrm{PSH}}_{i,0}$ (BNOK)")
    ax_bot.grid(axis="x")
    ax_bot.set_axisbelow(True)

    # Vertical reference at the BESS mean (in BNOK).
    ax_bot.axvline(bess_mean_bnok, color=COLORS.get("main", "#2F5C85"),
                   lw=1.4, ls="--", alpha=0.95)
    y_lo, y_hi = (-0.6, len(ordered) - 0.4)
    ax_bot.set_ylim(y_lo, y_hi)
    # Extend the PSH x-axis past +0.229 BNOK so the BESS reference line has
    # clear breathing room from the panel border.
    psh_x_lo = min(stats["p5"] for stats in psh_stats.values()) - 0.5
    ax_bot.set_xlim(psh_x_lo, 2.0)

    # BESS-mean reference callout: blended transform so the label sits centred
    # directly above the dashed line (data x) just outside the panel top (axes y).
    from matplotlib.transforms import blended_transform_factory
    _trans = blended_transform_factory(ax_bot.transData, ax_bot.transAxes)
    ax_bot.text(
        bess_mean_bnok, 1.02,
        f"BESS mean\n+{bess_mean_bnok:.3f} BNOK",
        transform=_trans, ha="center", va="bottom",
        color=COLORS.get("main", "#2F5C85"), fontsize=10,
    )

    # Forest-plot glyph legend placed *below* the panel as a single horizontal
    # row, so it does not visually compete with the scenario rows.
    from matplotlib.lines import Line2D
    fp_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=COLORS.get("neutral", "#4A4A4A"),
               markeredgecolor=COLORS.get("text", "#3A3A3A"),
               markersize=8, label="Mean"),
        Line2D([0], [0], color=COLORS.get("neutral", "#4A4A4A"), lw=2.4,
               label="5th–95th percentile"),
    ]
    ax_bot.legend(
        handles=fp_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.22),
        ncol=2, frameon=False, fontsize=10,
    )

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_capex_heatmap(
    sweep_path: str | Path,
    output_path: str | Path,
) -> dict[str, dict[str, float | None]]:
    """Single-panel CAPEX heatmap of P(PSH | exercise) with Concept guides.

    V_0 levels and threshold values appear in the Results.tex
    breakeven/crossover table; this figure carries only the regime-transition
    story (no markers). Returns the per-scenario
    {"breakeven_capex_bnok", "crossover_capex_bnok"} dict read from the sweep
    JSON's ``breakeven_summary`` so the CLI can still print the thresholds.
    """
    import matplotlib.colors as mcolors

    apply_style()
    with open(sweep_path) as f:
        d = json.load(f)
    capex_keys = sorted(d["sweep"].keys(), key=float)
    capex_vals = np.array([float(k) for k in capex_keys])
    available = set(d["sweep"][capex_keys[0]].keys())
    rows = [s for s in SCENARIO_ORDER_BY_PI_PSH_MEAN if s in available]
    summary = d.get("breakeven_summary", {})

    # P(PSH | exercise) matrix in row-order.
    Z = np.zeros((len(rows), len(capex_vals)), dtype=float)
    for i, s in enumerate(rows):
        for j, k in enumerate(capex_keys):
            Z[i, j] = d["sweep"][k][s]["p_psh_given_ex"]

    fig, ax = plt.subplots(figsize=(11, 5.4))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "psh_share", ["#FFFFFF", "#4C78A8", "#6F4E9C", "#C44E52"],
    )
    im = ax.imshow(
        Z, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0,
        extent=[capex_vals.min() - 0.25, capex_vals.max() + 0.25,
                len(rows) - 0.5, -0.5],
    )

    # Integer-only ticks: avoids the "3 4 4 4 5" duplicate-label artefact and
    # gives the cell annotations a clean grid to sit on.
    integer_ticks = [c for c in capex_vals if abs(c - round(c)) < 1e-9]
    ax.set_xticks(integer_ticks)
    ax.set_xticklabels([f"{int(round(c))}" for c in integer_ticks])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([SCENARIO_LABELS_SHORT[s] for s in rows])
    ax.set_xlabel("PSH CAPEX (BNOK)")

    # Cell labels at integer-BNOK positions only. Always print, even at 0.00,
    # so "PSH never wins here" is visible rather than inferred from a blank cell.
    for i in range(len(rows)):
        for j, c in enumerate(capex_vals):
            if abs(c - round(c)) > 1e-9:
                continue
            v = Z[i, j]
            color = "#FFFFFF" if v > 0.55 else COLORS.get("text", "#3A3A3A")
            ax.text(c, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    # ---- Breakeven / crossover thresholds (returned, not drawn) ----
    # The markers were removed at the methodology chat's request — the values
    # live in the Results.tex prose table instead. Keep the read-out so the
    # CLI in roa.plots can still print them.
    thresholds: dict[str, dict[str, float | None]] = {}
    for s in rows:
        be = summary.get(s, {}).get("breakeven_capex_bnok")
        xo = summary.get(s, {}).get("crossover_capex_bnok")
        thresholds[s] = {"breakeven_capex_bnok": be, "crossover_capex_bnok": xo}

    # ---- Concept guides ----
    ax.axvline(15.8, color=COLORS.get("text", "#3A3A3A"), lw=0.8, ls=":")
    ax.axvline(10.2, color=COLORS.get("text", "#3A3A3A"), lw=0.8, ls=":")
    label_y = -0.8
    ax.text(15.8, label_y, "Concept 2 (15.8 BNOK)",
            ha="center", va="bottom", fontsize=10,
            color=COLORS.get("text", "#3A3A3A"), clip_on=False)
    ax.text(10.2, label_y, "Concept 3 (10.2 BNOK)",
            ha="center", va="bottom", fontsize=10,
            color=COLORS.get("text", "#3A3A3A"), clip_on=False)

    ax.set_xlim(capex_vals.min() - 0.25, capex_vals.max() + 0.25)
    ax.set_ylim(len(rows) - 0.5, -0.9)
    ax.grid(False)

    # Colorbar — plain text + explicit rotation/labelpad (avoids the rotated-
    # mathtext kerning artefact).
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("P(PSH | exercise)", rotation=270, labelpad=15)

    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return thresholds

# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def make_all_plots(
    json_path: str | Path,
    output_dir: str | Path,
    sweep_path: str | Path | None = None,
    pi_k30: int | None = 30,
) -> dict:
    """Generate the two thesis figures.

    Parameters
    ----------
    json_path : path to main_pipeline_*.json (canonical pipeline output).
    output_dir : directory to write PDF figures into. Created if missing.
    sweep_path : path to psh_capex_sweep_*.json. Required to produce
                 ``capex_heatmap.pdf``; if None, only the Pi-distribution
                 figure is generated.
    pi_k30    : if a positive integer K, the Pi-distributions plot histograms
                K seeds of the BESS unit-cost GBM stacked into one sample
                (300,000 paths at K=30, N=10,000). Pass ``None`` to fall back
                to the single-seed arrays in the JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = load_results(json_path)

    figures: dict[str, str] = {}

    pi_pdf = output_dir / "pi_distributions_t0.pdf"
    k30_data = None
    if pi_k30 is not None and pi_k30 > 0:
        psh_concept = results.metadata.get("psh_concept", 1)
        print(f"[plots] Collecting Pi_(i,0) K={pi_k30} stacked arrays "
              f"(~{1.5 * pi_k30:.0f}s)...")
        k30_data = collect_pi_t0_k30(K=pi_k30, psh_concept=psh_concept)
    plot_pi_distributions_t0(results, pi_pdf, k30_data=k30_data)
    figures["pi_distributions_t0"] = str(pi_pdf)

    thresholds: dict[str, dict[str, float | None]] = {}
    if sweep_path is not None:
        heatmap_pdf = output_dir / "capex_heatmap.pdf"
        thresholds = plot_capex_heatmap(sweep_path, heatmap_pdf)
        figures["capex_heatmap"] = str(heatmap_pdf)

    return {"figures": figures, "thresholds_capex_bnok": thresholds}
