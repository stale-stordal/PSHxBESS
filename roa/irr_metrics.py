"""
Per-path internal rate of return (IRR) for the concept-1 investments.

Reconstructed from the per-path cash-flow builders in roa.lsmc /
roa.static_metrics. For every Monte-Carlo path we assemble the undiscounted
net cash-flow stream  net_k = F_k - I_k  (k = 0..H), solve  sum_k net_k /
(1+r)^k = 0  for the IRR r, then report the cross-path distribution.

Variants produced (concept 1):
    BESS_seq_detcapex_irr   : sequential no-flex reinvestment over H = T_psh,
                              deterministic (cross-path mean) CAPEX
    PSH_irr_GRsymmetric     : one-shot PSH at t=0, symmetric grunnrente

For each variant the per-path streams are written to <name>_concept1.csv
(path_id, irr, no_root), the summary distribution to <name>_summary_concept1.csv,
and a histogram to <plot>_hist_concept1.{pdf,png}.

Usage
-----
    cd masteroppgave
    python -m roa.irr_metrics
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from roa.config import ELAVGIFT_SCENARIO_VALUES, Config
from roa.lsmc import (
    build_opex_matrix,
    compute_ncf,
    compute_psh_cashflow_components,
    load_optimizer_data,
    precompute_charged_energy_matrix,
    precompute_psh_da_revenue_matrix,
    precompute_revenue_matrix,
    simulate_bess_capex,
)
from roa.static_metrics import _SEED, _Q0, replay_bess_noflex
from roa.main import _make_scenario_cfg

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INPUT_DIR = _REPO_ROOT / "roa" / "input_files"
_OUT_DIR = Path(__file__).resolve().parent / "plots" / "output"


# =============================================================================
# === DATA LOADING ===
# =============================================================================

def _load(cfg: Config, input_dir: Path) -> tuple:
    """Load the concept-1 optimiser CSVs (*_combined.csv) and build all matrices.

    Mirrors roa.static_metrics._load_concept1 but reads the combined files
    directly from *input_dir* instead of staging concept1-named copies.
    """
    data = load_optimizer_data(cfg, data_dir=input_dir)

    revenue_bess = {q: precompute_revenue_matrix(cfg, data, "BESS", q) for q in cfg.q_bess_grid}
    charge_bess  = {q: precompute_charged_energy_matrix(cfg, data, "BESS", q) for q in cfg.q_bess_grid}
    opex_bess    = {q: build_opex_matrix(cfg, charge_bess[q], "BESS", q) for q in cfg.q_bess_grid}
    revenue_psh    = precompute_revenue_matrix(cfg, data, "PSH", None)
    revenue_psh_DA = precompute_psh_da_revenue_matrix(data)
    charge_psh     = precompute_charged_energy_matrix(cfg, data, "PSH", None)
    return (data, revenue_bess, charge_bess, opex_bess,
            revenue_psh, revenue_psh_DA, charge_psh)


# =============================================================================
# === IRR SOLVER ===
# =============================================================================

_RATE_LO = -1.0 + 1e-9   # just above the r = -1 pole
_RATE_HI = 10.0          # 1000 % — upper end of the economically relevant range
_REL_TOL = 1e-6          # relative tolerance for the root / imaginary-part tests


def _npv(rate: float, net: np.ndarray) -> float:
    """NPV of an undiscounted net stream at a given rate (period 0 undiscounted)."""
    disc = (1.0 + rate) ** np.arange(net.size, dtype=float)
    return float(np.sum(net / disc))


def _real_irrs(net: np.ndarray) -> np.ndarray:
    """All distinct real IRRs of one net stream in (-1, _RATE_HI], sorted ascending.

    Roots of sum_k net_k x^k = 0 with x = 1/(1+r) are found via np.roots; the
    real positive roots are mapped back to rates r = 1/x - 1. Spurious roots
    from the high-degree polynomial are rejected by re-substituting into the
    NPV (|NPV(r)| must be small relative to the gross cash-flow scale), then
    near-duplicate rates are collapsed. The surviving count is the genuine
    multiplicity of real IRRs.
    """
    roots = np.roots(net[::-1])
    real = roots[np.abs(roots.imag) <= _REL_TOL * (1.0 + np.abs(roots.real))].real
    real = real[real > 0.0]
    if real.size == 0:
        return np.empty(0, dtype=float)

    rates = 1.0 / real - 1.0
    rates = rates[(rates > _RATE_LO) & (rates <= _RATE_HI)]

    scale = float(np.abs(net).sum())
    rates = np.array([r for r in rates if abs(_npv(r, net)) <= _REL_TOL * scale])
    if rates.size == 0:
        return rates

    rates = np.sort(rates)
    if rates.size > 1:                                   # collapse near-duplicates
        rates = rates[np.concatenate(([True], np.diff(rates) > 1e-6))]
    return rates


def _irr_single(net: np.ndarray) -> tuple[float, bool, bool]:
    """IRR of one net stream. Returns (irr, no_root, multiple_irr).

    no_root      : no real IRR exists in (-1, _RATE_HI]  -> irr = NaN.
    multiple_irr : more than one real IRR exists (non-conventional cash flows)
                   -> irr holds the root nearest 10 % but the path is flagged
                   for exclusion, the IRR being ill-defined.
    """
    rates = _real_irrs(net)
    if rates.size == 0:
        return float("nan"), True, False
    principal = float(rates[np.argmin(np.abs(rates - 0.1))])
    return principal, False, rates.size > 1


def irr_per_path(
    I: np.ndarray, F: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-path IRR of the net stream F - I.

    Returns (irr (N,), no_root (N,) bool, multiple_irr (N,) bool).
    """
    net = F - I
    N = net.shape[0]
    irr = np.empty(N, dtype=float)
    no_root = np.zeros(N, dtype=bool)
    multiple = np.zeros(N, dtype=bool)
    for i in range(N):
        irr[i], no_root[i], multiple[i] = _irr_single(net[i])
    return irr, no_root, multiple


# =============================================================================
# === CASH-FLOW STREAMS ===
# =============================================================================

def _deterministic_capex_paths(capex_paths: np.ndarray) -> np.ndarray:
    """Replace each stochastic CAPEX path by the cross-path mean trajectory."""
    mean_path = capex_paths.mean(axis=0)                     # (T_max+1,)
    return np.broadcast_to(mean_path, capex_paths.shape).copy()


def bess_seq_detcapex(
    cfg: Config,
    revenue_bess: dict[float, np.ndarray],
    opex_bess: dict[float, np.ndarray],
    capex_paths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequential no-flex 10 MWh reinvestment over H = T_psh, deterministic CAPEX."""
    nf = replay_bess_noflex(
        cfg, revenue_bess, opex_bess, _deterministic_capex_paths(capex_paths),
    )
    return nf["I"], nf["F"]


def psh_cashflow_symmetric(
    cfg_base: Config,
    revenue_psh: np.ndarray,
    revenue_psh_DA: np.ndarray,
    charge_psh: np.ndarray,
    psh_bid_DA_sell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot PSH at t=0 under the full-elavgift, symmetric-grunnrente scenario."""
    cfg_s = _make_scenario_cfg(
        cfg_base,
        elavgift_psh=ELAVGIFT_SCENARIO_VALUES["full"],
        grunnrente_mode="symmetric",
    )
    opex_psh = build_opex_matrix(
        cfg_s, charge_psh, "PSH", None, production_matrix=psh_bid_DA_sell,
    )
    L = cfg_s.psh_lifetime(0)
    comp = compute_psh_cashflow_components(
        cfg_s, revenue_psh[:, 0:L], revenue_psh_DA[:, 0:L], opex_psh[:, 0:L], year_idx=0,
    )
    return comp["I"], comp["F"]


# =============================================================================
# === OUTPUT ===
# =============================================================================

def _summary(irr: np.ndarray, valid: np.ndarray) -> list[tuple[str, float]]:
    x = irr[valid & np.isfinite(irr)] * 100.0
    return [
        ("Mean", float(x.mean())),
        ("Median", float(np.median(x))),
        ("Std. dev.", float(x.std(ddof=1))),
        ("P5", float(np.percentile(x, 5))),
        ("P25", float(np.percentile(x, 25))),
        ("P75", float(np.percentile(x, 75))),
        ("P95", float(np.percentile(x, 95))),
        ("Min", float(x.min())),
        ("Max", float(x.max())),
    ]


def _plot_hist(
    irr: np.ndarray, valid: np.ndarray, color: str, plot_base: str, out_dir: Path,
) -> None:
    """Half-width thesis-style IRR histogram with mean / median markers."""
    import sys
    import matplotlib.pyplot as plt

    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from thesis_style import COLORS, apply_style, figsize
    apply_style("half")

    x = irr[valid & np.isfinite(irr)] * 100.0
    mean_v, median_v = float(x.mean()), float(np.median(x))

    fig, ax = plt.subplots(figsize=figsize("half"))
    ax.grid(axis="y")
    ax.set_axisbelow(True)

    ax.hist(x, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(mean_v, color=COLORS["highlight"], lw=2.0, label=f"Mean {mean_v:.1f}%")
    ax.axvline(median_v, color=COLORS["neutral"], lw=2.0, ls="--",
               label=f"Median {median_v:.1f}%")

    ax.set_xlabel("Internal rate of return (%)")
    ax.set_ylabel("Number of paths")
    ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{plot_base}_hist_concept1.{ext}", dpi=200)
    plt.close(fig)


def _update_tex(tex_path: Path, summary: list[tuple[str, float]]) -> None:
    """Refresh the numeric IRR column in an existing LaTeX summary table.

    Preserves the hand-written caption/label; only the 'Stat & value \\\\' rows
    are rewritten. No-op when the .tex file does not exist.
    """
    if not tex_path.exists():
        return
    values = {stat: f"{val:.2f}" for stat, val in summary}
    out_lines = []
    for line in tex_path.read_text().splitlines():
        stripped = line.strip()
        stat = stripped.split("&", 1)[0].strip() if "&" in stripped else ""
        if stat in values:
            out_lines.append(f"{stat} & {values[stat]} \\\\")
        else:
            out_lines.append(line)
    tex_path.write_text("\n".join(out_lines) + "\n")


def _write_variant(
    name: str, plot_base: str, color: str,
    irr: np.ndarray, no_root: np.ndarray, multiple: np.ndarray, out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = ~no_root & ~multiple

    per_path = out_dir / f"{name}_concept1.csv"
    with per_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path_id", "irr", "no_root", "multiple_irr"])
        for i, (r, nr, mi) in enumerate(zip(irr, no_root, multiple)):
            w.writerow([f"path_{i}", r, nr, mi])

    summary = _summary(irr, valid)
    summary_csv = out_dir / f"{name}_summary_concept1.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Statistic", "IRR (%)"])
        for stat, val in summary:
            w.writerow([stat, round(val, 2)])

    _update_tex(out_dir / f"{name}_summary_concept1.tex", summary)
    _plot_hist(irr, valid, color, plot_base, out_dir)

    stats = dict(summary)
    print(f"  {name:28s} mean={stats['Mean']:6.2f}%  median={stats['Median']:6.2f}%  "
          f"std={stats['Std. dev.']:5.2f}  "
          f"no_root={int(no_root.sum())}  multiple_irr={int(multiple.sum())}  "
          f"used={int(valid.sum())}/{valid.size}")


# =============================================================================
# === DRIVER ===
# =============================================================================

def run(out_dir: Path = _OUT_DIR, input_dir: Path = _INPUT_DIR) -> None:
    import sys
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from thesis_style import COLORS

    cfg_base = Config()
    cfg_base.psh_concept = 1
    cfg_base.elavgift_nok_per_mwh_bess = ELAVGIFT_SCENARIO_VALUES["full"]
    cfg_base.elavgift_nok_per_mwh_psh = ELAVGIFT_SCENARIO_VALUES["full"]

    (data, revenue_bess, charge_bess, opex_bess,
     revenue_psh, revenue_psh_DA, charge_psh) = _load(cfg_base, input_dir)

    capex_paths = simulate_bess_capex(cfg_base, _SEED)

    print("=" * 72)
    print("  Per-path IRR — concept 1")
    print("=" * 72)

    # --- BESS: sequential no-flex reinvestment, deterministic (mean) CAPEX ---
    I_bess, F_bess = bess_seq_detcapex(cfg_base, revenue_bess, opex_bess, capex_paths)
    _write_variant(
        "BESS_seq_detcapex_irr", "BESS_seq_detcapex_irr", COLORS["secondary"],
        *irr_per_path(I_bess, F_bess), out_dir,
    )

    # --- PSH: one-shot, symmetric grunnrente ---
    I_psh, F_psh = psh_cashflow_symmetric(
        cfg_base, revenue_psh, revenue_psh_DA, charge_psh, data.psh_bid_DA_sell,
    )
    _write_variant(
        "PSH_irr_GRsymmetric", "PSH_irr", COLORS["main"],
        *irr_per_path(I_psh, F_psh), out_dir,
    )

    print("=" * 72)
    print(f"  Written to {out_dir}")


if __name__ == "__main__":
    run()
