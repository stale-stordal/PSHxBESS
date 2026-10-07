"""
roa/config.py — Central parameter store for the Real Options Analysis pipeline.

All monetary values in NOK unless explicitly noted.

All numerical constants live here. Import Config and pass it through the pipeline
instead of scattering magic numbers across modules.

CRITICAL DESIGN RULE: No derived value is ever hardcoded. Every computed quantity
is a @property so that changing any source parameter propagates automatically.

Base currency: NOK.
  - nok_per_eur is kept as a reference parameter but is used only once:
    to convert opex_psh_grid_fixed_eur_per_mwh → NOK inside the
    psh_production_opex_per_mwh property.
  - All CAPEX, OPEX, revenue, NCF, and option values are in NOK.

Data sources:
  - BESS CAPEX:  Å Energi industry quote (28.5 MNOK, 4.6 MW / 10 MWh turnkey)
  - GBM params:  gbm_mu and gbm_sigma fields below (BESS unit-cost dynamics)
  - PSH CAPEX:   NVE/industry spreadsheet (NOK/MW)
  - BESS OPEX:   Å Energi estimate (300 kNOK/year for baseline system)
  - PSH OPEX:    NVE spreadsheet (NOK units; one item originally in EUR, converted here)

Indexing convention
-------------------
Revenue data exists for years 0 .. T_max - 1.  The terminal continuation
value V^B_{i, T_max} = 0 is placed at index T_max (one past the last data
row).  Admissibility tests therefore use T_max - 1 as the last usable
revenue year:
  - PSH exercised at s must end at s + T_psh <= T_max - 1 (when no truncation).
  - BESS reinvestment at t requires t < T_max - 1 (at least one revenue year).
Revenue-data coverage (whether the loaded CSV actually contains rows for
years 0 .. T_max - 1) is validated at CSV load time, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Scenario design constants — shared across pipeline, sensitivity, and plots
# ---------------------------------------------------------------------------
#
# PSH elavgift sensitivity values used in Panel A. The "full" rate equals the
# Config default elavgift_nok_per_mwh_bess and elavgift_nok_per_mwh_psh.
ELAVGIFT_SCENARIO_VALUES: dict = {
    "full":    71.3,   # NOK/MWh, Skatteetaten 2026 alminnelig sats (= Config default)
    "reduced": 6.0,    # NOK/MWh, redusert sats (sensitivity)
    "zero":    0.0,    # NOK/MWh, exemption (sensitivity)
}

# Canonical five-scenario design used by the main pipeline, the CAPEX sweep,
# and the plotter. Each entry is (scenario_name, elavgift_key, grunnrente_mode).
# Order is the canonical reporting order.
SCENARIO_DEFINITIONS: tuple = (
    ("full_GRnone",      "full",    "none"),
    ("reduced_GRnone",   "reduced", "none"),
    ("zero_GRnone",      "zero",    "none"),
    ("full_GRsymmetric", "full",    "symmetric"),
    ("full_GRtruncated", "full",    "truncated"),
)


# ---------------------------------------------------------------------------
# PSH Concept helper — stores raw NOK/MW values; total CAPEX is in NOK
# ---------------------------------------------------------------------------

@dataclass
class PshConcept:
    """Raw capital cost components for one PSH development concept.

    All per-MW costs are in NOK/MW. Call total_nok(nok_per_eur) to obtain the
    full NOK CAPEX for the rated MW capacity of this concept.
    """

    capex_bygg_nok_per_mw: float    # Civil / construction cost (NOK/MW)
    capex_elek_nok_per_mw: float    # Electrical / mechanical cost (NOK/MW)
    capex_andre_nok_per_mw: float   # Other CAPEX components (NOK/MW)
    psh_mw: float                   # Rated power capacity (MW)

    def total_nok_per_mw(self) -> float:
        """Sum of all CAPEX components in NOK per MW."""
        return self.capex_bygg_nok_per_mw + self.capex_elek_nok_per_mw + self.capex_andre_nok_per_mw

    def total_nok(self) -> float:
        """Total project CAPEX in NOK for the full rated capacity."""
        return self.total_nok_per_mw() * self.psh_mw


# ---------------------------------------------------------------------------
# Main configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:

    # ------------------------------------------------------------------
    # EXCHANGE RATE
    # Reference only — used solely to convert the one EUR-denominated
    # PSH OPEX item (grid_fixed) to NOK in psh_production_opex_per_mwh.
    # ------------------------------------------------------------------
    nok_per_eur: float = 11.50          # NOK per EUR

    # ------------------------------------------------------------------
    # TIME PARAMETERS
    # ------------------------------------------------------------------
    dt: float = 1                           # decision step size (years)
    T_max: Optional[int] = None             # total evaluation horizon; populated from data by load_optimizer_data
    T_def: int = 10                         # deferral window (years); auto-reduced by adjust_T_def_for_horizon if needed
    T_psh: int = 67                         # PSH operational lifetime (years)
    T_bess: int = 10                        # single BESS cycle length (years)
    allow_bess_truncation: bool = True      # shorten final BESS cycle to fit within T_max
    allow_psh_truncation: bool = False      # if False, require s + T_psh <= T_max - 1 at exercise

    # ------------------------------------------------------------------
    # SIMULATION PARAMETERS
    # ------------------------------------------------------------------
    N: Optional[int] = None             # Monte Carlo paths; populated from data by load_optimizer_data

    # ------------------------------------------------------------------
    # FINANCIAL PARAMETERS
    # ------------------------------------------------------------------
    r: float = 0.04                     # risk-adjusted discount rate
    tax_rate: float = 0.22              # corporate tax rate
    dep_rate: float = 0.20             # BESS declining-balance depreciation rate
                                       # (Skatteloven §14-43 group d, machinery/fixtures)
    # PSH depreciation: component-wise linear schedule per Skatteloven §18-6.
    # Civil works (bygg) at 1.5%/yr × 67 yr; electromechanical (elek) and other
    # (andre) at 2.5%/yr × 40 yr. At T_psh = 67 all three components are fully
    # depreciated (civil: 1.5% × 67 = 100.5% → capped at 100%; elek/andre: fully
    # written off at year 40). Terminal civil book value is zero; no residual
    # terminal cash flow from psh_terminal_value_multiplier.
    psh_dep_rate_civil: float = 0.015   # bygg/anlegg, 1.5%/yr
    psh_dep_rate_elek:  float = 0.025   # elektromekanisk, 2.5%/yr
    psh_dep_rate_andre: float = 0.025   # andre, 2.5%/yr
    psh_dep_life_civil: int = 67
    psh_dep_life_elek:  int = 40
    psh_dep_life_andre: int = 40
    # PSH terminal-value multiplier on the year-T_psh civil-works book value.
    # 1.0 = sale at tax book (zero realised gain, post-tax = pre-tax).
    # > 1.0 = sale above book; the gain (mult-1)*book is taxed at tax_rate.
    # 1.5 / 2.0 are the planned sensitivity points.
    psh_terminal_value_multiplier: float = 1.0
    # Hydropower grunnrente tax for PSH only. Formal rate is 57.7%; this model
    # uses a simplified 45% effective rate. Post-2021 cash-flow treatment gives
    # immediate investment deduction. FCR capacity payments are excluded from the
    # grunnrente base under Skatteloven §18-3 second paragraph (production valued
    # at the spot price); see Method.tex sec:method:roa:cashflow.
    grunnrente_mode: str = "none"       # one of: none, symmetric, truncated
    grunnrente_effective_rate: float = 0.45
    psh_opex_allocation_for_grunnrente: str = "revenue_share"

    # ------------------------------------------------------------------
    # BESS CAPEX SOURCE DATA  (NOK)
    # Source: Å Energi industry quote — turnkey 4.6 MW / 10 MWh system
    # gbm_K0 property = capex_bess_total_nok / capex_bess_baseline_mwh (NOK/MWh)
    # ------------------------------------------------------------------
    capex_bess_total_nok: float = 28_500_000    # NOK, total turnkey quote
    capex_bess_baseline_mwh: float = 10.0       # MWh, system the quote covers

    gbm_mu: float = -0.022             # GBM drift for BESS CAPEX (annual)
    gbm_sigma: float = 0.25            # GBM volatility for BESS CAPEX (annual)

    # ------------------------------------------------------------------
    # PSH CAPEX — three development concepts  (NOK/MW)
    # Source: NVE/industry spreadsheet.
    # capex_psh and q_psh_mw are @properties that read the active concept.
    # ------------------------------------------------------------------
    psh_concepts: list = field(default_factory=lambda: [
        PshConcept(                             # Concept 1 — small, 100 MW
            capex_bygg_nok_per_mw=10_800_000,
            capex_elek_nok_per_mw=5_400_000,
            capex_andre_nok_per_mw=2_100_000,
            psh_mw=100,
        ),
        PshConcept(                             # Concept 2 — large, 1000 MW (high elec. cost)
            capex_bygg_nok_per_mw=5_300_000,
            capex_elek_nok_per_mw=7_500_000,
            capex_andre_nok_per_mw=3_000_000,
            psh_mw=1_000,
        ),
        PshConcept(                             # Concept 3 — large, 1000 MW (low-cost site)
            capex_bygg_nok_per_mw=3_800_000,
            capex_elek_nok_per_mw=4_400_000,
            capex_andre_nok_per_mw=2_000_000,
            psh_mw=1_000,
        ),
    ])

    psh_concept: int = 1                # Active concept selector: 1, 2, or 3

    # ------------------------------------------------------------------
    # CAPACITY GRID
    # ------------------------------------------------------------------
    q_bess_grid: list = field(
        default_factory=lambda: [5, 10, 15, 20]
    )

    # ------------------------------------------------------------------
    # BESS OPEX SOURCE DATA  (NOK)
    # Source: Å Energi estimate for baseline 10 MWh system
    # opex_bess_per_mwh_year property = opex_bess_annual_nok / capex_bess_baseline_mwh
    # ------------------------------------------------------------------
    opex_bess_annual_nok: float = 300_000   # NOK/year for baseline system

    # ------------------------------------------------------------------
    # PSH OPEX RAW PARAMETERS
    # All values are in NOK except opex_psh_grid_fixed_eur_per_mwh which is in
    # EUR/MWh (as it appears in the source spreadsheet). The runtime PSH OPEX
    # is path- and year-dependent: psh_capacity_opex_nok is the constant
    # capacity-scaled portion (O&M + rehab annuity); psh_production_opex_per_mwh
    # is the per-MWh rate applied pathwise to gross turbined volume bid_DA_sell.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # POWER TAX (ELAVGIFT)
    # Norwegian elavgift, Skatteetaten 2026:
    #   alminnelig sats 7.13 øre/kWh = 71.3 NOK/MWh,
    #   redusert sats 0.60 øre/kWh = 6.0 NOK/MWh.
    # Reduced rate is treated as a sensitivity, not assumed.
    # The two NOK fields below are the operational rates (technology-specific).
    # ------------------------------------------------------------------
    elavgift_nok_per_mwh_bess: float = 71.3           # BESS elavgift (NOK/MWh) — Skatteetaten 2026 alminnelig sats
    elavgift_nok_per_mwh_psh:  float = 71.3           # PSH elavgift (NOK/MWh) — Skatteetaten 2026 alminnelig sats; override per run for sensitivities

    opex_psh_om_nok_per_kw: float = 50                  # NOK/kW/year (O&M)
    opex_psh_property_tax_nok_per_kwh: float = 2.74     # NOK/kWh — taxable value (grunnlag), not the levy itself.
                                                        # Effective levy = opex_psh_property_tax_rate × this × 1000 NOK/MWh.
    opex_psh_grid_fixed_eur_per_mwh: float = 1.2        # EUR/MWh (source spreadsheet — converted to NOK at runtime)
    opex_psh_grid_system_nok_per_mwh: float = 10        # NOK/MWh (systemdrift, expected)
    opex_psh_concession_nok_per_mwh: float = 4          # NOK/MWh (konsesjonsavgifter, expected)
    opex_psh_rehab_nok_per_kw: float = 52.5             # NOK/kW/year (rehab annuity).
                                                        # Reduced from 60 by 12.5% (industry-partner guidance:
                                                        # 10–15% range, midpoint 12.5%) for the 40-year horizon
                                                        # adjustment.
    opex_psh_property_tax_rate: float = 0.007           # 0.7% (eiendomsskatt sats)

    # ------------------------------------------------------------------
    # LSMC PARAMETERS
    # ------------------------------------------------------------------
    laguerre_degree: int = 2            # max degree of weighted Laguerre polynomials
    # Diagnostic flag — only the convergence study should ever enable this.
    # When True and cfg.N > N_data, the loader bootstraps path indices with
    # replacement instead of erroring. Duplicated paths bias R² upward and
    # underestimate seed-to-seed SE; not for production runs.
    bootstrap_upsample: bool = False

    # ------------------------------------------------------------------
    # CANONICAL EFFICIENCY VALUES (documentation only)
    # These are the per-direction efficiencies used by the upstream dispatch
    # optimizer that produced the BESS/PSH CSVs. Production runs do not read
    # these fields; they are retained as a code-side cross-reference to the
    # Method.tex canonical specification.
    # ------------------------------------------------------------------
    bess_efficiency: float = 0.95   # one-way (per-direction) efficiency (Method.tex)
    psh_efficiency:  float = 0.85   # one-way (per-direction) efficiency (Method.tex; SINTEF)

    # ==================================================================
    # VALIDATION
    # ==================================================================

    def __post_init__(self) -> None:
        # T_max and N are Optional and populated by load_optimizer_data.
        # Skip checks that depend on them when None; validate_against_data
        # is called after the loader populates them.
        for name, val in [
            ('T_def',  self.T_def),
            ('T_psh',  self.T_psh),
            ('T_bess', self.T_bess),
        ]:
            if not isinstance(val, int) or val <= 0:
                raise ValueError(
                    f"{name} must be a positive integer, got {val!r}"
                )
        if self.T_max is not None:
            if not isinstance(self.T_max, int) or self.T_max <= 0:
                raise ValueError(f"T_max must be a positive integer, got {self.T_max!r}")
            if self.T_def >= self.T_max:
                raise ValueError(
                    f"T_def ({self.T_def}) must be less than T_max ({self.T_max})"
                )
            if not self.allow_psh_truncation:
                if self.T_def + self.T_psh > self.T_max - 1:
                    raise ValueError(
                        f"With allow_psh_truncation=False, T_def + T_psh must be "
                        f"<= T_max - 1, but {self.T_def} + {self.T_psh} = "
                        f"{self.T_def + self.T_psh} > {self.T_max - 1}"
                    )
        if self.N is not None:
            if not isinstance(self.N, int) or self.N <= 0:
                raise ValueError(f"N must be a positive integer, got {self.N!r}")
        for _fname, _fval in [
            ("elavgift_nok_per_mwh_bess", self.elavgift_nok_per_mwh_bess),
            ("elavgift_nok_per_mwh_psh",  self.elavgift_nok_per_mwh_psh),
        ]:
            if _fval < 0:
                raise ValueError(f"{_fname} must be >= 0, got {_fval!r}")
        if self.grunnrente_mode not in {"none", "symmetric", "truncated"}:
            raise ValueError(
                "grunnrente_mode must be one of {'none', 'symmetric', 'truncated'}, "
                f"got {self.grunnrente_mode!r}"
            )
        if not (0.0 <= self.grunnrente_effective_rate <= 1.0):
            raise ValueError(
                "grunnrente_effective_rate must be between 0 and 1, "
                f"got {self.grunnrente_effective_rate!r}"
            )
        if self.psh_opex_allocation_for_grunnrente != "revenue_share":
            raise ValueError(
                "psh_opex_allocation_for_grunnrente currently only supports "
                f"'revenue_share', got {self.psh_opex_allocation_for_grunnrente!r}"
            )

    # ==================================================================
    # POST-LOAD VALIDATION AND ADJUSTMENT
    # ==================================================================

    def validate_against_data(self) -> None:
        """Run T_max-dependent checks after the loader populates N and T_max.

        Mirrors the gated checks in __post_init__. Called by load_optimizer_data
        once the data shape has been determined.
        """
        if self.T_max is None:
            raise ValueError("T_max is None; load_optimizer_data must populate it first.")
        if self.N is None:
            raise ValueError("N is None; load_optimizer_data must populate it first.")
        if not isinstance(self.T_max, int) or self.T_max <= 0:
            raise ValueError(f"T_max must be a positive integer, got {self.T_max!r}")
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError(f"N must be a positive integer, got {self.N!r}")
        if self.T_def >= self.T_max:
            raise ValueError(
                f"T_def ({self.T_def}) must be less than T_max ({self.T_max})"
            )
        if not self.allow_psh_truncation:
            if self.T_def + self.T_psh > self.T_max - 1:
                raise ValueError(
                    f"With allow_psh_truncation=False, T_def + T_psh must be "
                    f"<= T_max - 1, but {self.T_def} + {self.T_psh} = "
                    f"{self.T_def + self.T_psh} > {self.T_max - 1}"
                )

    def adjust_T_def_for_horizon(self) -> None:
        """Auto-reduce T_def to the maximum admissible value for the loaded horizon.

        Constraint: T_def + T_psh <= T_max - 1 when allow_psh_truncation is False.
        """
        if self.T_max is None or self.allow_psh_truncation:
            return
        max_T_def = self.T_max - 1 - self.T_psh
        if self.T_def > max_T_def:
            print(
                f"[config] T_def reduced from {self.T_def} to {max_T_def} to satisfy "
                f"T_def + T_psh <= T_max - 1 (T_max={self.T_max}, T_psh={self.T_psh})."
            )
            self.T_def = max_T_def

    # ==================================================================
    # HORIZON HELPER METHODS
    # ==================================================================

    def psh_admissible(self, s: int) -> bool:
        """PSH can be exercised at deferral time s if it fits within the horizon.

        When allow_psh_truncation is False: s + T_psh <= T_max - 1.
        When allow_psh_truncation is True:  s <= T_max - 2
        (needs at least one revenue year: s + 1 <= T_max - 1).
        """
        if self.allow_psh_truncation:
            return s <= self.T_max - 2
        return s + self.T_psh <= self.T_max - 1

    def psh_lifetime(self, s: int) -> int:
        """Operating lifetime (years) if PSH is exercised at deferral time s.

        When allow_psh_truncation is False: always T_psh (s is unused).
        When allow_psh_truncation is True:  min(T_psh, T_max - 1 - s).
        """
        if self.allow_psh_truncation:
            return min(self.T_psh, self.T_max - 1 - s)
        return self.T_psh

    def bess_admissible(self, t: int) -> bool:
        """BESS reinvestment at time t requires at least one revenue year."""
        return t < self.T_max - 1

    def bess_cycle_length(self, t: int) -> int:
        """Length of a BESS cycle starting at t.

        When allow_bess_truncation is True:  min(T_bess, T_max - 1 - t).
        When allow_bess_truncation is False: T_bess
        (admissibility upstream must ensure t + T_bess <= T_max - 1).
        """
        if self.allow_bess_truncation:
            return min(self.T_bess, self.T_max - 1 - t)
        return self.T_bess

    def bess_cycle_end(self, t: int) -> int:
        """Last year of the BESS cycle starting at t."""
        return t + self.bess_cycle_length(t)

    def bess_has_continuation(self, t: int) -> bool:
        """Continuation value V^B_{i, t+T_bess} is added only for full,
        non-terminal cycles: cycle must be exactly T_bess long and its end
        must not be the final index.
        """
        return (
            self.bess_cycle_length(t) == self.T_bess
            and self.bess_cycle_end(t) + 1 <= self.T_max
        )

    # ==================================================================
    # COMPUTED PROPERTIES — never hardcode these, always derive
    # ==================================================================

    @property
    def delta(self) -> float:
        """Discount factor per step — derived from r and dt."""
        return (1 + self.r) ** (-self.dt)

    @property
    def gbm_K0(self) -> float:
        """Initial BESS CAPEX (NOK/MWh) — derived from Å Energi turnkey quote.

        = capex_bess_total_nok / capex_bess_baseline_mwh
        = 28,500,000 NOK / 10 MWh = 2,850,000 NOK/MWh
        """
        return self.capex_bess_total_nok / self.capex_bess_baseline_mwh

    @property
    def capex_psh(self) -> float:
        """Total PSH CAPEX in NOK for the active concept (psh_concept = 1/2/3)."""
        return self.psh_concepts[self.psh_concept - 1].total_nok()

    @property
    def q_psh_mw(self) -> float:
        """Rated PSH power capacity (MW) for the active concept."""
        return self.psh_concepts[self.psh_concept - 1].psh_mw

    @property
    def opex_bess_per_mwh_year(self) -> float:
        """Annual BESS OPEX per MWh of installed capacity (NOK/MWh/year).

        = opex_bess_annual_nok / capex_bess_baseline_mwh
        = 300,000 NOK/year / 10 MWh = 30,000 NOK/MWh/year
        """
        return self.opex_bess_annual_nok / self.capex_bess_baseline_mwh

    @property
    def psh_capacity_opex_nok(self) -> float:
        """Annual PSH capacity-scaled OPEX (NOK/year) = O&M + rehab annuity.

        Constant across paths and years. Production-scaled components
        (property tax, fixed grid, system grid, concession) are path- and
        year-dependent and applied separately via psh_production_opex_per_mwh
        × bid_DA_sell at the OPEX-matrix construction site.
        """
        om    = self.opex_psh_om_nok_per_kw    * self.q_psh_mw * 1_000
        rehab = self.opex_psh_rehab_nok_per_kw * self.q_psh_mw * 1_000
        return om + rehab

    @property
    def psh_production_opex_per_mwh(self) -> float:
        """Per-MWh PSH production-scaled OPEX rate (NOK/MWh of gross turbined energy).

        Sum of:
          - property tax: rate × taxable value × 1000
            = 0.007 × 2.74 × 1000 = 19.18 NOK/MWh
          - fixed grid:   1.20 EUR/MWh × 11.50 NOK/EUR = 13.80 NOK/MWh
          - system grid:  10 NOK/MWh
          - concession:    4 NOK/MWh

        Multiplied pathwise by bid_DA_sell (gross turbined volume from the
        dispatch optimizer output, MWh/year) to obtain the per-(i, t) OPEX
        contribution.
        """
        property_tax = (
            self.opex_psh_property_tax_rate
            * self.opex_psh_property_tax_nok_per_kwh
            * 1_000
        )
        grid_fixed = self.opex_psh_grid_fixed_eur_per_mwh * self.nok_per_eur
        return (
            property_tax
            + grid_fixed
            + self.opex_psh_grid_system_nok_per_mwh
            + self.opex_psh_concession_nok_per_mwh
        )


# Default singleton intentionally removed: Config() now defers N and T_max to
# load_optimizer_data, so a module-level default would carry None placeholders
# that would silently bypass T_max-dependent validation. Construct Config() at
# call sites instead.
