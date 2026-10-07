"""ROA sensitivity sweeps."""

# Relative tolerance for the BESS-only V0 floor in CAPEX-breakeven detection.
# Used by psh_capex_sweep._summarise_thresholds.
BREAKEVEN_REL_EPS = 1e-3

from .psh_capex_sweep import run_capex_sweep, override_psh_capex

__all__ = ["run_capex_sweep", "override_psh_capex", "BREAKEVEN_REL_EPS"]
