"""CLI entry point: python -m roa.plots [json_path] [output_dir] [--sweep sweep_json]"""

from __future__ import annotations

import argparse
from pathlib import Path

from .figures import make_all_plots


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate ROA thesis figures.")
    p.add_argument("json_path",
                   help="Path to main_pipeline_*.json from roa.main.")
    p.add_argument("output_dir", nargs="?", default=None,
                   help="Output directory for PDF figures (default: roa/plots/output/).")
    p.add_argument("--sweep", default=None,
                   help="Optional path to psh_capex_sweep_*.json for the CAPEX sensitivity figure.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "output"
    out = make_all_plots(args.json_path, output_dir, sweep_path=args.sweep)
    print("Generated figures:")
    for name, path in out["figures"].items():
        print(f"  {name:38s} {path}")
    if out.get("thresholds_capex_bnok"):
        print("\nPSH CAPEX thresholds (BNOK) per scenario:")
        print(f"  {'Scenario':24s}  {'Breakeven':>14s}  {'Crossover':>14s}")
        for name, vals in out["thresholds_capex_bnok"].items():
            be = vals.get("breakeven_capex_bnok")
            xo = vals.get("crossover_capex_bnok")
            be_s = f"{be:.2f}" if be is not None else "no crossing"
            xo_s = f"{xo:.2f}" if xo is not None else "no crossing"
            print(f"  {name:24s}  {be_s:>14s}  {xo_s:>14s}")


if __name__ == "__main__":
    main()
