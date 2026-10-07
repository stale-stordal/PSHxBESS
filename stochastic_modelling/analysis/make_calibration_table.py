"""
Build a thesis-ready table of calibrated MRS parameters for Spot and FCR.

The raw params.csv has parameters in deseasonalised, pre-conversion units
(spot in øre/kWh, FCR in raw units). We display them in NOK/MWh and NOK/MW
where applicable so the table aligns with the rest of the thesis.

Outputs:
    data/calibration_table.csv   (machine readable)
    data/calibration_table.tex   (LaTeX booktabs, ready for thesis)
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stochastic_modelling.analysis._constants import ROOT, DATA_OUT, SPOT_SCALE, FCR_SCALE, UNIT_SPOT, UNIT_FCR

PARAMS_FILE = os.path.join(ROOT, "results", "params.csv")


def main():
    df = pd.read_csv(PARAMS_FILE)
    rows = []
    for _, row in df.iterrows():
        series = row["series"]
        scale  = SPOT_SCALE if series == "Spot" else FCR_SCALE
        unit   = UNIT_SPOT  if series == "Spot" else UNIT_FCR

        # Mean parameters (rates of mean reversion in regime 1)
        # Regime-1 long-run mean = alpha1 / beta1 (deseasonalised)
        lr_mean_des = row["alpha1"] / row["beta1"]
        rec = {
            "Series":            series,
            "Unit":              unit,
            "α₁ (mean reversion)":         f"{row['alpha1']:.3f}",
            "β₁":                          f"{row['beta1']:.4f}",
            "σ₁ (base vol)":               f"{row['sigma1'] * scale:.3f}",
            "γ₁":                          f"{row['gamma1']:.3f}",
            "Long-run regime-1 mean":      f"{lr_mean_des * scale:.2f}",
            "α₂ (jump-up speed)":          f"{row['alpha2']:.3f}",
            "σ₂":                          f"{row['sigma2'] * scale:.3f}",
            "q₂ (jump-up centre)":         f"{row['q2'] * scale:.2f}",
            "α₃ (jump-down speed)":        f"{row['alpha3']:.3f}",
            "σ₃":                          f"{row['sigma3'] * scale:.3f}",
            "q₃ (jump-down centre)":       f"{row['q3'] * scale:.2f}",
            "P₀₀":                         f"{row['P_00']:.4f}",
            "P₁₁":                         f"{row['P_11']:.4f}",
            "P₂₂":                         f"{row['P_22']:.4f}",
        }
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("Series").T
    csv_path = os.path.join(DATA_OUT, "calibration_table.csv")
    out.to_csv(csv_path)
    print(f"Saved {csv_path}")
    print(out)

    # LaTeX (booktabs, hand-built to avoid jinja2 dependency)
    tex_path = os.path.join(DATA_OUT, "calibration_table.tex")
    cols = list(out.columns)
    lines = [
        "\\begin{tabular}{l" + "r" * len(cols) + "}",
        "\\toprule",
        " & " + " & ".join(cols) + " \\\\",
        "\\midrule",
    ]
    for label, row in out.iterrows():
        lines.append(f"{label} & " + " & ".join(str(v) for v in row.values) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
