"""
Convert results CSVs into LaTeX table snippets matching paper Tables 2–5.

For each λ configuration, produces an Elsevier-style 9-row table:
    Model | AC | SN | SP | PR | F1 | AUC

The best row per table is bolded. The output is printed to stdout AND saved
to paper_pipeline/output/figures/table_<lambda>.tex so you can `\\input{}` it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "results"
TABLES_DIR = PROJECT_ROOT / "paper_pipeline" / "output" / "tables"

# Map λ name → human-readable column header for the table caption.
LAMBDA_LABEL = {
    "low_pass":      r"$\lambda_k = 1/(k+1)$",
    "high_pass":     r"$\lambda_k = 1/(N-k)$",
    "dft":           r"$\lambda_k = e^{i 2\pi k / N}$",
    "odd_harmonic":  r"$\lambda_k = e^{i (2k+1) \pi / N}$",
}

# Map paper-friendly classifier order.
DISPLAY_ORDER = [
    "SVM (RBF)",
    "LightGBM",
    "CatBoost",
    "Gradient Boosting",
    "Extra Trees",
    "KNN",
    "Logistic Regression",
    "MLP",
    "Deep DNN",
]


def fmt_pct(v: float) -> str:
    if pd.isna(v):
        return "--"
    return f"{100 * v:.1f}"


def bold_best(values: list[float]) -> list[bool]:
    if not values:
        return []
    m = max(values)
    return [abs(v - m) < 1e-9 for v in values]


def format_table_for_lambda(df: pd.DataFrame, lambda_name: str) -> str:
    sub = df[df["lambda"] == lambda_name].copy()
    if sub.empty:
        return f"% no rows for lambda={lambda_name}\n"

    # Sort by display order; warn about missing rows.
    sub["__order"] = sub["classifier"].apply(
        lambda c: DISPLAY_ORDER.index(c) if c in DISPLAY_ORDER else 999
    )
    sub = sub.sort_values("__order").drop(columns="__order")

    metrics = ["test_accuracy", "test_sensitivity", "test_specificity",
               "test_precision", "test_f1", "test_auc"]
    best_acc_idx = sub["test_accuracy"].idxmax()

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"    \centering")
    label_math = LAMBDA_LABEL.get(lambda_name, lambda_name)
    lines.append(
        r"    \caption{Performance of machine learning models using Krawtchouk "
        f"features ({label_math}, $p=0.5$, $N=64$) with geometric, color, and texture features.}}"
    )
    lines.append(rf"    \label{{tab:krawtchouk_{lambda_name}_p05}}")
    lines.append(r"    \resizebox{\columnwidth}{!}{%")
    lines.append(r"    \begin{tabular}{l|c|c|c|c|c|c}")
    lines.append(r"        \hline")
    lines.append(r"        \textbf{Model} & \textbf{AC (\%)} & \textbf{SN (\%)} & \textbf{SP (\%)} & "
                 r"\textbf{PR (\%)} & \textbf{F1 (\%)} & \textbf{AUC (\%)} \\")
    lines.append(r"        \hline")

    for idx, row in sub.iterrows():
        cells = [fmt_pct(row[m]) for m in metrics]
        name = row["classifier"]
        if idx == best_acc_idx:
            cells = [rf"\textbf{{{c}}}" for c in cells]
            name = rf"\textbf{{{name}}}"
        lines.append(f"        {name} & " + " & ".join(cells) + r" \\")

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}%")
    lines.append(r"    }")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv", type=Path, default=None,
        help="Path to summary CSV (default: most recent in results/)."
    )
    args = parser.parse_args()

    if args.summary_csv is None:
        candidates = sorted(RESULTS_DIR.glob("summary_*.csv"))
        if not candidates:
            print("ERROR: no summary_*.csv in", RESULTS_DIR, file=sys.stderr)
            return 2
        args.summary_csv = candidates[-1]

    print(f"% Source: {args.summary_csv}")
    df = pd.read_csv(args.summary_csv)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for lam in ["low_pass", "high_pass", "dft", "odd_harmonic"]:
        snippet = format_table_for_lambda(df, lam)
        out_path = TABLES_DIR / f"table_{lam}.tex"
        out_path.write_text(snippet)
        print(f"\n% ===== Table for λ={lam}  →  {out_path} =====")
        print(snippet)

    return 0


if __name__ == "__main__":
    sys.exit(main())
