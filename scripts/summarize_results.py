from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def best_row(results_csv: Path) -> dict[str, object]:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    sort_col = "metrics/mAP50-95(B)" if "metrics/mAP50-95(B)" in df.columns else "metrics/mAP50(B)"
    row = df.loc[df[sort_col].idxmax()]
    output: dict[str, object] = {
        "run": results_csv.parent.name,
        "epoch": int(row.get("epoch", -1)),
    }
    for col in METRIC_COLUMNS:
        if col in df.columns:
            output[col] = float(row[col])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect best YOLO metrics into CSV/XLSX tables.")
    parser.add_argument("--runs-dir", default="runs/paper", help="Directory containing YOLO run folders.")
    parser.add_argument("--out", default="paper_results.xlsx", help="Output .xlsx or .csv path.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = [best_row(p) for p in sorted(runs_dir.glob("*/results.csv"))]
    if not rows:
        raise FileNotFoundError(f"No results.csv found under {runs_dir}")

    summary = pd.DataFrame(rows).sort_values("run")
    out = Path(args.out)
    if out.suffix.lower() == ".csv":
        summary.to_csv(out, index=False)
    else:
        summary.to_excel(out, index=False)
    print(f"Wrote {out} with {len(summary)} runs")


if __name__ == "__main__":
    main()
