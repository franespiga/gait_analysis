#!/usr/bin/env python3
"""
Evaluate predicted step labels against expert annotations.

Usage
-----
::

    python scripts/evaluate.py \\
        --predictions results/batch/batch_summary.csv \\
        --annotations data/expert_labels.csv \\
        --output-dir results/evaluation

Expected CSV formats
--------------------

**Predictions** (from ``scripts/run_batch.py`` per-trial ``steps.csv``)::

    foot,ic_time_ms,contact_type,...
    left,1200.0,heel_strike,...
    right,1600.0,toe_walk,...

**Annotations** (expert gold standard, same columns at minimum)::

    foot,ic_time_ms,contact_type,...

Matching is done by closest ``ic_time_ms`` within a tolerance window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gait_analysis.eval.classification import evaluate_step_labels
from gait_analysis.eval.agreement import compute_mae, bland_altman


def match_events(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    time_col: str = "ic_time_ms",
    tolerance_ms: float = 150.0,
) -> pd.DataFrame:
    """Match predicted and gold events by closest time within tolerance."""
    matched = []
    gold_used: set[int] = set()
    for _, pred_row in pred_df.iterrows():
        t_pred = pred_row[time_col]
        best_idx = None
        best_dist = tolerance_ms + 1
        for gi, grow in gold_df.iterrows():
            if gi in gold_used:
                continue
            d = abs(grow[time_col] - t_pred)
            if d < best_dist:
                best_dist = d
                best_idx = gi
        if best_idx is not None and best_dist <= tolerance_ms:
            gold_used.add(best_idx)
            matched.append({
                "pred_time_ms": t_pred,
                "gold_time_ms": gold_df.loc[best_idx, time_col],
                "pred_contact": pred_row.get("contact_type", "unknown"),
                "gold_contact": gold_df.loc[best_idx, "contact_type"],
                "pred_foot": pred_row.get("foot", ""),
                "gold_foot": gold_df.loc[best_idx, "foot"] if "foot" in gold_df else "",
                "time_error_ms": best_dist,
            })
    return pd.DataFrame(matched)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate gait step labels against expert annotations."
    )
    ap.add_argument("--predictions", type=Path, required=True,
                    help="Predicted steps CSV (from pipeline)")
    ap.add_argument("--annotations", type=Path, required=True,
                    help="Expert annotations CSV (gold standard)")
    ap.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    ap.add_argument("--tolerance-ms", type=float, default=150.0,
                    help="Time tolerance for event matching (ms)")
    args = ap.parse_args()

    if not args.predictions.exists():
        print(f"Predictions not found: {args.predictions}", file=sys.stderr)
        sys.exit(1)
    if not args.annotations.exists():
        print(f"Annotations not found: {args.annotations}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = pd.read_csv(args.predictions)
    gold_df = pd.read_csv(args.annotations)

    matched = match_events(pred_df, gold_df, tolerance_ms=args.tolerance_ms)
    matched.to_csv(args.output_dir / "matched_events.csv", index=False)
    print(f"Matched {len(matched)} / {len(pred_df)} predicted events "
          f"(of {len(gold_df)} gold events)")

    if matched.empty:
        print("No matches — skipping metrics.")
        return

    # Classification metrics
    report = evaluate_step_labels(
        y_true=matched["gold_contact"].tolist(),
        y_pred=matched["pred_contact"].tolist(),
    )
    report_dict = report.to_dict()
    with open(args.output_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\nAccuracy: {report.accuracy:.3f}")
    print(f"Macro F1: {report.macro_f1:.3f}")
    for cls, metrics in report.per_class.items():
        print(f"  {cls}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}")

    # Temporal MAE
    mae = compute_mae(
        matched["pred_time_ms"].values,
        matched["gold_time_ms"].values,
    )
    print(f"\nIC timing MAE: {mae:.1f} ms")

    # Bland-Altman on timing
    ba = bland_altman(matched["pred_time_ms"].values, matched["gold_time_ms"].values)
    ba_dict = ba.to_dict()
    with open(args.output_dir / "bland_altman_timing.json", "w") as f:
        json.dump(ba_dict, f, indent=2)
    print(f"Bland-Altman bias: {ba.mean_diff:.1f} ms, LoA: [{ba.lower_loa:.1f}, {ba.upper_loa:.1f}]")

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
