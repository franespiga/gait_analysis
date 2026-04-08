#!/usr/bin/env python3
"""
Build quality-aware summaries from a pipeline steps CSV.

Outputs:
- summary_all.json
- summary_primary_only.json (excluded_from_primary_metrics == False)

Usage:
    python scripts/summarize_quality.py \
        --steps-csv results/test_run_tuned/VID_xxx_steps.csv \
        --summary-json results/test_run_tuned/VID_xxx_summary.json \
        --output-dir results/test_run_tuned
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gait_analysis.gait_metrics import StepRecord, compute_summary


def _none_if_nan(v: Any) -> Any:
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"true", "1", "yes", "y"}
    return False


def _records_from_steps_df(df: pd.DataFrame) -> list[StepRecord]:
    records: list[StepRecord] = []
    for _, row in df.iterrows():
        records.append(
            StepRecord(
                foot=str(row.get("foot", "")),
                ic_time_ms=float(row.get("ic_time_ms", 0.0)),
                to_time_ms=float(row.get("to_time_ms", 0.0)),
                next_ic_time_ms=_none_if_nan(row.get("next_ic_time_ms")),
                contact_type=str(row.get("contact_type", "unknown")),
                stride_time_ms=_none_if_nan(row.get("stride_time_ms")),
                stance_time_ms=_none_if_nan(row.get("stance_time_ms")),
                swing_time_ms=_none_if_nan(row.get("swing_time_ms")),
                step_length_px=_none_if_nan(row.get("step_length_px")),
                step_length_m=_none_if_nan(row.get("step_length_m")),
                stride_length_px=_none_if_nan(row.get("stride_length_px")),
                stride_length_m=_none_if_nan(row.get("stride_length_m")),
                quality_score=_none_if_nan(row.get("quality_score")),
                overlap_flag=_to_bool(row.get("overlap_flag", False)),
                low_confidence_flag=_to_bool(row.get("low_confidence_flag", False)),
                excluded_from_primary_metrics=_to_bool(
                    row.get("excluded_from_primary_metrics", False)
                ),
            )
        )
    return records


def _load_duration_s(summary_json: Path | None, steps_df: pd.DataFrame) -> float:
    if summary_json and summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        duration = d.get("duration_s")
        if isinstance(duration, (int, float)):
            return float(duration)
    # Fallback: infer from IC time span (conservative)
    if len(steps_df) > 1:
        t0 = float(steps_df["ic_time_ms"].min())
        t1 = float(steps_df["ic_time_ms"].max())
        return max(0.0, (t1 - t0) / 1000.0)
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate all-steps and primary-only summaries from steps CSV."
    )
    ap.add_argument("--steps-csv", type=Path, required=True, help="Input steps CSV")
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Original summary JSON (used for accurate duration_s)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to steps CSV directory)",
    )
    args = ap.parse_args()

    if not args.steps_csv.exists():
        raise FileNotFoundError(f"Steps CSV not found: {args.steps_csv}")

    out_dir = args.output_dir or args.steps_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_df = pd.read_csv(args.steps_csv)
    duration_s = _load_duration_s(args.summary_json, steps_df)
    scale_m_per_px = None

    all_records = _records_from_steps_df(steps_df)
    all_summary = compute_summary(all_records, duration_s, scale_m_per_px=scale_m_per_px)

    primary_df = steps_df[
        ~steps_df.get("excluded_from_primary_metrics", pd.Series([False] * len(steps_df))).astype(bool)
    ].copy()
    primary_records = _records_from_steps_df(primary_df)
    primary_summary = compute_summary(
        primary_records, duration_s, scale_m_per_px=scale_m_per_px
    )

    with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
        json.dump(all_summary.to_dict(), f, indent=2)
    with open(out_dir / "summary_primary_only.json", "w", encoding="utf-8") as f:
        json.dump(primary_summary.to_dict(), f, indent=2)

    quality_report = {
        "n_steps_all": len(all_records),
        "n_steps_primary_only": len(primary_records),
        "n_steps_excluded": len(all_records) - len(primary_records),
        "excluded_pct": (
            100.0 * (len(all_records) - len(primary_records)) / len(all_records)
            if len(all_records) > 0
            else 0.0
        ),
        "duration_s": duration_s,
    }
    with open(out_dir / "quality_report.json", "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)

    print(f"Wrote: {out_dir / 'summary_all.json'}")
    print(f"Wrote: {out_dir / 'summary_primary_only.json'}")
    print(f"Wrote: {out_dir / 'quality_report.json'}")
    print(
        f"Steps: all={quality_report['n_steps_all']}, "
        f"primary={quality_report['n_steps_primary_only']}, "
        f"excluded={quality_report['n_steps_excluded']} "
        f"({quality_report['excluded_pct']:.1f}%)"
    )


if __name__ == "__main__":
    main()

