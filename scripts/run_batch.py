#!/usr/bin/env python3
"""
Batch gait analysis: process a manifest CSV of videos and produce
per-trial events/steps/summary + aggregated CSVs.

Usage
-----
::

    python scripts/run_batch.py manifest.csv \\
        --model models/foot_pose/best.pt \\
        --output-dir results/batch_2024

The manifest CSV must have at least a ``video_path`` column.
Optional columns: ``participant_id``, ``trial_id``, ``condition``,
``group``, ``notes``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
_src = _project_root / "src"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gait_analysis.inference_pipeline import PipelineConfig
from gait_analysis.batch_pipeline import run_batch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Batch gait analysis from a manifest CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Manifest CSV format:
  participant_id,trial_id,video_path,condition,group,notes
  P001,T01,videos/P001_trial1.mp4,barefoot,ASD,
  P001,T02,videos/P001_trial2.mp4,shoes,ASD,

Only video_path is required.
""",
    )
    ap.add_argument("manifest", type=Path, help="Manifest CSV")
    ap.add_argument("--model", type=str, default="models/foot_pose/best.pt",
                    help="YOLO model path")
    ap.add_argument("--output-dir", type=Path, default=Path("results/batch"),
                    help="Output directory")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Keypoint confidence threshold")
    ap.add_argument("--device", type=str, default="auto",
                    help="Inference device (auto/cpu/cuda)")
    ap.add_argument("--scale-m-per-px", type=float, default=None,
                    help="Meters per pixel for length calibration")
    ap.add_argument("--velocity-threshold", type=float, default=2.0)
    ap.add_argument("--stability-frames", type=int, default=5)
    ap.add_argument("--min-step-duration-ms", type=float, default=200.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.manifest.exists():
        print(f"Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    config = PipelineConfig(
        velocity_threshold=args.velocity_threshold,
        stability_frames=args.stability_frames,
        min_step_duration_ms=args.min_step_duration_ms,
        scale_m_per_px=args.scale_m_per_px,
    )

    results = run_batch(
        manifest_path=args.manifest,
        model_path=args.model,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
        conf_threshold=args.conf,
    )

    ok = sum(1 for r in results if r.error is None)
    fail = len(results) - ok
    print(f"\nBatch complete: {ok} succeeded, {fail} failed")
    print(f"Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
