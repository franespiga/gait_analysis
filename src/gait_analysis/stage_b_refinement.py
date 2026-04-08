"""
Stage B foot refinement — two-stage inference safeguard.

When Stage A (full-frame HALPE-26) detects low foot confidence, inter-foot
overlap, or high keypoint jitter, this module crops a region around the
lower legs / feet, runs a second pose model on the crop, and replaces the
foot keypoints in the full-frame result with the higher-confidence crop
predictions.

This module is backward-compatible: if Stage B is disabled (default),
the pipeline behaves identically to single-stage inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

LOG = logging.getLogger(__name__)

FOOT_KPT_INDICES = [15, 16, 20, 21, 22, 23, 24, 25]

L_ANKLE, R_ANKLE = 15, 16


@dataclass
class StageBConfig:
    """Configuration for Stage B foot refinement."""
    enabled: bool = False
    model_path: str = "models/halpe26_stage_b/best.pt"
    trigger_min_foot_conf: float = 0.3
    trigger_overlap_flag: bool = True
    trigger_max_jitter_px: float = 15.0
    crop_margin: float = 0.4
    crop_min_px: int = 128
    crop_max_px: int = 384


def needs_stage_b(
    kpts: np.ndarray,
    prev_kpts: Optional[np.ndarray],
    cfg: StageBConfig,
) -> bool:
    """Decide whether Stage B should run for a single frame.

    Parameters
    ----------
    kpts : ndarray (K, 3)
        Keypoints for this frame (x, y, conf).
    prev_kpts : ndarray (K, 3) or None
        Keypoints from the previous frame (for jitter check).
    cfg : StageBConfig
    """
    if not cfg.enabled:
        return False

    # Low foot confidence
    foot_confs = [kpts[i, 2] for i in FOOT_KPT_INDICES if i < kpts.shape[0]]
    if foot_confs:
        mean_foot_conf = float(np.mean(foot_confs))
        if mean_foot_conf < cfg.trigger_min_foot_conf:
            return True

    # Overlap heuristic: left and right foot centroids very close
    if cfg.trigger_overlap_flag:
        left_pts = [kpts[i, :2] for i in [15, 20, 22, 24]
                     if i < kpts.shape[0] and kpts[i, 2] > 0.1]
        right_pts = [kpts[i, :2] for i in [16, 21, 23, 25]
                      if i < kpts.shape[0] and kpts[i, 2] > 0.1]
        if left_pts and right_pts:
            left_c = np.mean(left_pts, axis=0)
            right_c = np.mean(right_pts, axis=0)
            dist = float(np.linalg.norm(left_c - right_c))
            if dist < 30:
                return True

    # Jitter
    if prev_kpts is not None:
        jitters = []
        for i in FOOT_KPT_INDICES:
            if i < kpts.shape[0] and i < prev_kpts.shape[0]:
                if kpts[i, 2] > 0.1 and prev_kpts[i, 2] > 0.1:
                    d = float(np.linalg.norm(kpts[i, :2] - prev_kpts[i, :2]))
                    jitters.append(d)
        if jitters and np.mean(jitters) > cfg.trigger_max_jitter_px:
            return True

    return False


def compute_foot_crop(
    kpts: np.ndarray,
    img_w: int,
    img_h: int,
    margin: float = 0.4,
    min_px: int = 128,
    max_px: int = 384,
) -> Optional[tuple[int, int, int, int]]:
    """Compute a crop box around the feet / lower legs.

    Returns (x1, y1, x2, y2) or None.
    """
    pts = []
    for idx in FOOT_KPT_INDICES:
        if idx < kpts.shape[0] and kpts[idx, 2] > 0.05:
            pts.append(kpts[idx, :2])
    if not pts:
        return None

    pts_arr = np.array(pts)
    cx = float(pts_arr[:, 0].mean())
    cy = float(pts_arr[:, 1].mean())
    span = max(
        float(pts_arr[:, 0].max() - pts_arr[:, 0].min()),
        float(pts_arr[:, 1].max() - pts_arr[:, 1].min()),
        50,
    )
    half = int(span * (1 + margin) / 2)
    half = max(half, min_px // 2)
    half = min(half, max_px // 2)

    cy_shifted = cy + half * 0.1

    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy_shifted - half))
    x2 = min(img_w, int(cx + half))
    y2 = min(img_h, int(cy_shifted + half))

    if x2 - x1 < 32 or y2 - y1 < 32:
        return None
    return x1, y1, x2, y2


def refine_foot_keypoints(
    frame: np.ndarray,
    kpts: np.ndarray,
    model,
    cfg: StageBConfig,
    conf_threshold: float = 0.25,
    device: str = "cpu",
) -> np.ndarray:
    """Run Stage B model on a foot crop and merge improved keypoints.

    Parameters
    ----------
    frame : ndarray (H, W, 3)
        Original video frame.
    kpts : ndarray (K, 3)
        Stage A keypoints for the selected person.
    model : ultralytics.YOLO
        Loaded Stage B model.
    cfg : StageBConfig
    conf_threshold : float
    device : str

    Returns
    -------
    ndarray (K, 3)
        Keypoints with foot indices replaced by Stage B predictions
        if they are higher confidence.
    """
    img_h, img_w = frame.shape[:2]
    crop_box = compute_foot_crop(
        kpts, img_w, img_h,
        margin=cfg.crop_margin,
        min_px=cfg.crop_min_px,
        max_px=cfg.crop_max_px,
    )
    if crop_box is None:
        return kpts

    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return kpts

    results = model(crop, verbose=False, conf=conf_threshold, device=device)
    if not results or results[0].keypoints is None:
        return kpts

    kp_data = results[0].keypoints
    if kp_data.xy is None or len(kp_data.xy) == 0:
        return kpts

    # Select best person in crop
    if kp_data.conf is not None and len(kp_data.xy) > 1:
        mean_confs = kp_data.conf.mean(dim=1).cpu().numpy()
        best = int(np.argmax(mean_confs))
    else:
        best = 0

    crop_xy = kp_data.xy[best].cpu().numpy()
    crop_conf = (kp_data.conf[best].cpu().numpy()
                 if kp_data.conf is not None
                 else np.ones(len(crop_xy)))

    cw, ch = x2 - x1, y2 - y1
    out = kpts.copy()

    for idx in FOOT_KPT_INDICES:
        if idx >= crop_xy.shape[0] or idx >= out.shape[0]:
            continue
        stage_b_conf = float(crop_conf[idx])
        stage_a_conf = float(out[idx, 2])
        if stage_b_conf > stage_a_conf:
            abs_x = crop_xy[idx, 0] + x1
            abs_y = crop_xy[idx, 1] + y1
            out[idx, 0] = abs_x
            out[idx, 1] = abs_y
            out[idx, 2] = stage_b_conf

    return out
