"""
Temporal smoothing of keypoint trajectories for gait analysis.

Uses Savitzky-Golay filter by default (FPS-aware window when possible).
Handles missing keypoints via confidence gating and interpolation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d


def smooth_keypoints(
    keypoints: np.ndarray,
    method: str = "savgol",
    window_length: int = 11,
    polyorder: int = 3,
    conf_threshold: float = 0.0,
    axis: int = 0,
) -> np.ndarray:
    """
    Smooth keypoint coordinates over time.

    Args:
        keypoints: Shape (T, N, 3) with (x, y, conf) per keypoint. T = frames, N = num keypoints.
        method: "savgol" (Savitzky-Golay) or "none".
        window_length: Savitzky-Golay window (must be odd, <= T).
        polyorder: Savitzky-Golay polynomial order.
        conf_threshold: Only smooth (x,y) where conf >= this; else leave as-is or interpolate.
        axis: Time axis (default 0).

    Returns:
        Smoothed array same shape as keypoints. Confidence channel unchanged.
    """
    if keypoints.size == 0:
        return keypoints
    out = np.asarray(keypoints, dtype=np.float64).copy()
    T = out.shape[axis]
    if T < 2 or method == "none":
        return out

    # Ensure window is odd and <= T
    w = min(window_length | 1, T if T % 2 else T - 1)
    if w < 3:
        return out

    # Smooth each keypoint's x and y over time
    for n in range(out.shape[1]):
        x = out[:, n, 0]
        y = out[:, n, 1]
        c = out[:, n, 2]
        valid = c >= conf_threshold
        if not np.any(valid):
            continue
        if np.all(valid):
            try:
                out[:, n, 0] = savgol_filter(x, w, polyorder)
                out[:, n, 1] = savgol_filter(y, w, polyorder)
            except Exception:
                pass
        else:
            # Interpolate missing, then smooth valid segment or full after fill
            t = np.arange(T, dtype=float)
            if np.sum(valid) >= w:
                x_fill = _fill_missing(t, x, valid)
                y_fill = _fill_missing(t, y, valid)
                try:
                    sx = savgol_filter(x_fill, w, polyorder)
                    sy = savgol_filter(y_fill, w, polyorder)
                    out[:, n, 0] = np.where(valid, sx, x)
                    out[:, n, 1] = np.where(valid, sy, y)
                except Exception:
                    pass
    return out


def _fill_missing(t: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linear interpolation to fill missing values."""
    out = y.copy()
    if np.all(valid):
        return out
    t_valid = t[valid]
    y_valid = y[valid]
    if len(t_valid) < 2:
        return out
    f = interp1d(t_valid, y_valid, kind="linear", bounds_error=False, fill_value="extrapolate")
    out[~valid] = f(t[~valid])
    return out


def fps_aware_window_length(fps: float, window_sec: float = 0.15) -> int:
    """Return odd window length for Savitzky-Golay from desired duration in seconds."""
    w = max(3, int(round(fps * window_sec)))
    return w if w % 2 else w + 1
