"""Signal helpers that work on irregularly sampled data."""

from __future__ import annotations

import numpy as np


def moving_average(t, x, window_s: float, centred: bool = True) -> np.ndarray:
    """Time-based moving average (handles irregular sampling and NaNs).

    ``centred=True`` averages over ``[t - w/2, t + w/2]`` (offline).
    ``centred=False`` averages over ``[t - w, t]`` (causal, for real time).
    ``window_s <= 0`` returns a copy of ``x``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if window_s <= 0 or x.size == 0:
        return x.copy()
    valid = np.isfinite(x)
    csum = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    ccount = np.concatenate([[0], np.cumsum(valid)])
    # a tiny tolerance makes the window symmetric despite float rounding of t
    eps = 1e-9 * max(window_s, 1.0)
    if centred:
        lo = np.searchsorted(t, t - window_s / 2 - eps, side="left")
        hi = np.searchsorted(t, t + window_s / 2 + eps, side="right")
    else:
        lo = np.searchsorted(t, t - window_s - eps, side="left")
        hi = np.arange(1, t.size + 1)
    n = ccount[hi] - ccount[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (csum[hi] - csum[lo]) / n, np.nan)


def derivative(t, x) -> np.ndarray:
    """dx/dt with second-order differences (non-uniform spacing allowed)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros_like(x)
    return np.gradient(x, t)


def jerk(t, acc, window_s: float = 1.0) -> np.ndarray:
    """Jerk (m/s^3) as the derivative of the smoothed acceleration.

    Differentiating raw 100 Hz IMU data amplifies sensor noise by orders of
    magnitude: the raw nuScenes signals give a median |jerk| of about 5-7 m/s^3,
    far above every comfort threshold. A moving average of about 1 s brings it
    back to physically meaningful values (median about 0.2-0.3 m/s^3).
    """
    return derivative(t, moving_average(t, acc, window_s))
