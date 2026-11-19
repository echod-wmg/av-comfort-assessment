"""Offline comfort assessment of a drive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .model import ComfortModel, default_model
from .signals import jerk, moving_average

REQUIRED = ("time_s", "acc_lat", "acc_long")


@dataclass
class AssessOptions:
    #: moving-average window applied to acceleration before classification (0 = raw)
    accel_window_s: float = 0.0
    #: moving-average window applied before differentiating for jerk
    jerk_window_s: float = 1.0
    #: include jerk in the classification (V1 effectively ignored it)
    use_jerk: bool = True

    @classmethod
    def legacy(cls) -> "AssessOptions":
        """Reproduce AssessTool-V1: raw acceleration, jerk not used."""
        return cls(accel_window_s=0.0, use_jerk=False)


def assess(
    drive: pd.DataFrame,
    model: Optional[ComfortModel] = None,
    options: Optional[AssessOptions] = None,
) -> pd.DataFrame:
    """Classify every sample of a drive.

    ``drive`` needs ``time_s``, ``acc_lat`` (+ left) and ``acc_long``
    (+ forward / accelerating) in m/s^2. Other columns, such as position and
    speed, are passed through. Adds ``acc_lat_f``, ``acc_long_f`` (the
    filtered acceleration used), ``jerk_lat``, ``jerk_long``, ``level``,
    ``rating`` and ``score``.
    """
    missing = [c for c in REQUIRED if c not in drive.columns]
    if missing:
        raise ValueError(f"drive data is missing columns {missing}")
    model = model or default_model()
    opt = options or AssessOptions()
    df = drive.sort_values("time_s").reset_index(drop=True).copy()
    df = df.dropna(subset=["acc_lat", "acc_long"]).reset_index(drop=True)
    t = df["time_s"].to_numpy()
    lat = moving_average(t, df["acc_lat"].to_numpy(), opt.accel_window_s)
    lon = moving_average(t, df["acc_long"].to_numpy(), opt.accel_window_s)
    df["acc_lat_f"], df["acc_long_f"] = lat, lon
    df["jerk_lat"] = jerk(t, df["acc_lat"].to_numpy(), opt.jerk_window_s)
    df["jerk_long"] = jerk(t, df["acc_long"].to_numpy(), opt.jerk_window_s)
    if opt.use_jerk:
        level = model.classify(lat, lon, df["jerk_lat"], df["jerk_long"])
    else:
        level = model.classify(lat, lon)
    df["level"] = level
    df["rating"] = np.array(model.ratings, dtype=object)[level]
    df["score"] = model.score(level, lat, lon)
    return df


def summarise(result: pd.DataFrame, model: Optional[ComfortModel] = None) -> dict:
    """Scene-level summary: mean score, overall rating and time share per level."""
    model = model or default_model()
    t = result["time_s"].to_numpy()
    # time-weighted shares (each sample represents the interval to the next one)
    dt = np.diff(t, append=t[-1] + (np.median(np.diff(t)) if t.size > 1 else 1.0))
    share = {r: 0.0 for r in model.ratings}
    for lv, w in zip(result["level"].to_numpy(), dt):
        share[model.ratings[lv]] += float(w)
    total = float(dt.sum())
    mean_score = float(np.average(result["score"], weights=dt))
    return {
        "duration_s": float(t[-1] - t[0]),
        "samples": int(len(result)),
        "mean_score": round(mean_score, 2),
        "min_score": round(float(result["score"].min()), 2),
        "overall_rating": model.overall_rating(mean_score),
        "time_share_pct": {k: round(100 * v / total, 1) for k, v in share.items()},
        "max_abs_acc_lat": round(float(result["acc_lat_f"].abs().max()), 3),
        "max_acc_long": round(float(result["acc_long_f"].max()), 3),
        "max_decel": round(max(0.0, float(-result["acc_long_f"].min())), 3),
        "p95_abs_jerk": round(float(np.nanpercentile(np.hypot(result["jerk_lat"], result["jerk_long"]), 95)), 3),
    }
