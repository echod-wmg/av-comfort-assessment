"""Streaming (sample-by-sample) assessment for an on-board data logger."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .model import ComfortModel, default_model


@dataclass
class Sample:
    time_s: float
    acc_lat: float
    acc_long: float
    jerk_lat: float
    jerk_long: float
    level: int
    rating: str
    score: float


class RealtimeAssessor:
    """Assess comfort one IMU sample at a time.

    Jerk is the slope of a causal moving average of the acceleration. The
    smoothed values over the last ``jerk_window_s`` seconds are compared at
    both ends of the window, which needs no future samples. This replaces the
    2022 logger's raw ``(a - a_prev) / dt``, which made jerk too noisy to use.
    """

    def __init__(self, model: Optional[ComfortModel] = None, jerk_window_s: float = 1.0, use_jerk: bool = True):
        self.model = model or default_model()
        self.window = float(jerk_window_s)
        self.use_jerk = use_jerk
        self._raw = deque()  # (t, lat, long)
        self._smooth = deque()  # (t, lat_avg, long_avg)

    def update(self, time_s: float, acc_lat: float, acc_long: float) -> Sample:
        self._raw.append((time_s, acc_lat, acc_long))
        while self._raw and self._raw[0][0] < time_s - self.window:
            self._raw.popleft()
        arr = np.array(self._raw)
        avg = (time_s, float(arr[:, 1].mean()), float(arr[:, 2].mean()))
        self._smooth.append(avg)
        while len(self._smooth) > 1 and self._smooth[0][0] < time_s - self.window:
            self._smooth.popleft()
        t0, lat0, long0 = self._smooth[0]
        dt = time_s - t0
        j_lat = (avg[1] - lat0) / dt if dt > 0 else 0.0
        j_long = (avg[2] - long0) / dt if dt > 0 else 0.0
        if self.use_jerk:
            level = int(self.model.classify(acc_lat, acc_long, j_lat, j_long)[0])
        else:
            level = int(self.model.classify(acc_lat, acc_long)[0])
        score = float(self.model.score(level, acc_lat, acc_long)[0])
        return Sample(time_s, acc_lat, acc_long, j_lat, j_long, level, self.model.ratings[level], score)
