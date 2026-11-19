"""The comfort model: acceleration / jerk threshold regions and the 0-100 score.

Comfort levels follow the VeriCAV occupant-comfort scale (A* ... F). Each
level is a *kite* (rhombus) in the lateral / longitudinal plane, which allows
different limits for lateral, accelerating and braking motion. A sample
belongs to the first level whose acceleration kite **and** jerk kite both
contain it. The levels are nested, so in practice the result is the worse of
the acceleration level and the jerk level.

The default thresholds are the values used in the 2022 dissertation code
(AssessTool-V1). They come from Bae et al. (2019), Karjanto et al. (2017) and
the VeriCAV project. The dissertation's Table 3 lists lateral limits of
3.7 / 4.3 m/s^2 for "acceptable" / "poor", while the code used 3.6 / 4.2.
The code values are kept so that the published results reproduce exactly.
Pass a custom :class:`ComfortModel` (or JSON file) to change them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class Kite:
    """Rhombus centred on the origin in the (lateral, longitudinal) plane.

    ``lateral``: half-width (left and right). ``accel``: forward extent
    (positive longitudinal). ``decel``: rearward extent (braking), given as a
    positive number.
    """

    lateral: float
    accel: float
    decel: float

    @classmethod
    def symmetric(cls, limit: float) -> "Kite":
        return cls(limit, limit, limit)

    def normalised(self, lat, long):
        """``< 1`` inside, ``1`` on the boundary, ``> 1`` outside."""
        lat = np.abs(np.asarray(lat, dtype=float))
        long = np.asarray(long, dtype=float)
        along = np.where(long >= 0, long / self.accel, -long / self.decel)
        return lat / self.lateral + along

    def contains(self, lat, long):
        # strict interior, like shapely's Polygon.contains used in V1
        return self.normalised(lat, long) < 1.0

    def vertices(self):
        """Closed outline for plotting: (lateral, longitudinal) arrays."""
        x = np.array([-self.lateral, 0.0, self.lateral, 0.0, -self.lateral])
        y = np.array([0.0, -self.decel, 0.0, self.accel, 0.0])
        return x, y


@dataclass(frozen=True)
class ComfortLevel:
    name: str
    rating: str
    colour: str
    accel: Optional[Kite]  # None = catch-all (worst) level
    jerk: Optional[Kite]
    score_base: float
    score_points: float
    score_divisor_accel: float  # used when longitudinal acceleration >= 0
    score_divisor_decel: float  # used when braking


@dataclass
class ComfortModel:
    levels: List[ComfortLevel] = field(default_factory=list)
    #: overall rating from a mean score (dissertation Table 4): (min score, rating)
    rating_bands: List[tuple] = field(
        default_factory=lambda: [(90, "A*"), (80, "A"), (75, "B"), (65, "C"), (60, "D"), (35, "E"), (-np.inf, "F")]
    )

    def __post_init__(self):
        if not self.levels:
            raise ValueError("a comfort model needs at least one level")
        if self.levels[-1].accel is not None:
            raise ValueError("the last level must be the catch-all level (accel=None)")

    @property
    def ratings(self) -> List[str]:
        return [lv.rating for lv in self.levels]

    def classify(self, acc_lat, acc_long, jerk_lat=None, jerk_long=None) -> np.ndarray:
        """Comfort level index per sample (0 = best). Jerk ``None`` = ignore jerk."""
        acc_lat = np.atleast_1d(np.asarray(acc_lat, dtype=float))
        acc_long = np.atleast_1d(np.asarray(acc_long, dtype=float))
        use_jerk = jerk_lat is not None and jerk_long is not None
        if use_jerk:
            jerk_lat = np.atleast_1d(np.asarray(jerk_lat, dtype=float))
            jerk_long = np.atleast_1d(np.asarray(jerk_long, dtype=float))
        level = np.full(acc_lat.shape, len(self.levels) - 1, dtype=int)
        for i in range(len(self.levels) - 2, -1, -1):
            lv = self.levels[i]
            ok = lv.accel.contains(acc_lat, acc_long)
            if use_jerk:
                ok &= lv.jerk.contains(jerk_lat, jerk_long)
            level[ok] = i
        return level

    def score(self, level, acc_lat, acc_long) -> np.ndarray:
        """0-100 comfort score (V1 formula), clipped to [0, 100].

        ``score = base - points * (|a_lat| + |a_long|) / divisor``, where base,
        points and divisor depend on the level and the divisor also depends
        on whether the vehicle is braking.
        """
        level = np.atleast_1d(np.asarray(level, dtype=int))
        acc_lat = np.atleast_1d(np.asarray(acc_lat, dtype=float))
        acc_long = np.atleast_1d(np.asarray(acc_long, dtype=float))
        base = np.array([lv.score_base for lv in self.levels])[level]
        pts = np.array([lv.score_points for lv in self.levels])[level]
        div_a = np.array([lv.score_divisor_accel for lv in self.levels])[level]
        div_d = np.array([lv.score_divisor_decel for lv in self.levels])[level]
        divisor = np.where(acc_long < 0, div_d, div_a)
        s = base - pts * (np.abs(acc_lat) + np.abs(acc_long)) / divisor
        return np.clip(s, 0.0, 100.0)

    def overall_rating(self, mean_score: float) -> str:
        for threshold, rating in self.rating_bands:
            if mean_score > threshold:
                return rating
        return self.rating_bands[-1][1]

    # -- persistence -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "levels": [asdict(lv) for lv in self.levels],
            "rating_bands": [[float(t), r] for t, r in self.rating_bands],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComfortModel":
        levels = []
        for d in data["levels"]:
            d = dict(d)
            d["accel"] = Kite(**d["accel"]) if d.get("accel") else None
            d["jerk"] = Kite(**d["jerk"]) if d.get("jerk") else None
            levels.append(ComfortLevel(**d))
        kwargs = {"levels": levels}
        if "rating_bands" in data:
            kwargs["rating_bands"] = [(float(t), r) for t, r in data["rating_bands"]]
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ComfortModel":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def default_model() -> ComfortModel:
    """Thresholds and score parameters used in AssessTool-V1 (2022)."""
    sym = Kite.symmetric
    return ComfortModel(
        [
            ComfortLevel("Extremely comfortable", "A*", "darkgreen", sym(0.3), sym(0.2), 100, 10, 0.6, 0.6),
            ComfortLevel("Very comfortable", "A", "green", sym(0.6), sym(0.4), 90, 10, 1.2, 1.2),
            ComfortLevel("Comfortable", "B", "limegreen", sym(0.9), sym(0.6), 80, 5, 1.8, 1.8),
            ComfortLevel("Acceptable", "C", "gold", Kite(3.6, 1.3, 1.8), sym(0.8), 75, 10, 5.0, 5.5),
            ComfortLevel("Poor", "D", "orange", Kite(4.2, 1.5, 2.4), sym(0.95), 65, 5, 5.8, 6.7),
            ComfortLevel("Uncomfortable", "E", "red", Kite(5.6, 3.07, 5.08), sym(2.0), 60, 25, 8.7, 10.7),
            ComfortLevel("Extremely uncomfortable", "F", "black", None, None, 35, 35, 15.2, 15.2),
        ]
    )
