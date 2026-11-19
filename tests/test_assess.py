from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from avcomfort import AssessOptions, RealtimeAssessor, assess, load_csv, summarise
from avcomfort.signals import jerk, moving_average

DATA = Path(__file__).parent / "data"


def test_reproduces_assesstool_v1_exactly():
    """The 2022 V1 output for the Bath scene-02 drive is reproduced sample by sample."""
    drive = load_csv(DATA / "bath_scene02.csv")
    expected = pd.read_csv(DATA / "bath_scene02_v1_output.csv", comment="#")
    res = assess(drive, options=AssessOptions.legacy())
    assert res["level"].tolist() == expected["level"].tolist()
    assert np.allclose(res["score"], expected["score"], atol=1e-9)


def test_moving_average_irregular_sampling():
    t = np.array([0.0, 0.1, 0.5, 0.6, 2.0])
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    assert np.allclose(moving_average(t, x, 0.0), x)
    assert moving_average(t, x, 0.3)[1] == pytest.approx(2.0)  # 0 and 0.1
    assert moving_average(t, x, 0.3, centred=False)[3] == pytest.approx(6.0)  # 0.5 and 0.6
    assert moving_average(t, np.array([1.0, np.nan, 3.0, np.nan, 5.0]), 10)[0] == pytest.approx(3.0)


def test_jerk_of_ramp():
    t = np.arange(0, 10, 0.01)
    a = 0.5 * t  # constant jerk 0.5 m/s^3
    j = jerk(t, a, window_s=1.0)
    assert np.allclose(j[100:-100], 0.5, atol=1e-6)


def _synthetic_drive(n=3000, fs=100.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    lon = 1.5 * np.sin(2 * np.pi * t / 15) + rng.normal(0, 0.05, n)
    lat = 1.0 * np.sin(2 * np.pi * t / 9) + rng.normal(0, 0.05, n)
    return pd.DataFrame({"time_s": t, "acc_lat": lat, "acc_long": lon, "x_m": t, "y_m": 0 * t})


def test_assess_columns_and_summary():
    res = assess(_synthetic_drive())
    for c in ("jerk_lat", "jerk_long", "level", "rating", "score", "x_m"):
        assert c in res.columns
    s = summarise(res)
    assert s["samples"] == 3000
    assert sum(s["time_share_pct"].values()) == pytest.approx(100, abs=0.2)
    assert 0 <= s["mean_score"] <= 100
    assert s["overall_rating"] in {"A*", "A", "B", "C", "D", "E", "F"}


def test_jerk_never_improves_rating():
    drive = _synthetic_drive()
    with_jerk = assess(drive)
    without = assess(drive, options=AssessOptions(use_jerk=False))
    assert np.all(with_jerk["level"] >= without["level"])


def test_missing_columns():
    with pytest.raises(ValueError, match="missing"):
        assess(pd.DataFrame({"time_s": [0, 1], "acc_lat": [0, 0]}))


def test_realtime_matches_offline_levels_without_jerk():
    drive = _synthetic_drive(500)
    rt = RealtimeAssessor(use_jerk=False)
    online = [rt.update(*row).level for row in drive[["time_s", "acc_lat", "acc_long"]].to_numpy()]
    offline = assess(drive, options=AssessOptions(use_jerk=False))["level"].tolist()
    assert online == offline


def test_realtime_jerk_is_sane():
    t = np.arange(0, 5, 0.01)
    rt = RealtimeAssessor(jerk_window_s=1.0)
    samples = [rt.update(ti, 0.0, 0.4 * ti) for ti in t]
    assert samples[-1].jerk_long == pytest.approx(0.4, rel=1e-6)
    assert samples[-1].jerk_lat == 0.0
