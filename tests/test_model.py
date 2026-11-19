import json

import numpy as np
import pytest

from avcomfort import ComfortModel, Kite, default_model


def test_kite_regions():
    k = Kite(3.6, 1.3, 1.8)
    assert k.contains(0.0, 0.0)
    assert k.contains(3.5, 0.0) and not k.contains(3.7, 0.0)
    assert k.contains(0.0, 1.25) and not k.contains(0.0, 1.35)  # accelerating
    assert k.contains(0.0, -1.75) and not k.contains(0.0, -1.85)  # braking
    assert not k.contains(3.6, 0.0)  # boundary is outside (strict, like shapely)
    assert k.contains(-1.8, 0.6) == (1.8 / 3.6 + 0.6 / 1.3 < 1)


def test_levels_without_jerk():
    m = default_model()
    lat = [0.1, 0.5, 0.8, 2.0, 0.0, 0.0, 5.0, 7.0]
    lon = [0.1, 0.0, 0.0, 0.0, -2.0, 2.5, 0.0, 0.0]
    assert m.classify(lat, lon).tolist() == [0, 1, 2, 3, 4, 5, 5, 6]


def test_jerk_makes_level_worse_never_better():
    m = default_model()
    base = m.classify([0.1], [0.1])[0]
    assert m.classify([0.1], [0.1], [0.0], [0.0])[0] == base == 0
    assert m.classify([0.1], [0.1], [0.5], [0.0])[0] == 2  # jerk 0.5 -> "comfortable"
    assert m.classify([0.1], [0.1], [3.0], [0.0])[0] == 6
    assert m.classify([2.0], [0.0], [0.0], [0.0])[0] == 3


def test_score_formula_and_bounds():
    m = default_model()
    # level C while braking uses the larger divisor 5.5
    assert m.score([3], [1.0], [-1.0])[0] == pytest.approx(75 - 10 * 2.0 / 5.5)
    assert m.score([3], [1.0], [0.5])[0] == pytest.approx(75 - 10 * 1.5 / 5.0)
    assert m.score([6], [40.0], [0.0])[0] == 0.0  # clipped


@pytest.mark.parametrize("score,rating", [(95, "A*"), (85, "A"), (77, "B"), (70, "C"), (62, "D"), (40, "E"), (10, "F")])
def test_overall_rating_bands(score, rating):
    assert default_model().overall_rating(score) == rating


def test_model_json_roundtrip(tmp_path):
    m = default_model()
    p = tmp_path / "model.json"
    p.write_text(json.dumps(m.to_dict()))
    m2 = ComfortModel.from_json(p)
    rng = np.random.default_rng(0)
    lat, lon, jl, jg = rng.normal(0, 2, (4, 500))
    assert np.array_equal(m.classify(lat, lon, jl, jg), m2.classify(lat, lon, jl, jg))
    assert m2.ratings == m.ratings


def test_last_level_must_be_catch_all():
    lv = default_model().levels[0]
    with pytest.raises(ValueError):
        ComfortModel([lv])
