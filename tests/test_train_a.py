"""Training aggregator A on the episode cache (session 03, step 4)."""

import json
from pathlib import Path

import numpy as np
import pytest

from sentinel.aggregators.rules import FEATURE_NAMES, RulesAggregator, scene_to_features
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import CachedEpisode
from sentinel.eval.train_a import (
    episode_score,
    f1_threshold,
    fit_a,
    quantile_thresholds,
    train_windows,
)
from sentinel.schema import Interaction, Person, SceneDescription, Summary, Window, Zone

CONFIG = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)
ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(rng: np.random.Generator, fight: bool, people: int = 2) -> SceneDescription:
    persons = tuple(
        Person(
            id=i,
            speed=float(rng.uniform(0.5, 2.5) if fight else rng.uniform(0.0, 1.2)),
            limb_speed=float(rng.uniform(1.0, 3.0) if fight else rng.uniform(0.0, 1.5)),
            limb_accel=float(rng.uniform(0, 30)),
            pose_motion=float(rng.uniform(0, 1)),
            motion=float(rng.uniform(0, 30)),
            holds=None,
            conf=0.9,
        )
        for i in range(people)
    )
    pairs = (Interaction(pair=(0, 1), distance_h=0.6, approach_speed=0.2, facing=None, duration_s=1.0),)
    return SceneDescription(
        window=Window(t_start=0.0, t_end=1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(
            n_persons=people,
            crowd_motion=float(np.mean([p.speed for p in persons])) if people else 0.0,
            motion_energy=float(rng.uniform(2, 12)),
            motion_area=float(rng.uniform(0, 10)),
            weapon_visible=False,
        ),
        persons=persons,
        interactions=pairs if people >= 2 else (),
    )


def _episodes(n: int, seed: int, part: str) -> list[CachedEpisode]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        fight = i % 2 == 0
        scenes = [_scene(rng, fight) for _ in range(4)] + [_scene(rng, fight, people=0)]
        meta = {"episode_id": f"{part}/{i}", "label": int(fight), "part": part, "perception_version": "3.0"}
        out.append(CachedEpisode(meta, scenes, [{} for _ in scenes]))
    return out


def test_windows_without_people_are_not_used_for_training():
    x, y = train_windows(_episodes(10, 0, "train"), CONFIG)
    assert x.shape == (40, len(FEATURE_NAMES)) and y.tolist().count(1) == 20


def test_fitted_a_separates_the_classes_and_reproduces_sklearn_exactly():
    train, val = _episodes(80, 1, "train"), _episodes(40, 2, "val")
    result = fit_a(train, val, CONFIG, c_grid=(0.01, 1.0), seed=2026)
    assert result.c in (0.01, 1.0) and result.window_auc_val > 0.8
    aggregator = result.aggregator
    rng = np.random.default_rng(3)
    for scene in [_scene(rng, fight) for fight in (True, False) for _ in range(5)]:
        expected = result.pipeline.predict_proba(scene_to_features(scene, CONFIG)[None, :])[0, 1]
        assert aggregator.assess(scene).risk == pytest.approx(
            expected, abs=0.006
        )  # risk is rounded to 2 places


def test_empty_scene_still_gets_zero_risk_from_a_trained_a():
    result = fit_a(_episodes(40, 1, "train"), _episodes(20, 2, "val"), CONFIG, c_grid=(1.0,), seed=2026)
    rng = np.random.default_rng(0)
    assert result.aggregator.assess(_scene(rng, True, people=0)).risk == 0.0


@pytest.mark.parametrize(
    ("rule", "expected"), [("max", 0.9), ("top2", pytest.approx(0.8)), ("mean", pytest.approx(0.4))]
)
def test_episode_score_from_window_risks(rule: str, expected):
    assert episode_score([0.1, 0.9, 0.7, 0.0, 0.3], rule) == expected


def test_f1_threshold_picks_the_best_cut():
    scores = np.array([0.1, 0.2, 0.35, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 0, 1, 1])
    threshold, f1 = f1_threshold(scores, labels)  # >= 0.35: 3 true, 1 false positive, none missed
    assert threshold == pytest.approx(0.35) and f1 == pytest.approx(6 / 7)


def test_automaton_thresholds_come_from_calm_windows():
    calm = np.linspace(0.0, 1.0, 101)
    high, low = quantile_thresholds(calm, high_q=0.95, low_q=0.8)
    assert high == pytest.approx(0.95) and low == pytest.approx(0.8)


def test_saved_model_loads_with_its_metadata_and_refuses_another_perception(tmp_path: Path):
    result = fit_a(_episodes(40, 1, "train"), _episodes(20, 2, "val"), CONFIG, c_grid=(1.0,), seed=2026)
    path = tmp_path / "a.json"
    result.aggregator.save(path, extra={"perception_version": "3.0", "c": result.c})
    data = json.loads(path.read_text())
    assert data["schema_version"] == 3 and data["perception_version"] == "3.0" and data["c"] == 1.0
    loaded = RulesAggregator.load(path, CONFIG)
    assert np.allclose(loaded.weights, result.aggregator.weights)
    data["perception_version"] = "2.0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="perception"):
        RulesAggregator.load(path, CONFIG)
