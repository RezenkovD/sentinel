"""The test experiment (session 06, step 1): statistics, alarm timing and the freeze check."""

import json

import numpy as np
import pytest

from sentinel.eval.cache import CachedEpisode
from sentinel.eval.experiment import (
    _onsets,
    alarm_timing,
    at_threshold,
    bootstrap,
    episode_metrics,
    segments_of,
    weapon_segments,
    window_auc,
)
from sentinel.schema import SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _episode(labels: list[int], label: int = 1) -> CachedEpisode:
    scenes = [
        SceneDescription(
            window=Window(t_start=float(t), t_end=t + 1.0, source="cam-1"),
            zone=ZONE,
            summary=Summary(n_persons=0, crowd_motion=0.0, weapon_visible=False),
        )
        for t in range(len(labels))
    ]
    return CachedEpisode(
        {"episode_id": "e", "label": label, "part": "test"}, scenes, [{"label": v} for v in labels]
    )


def test_metrics_at_the_val_threshold():
    scores = np.array([0.9, 0.8, 0.3, 0.6, 0.1])
    labels = np.array([1, 1, 1, 0, 0])
    m = at_threshold(scores, labels, 0.5)
    assert m["confusion"] == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}
    assert m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_bootstrap_is_seeded_and_brackets_the_estimate():
    values = np.array([0.0] * 30 + [1.0] * 70)

    def mean(idx):
        return float(values[idx].mean())

    ci = bootstrap(mean, len(values))
    assert ci == bootstrap(mean, len(values))  # same seed, same interval
    assert ci[0] < 0.7 < ci[1] and ci[1] - ci[0] < 0.25


def test_bootstrap_skips_resamples_where_the_statistic_is_undefined():
    calls = []

    def sometimes(idx):
        calls.append(1)
        return None if len(calls) % 2 else 1.0

    assert bootstrap(sometimes, 10, resamples=10) == [1.0, 1.0]
    assert bootstrap(lambda idx: None, 10, resamples=5) is None


def test_episode_metrics_carry_auc_intervals_and_the_roc():
    rng = np.random.default_rng(0)
    labels = np.array([1] * 50 + [0] * 50)
    scores = labels * 0.3 + rng.random(100)
    m = episode_metrics(scores, labels, 0.6)
    assert 0.5 < m["auc"] < 1.0 and m["auc_ci"][0] < m["auc"] < m["auc_ci"][1]
    assert m["roc"]["fpr"][0] == 0.0 and m["roc"]["tpr"][-1] == 1.0
    assert m["episodes"] == 100 and m["positives"] == 50


def test_window_auc_resamples_whole_episodes():
    per = [(np.array([1, 1, 0]), np.array([0.9, 0.8, 0.1])), (np.array([0, 0]), np.array([0.2, 0.3]))]
    w = window_auc(per)
    assert w["auc"] == 1.0 and w["windows"] == 5 and w["positive_windows"] == 2


def test_violent_segments_and_alarm_onsets():
    assert segments_of([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert segments_of([1, 1]) == [(0, 1)] and segments_of([0, 0]) == []
    assert _onsets([True, True, False, True]) == [0, 3]


def test_alarm_timing_delay_and_false_alarms_outside_segments():
    # violence in windows 2..4; the alarm starts at window 3 (delay = end of 3 − start of 2 = 2 s) and lasts
    # to window 6 — inside M + 5 s after the segment, so not false; a second onset at window 19 is false
    labels = [0, 0, 1, 1, 1] + [0] * 15
    flags = [False] * 3 + [True] * 4 + [False] * 12 + [True]
    t = alarm_timing([_episode(labels)], [flags], quiet_s=5.0)
    assert t["segments"] == 1 and t["detected"] == 1.0
    assert t["median_delay_s"] == pytest.approx(2.0)
    assert t["false_onsets"] == 1
    assert t["normal_h"] == pytest.approx(17 / 3600)
    assert t["false_per_h"] == pytest.approx(1 / (17 / 3600))


def test_weapon_segments_recall_time_and_false_onsets():
    hit = _episode([0, 1, 1, 0, 0, 0])
    missed = _episode([0, 0, 1, 0])
    alarms = {id(hit): [False, False, True, False, True, False], id(missed): [True, False, False, False]}
    r = weapon_segments([hit, missed], lambda e: alarms[id(e)])
    assert r["recall"] == 0.5 and r["times_to_alarm_s"] == [pytest.approx(2.0)]
    assert r["alarms_before_weapon"] == 1  # the missed segment alarmed before its weapon appeared
    assert r["false_onsets"] == 2  # window 4 of the first segment and window 0 of the second
    assert r["weapon_free_h"] == pytest.approx(7 / 3600)


def test_check_frozen_refuses_when_anything_changed(tmp_path, monkeypatch):
    from sentinel.eval import experiment

    state = {"config": {"sha256": "a"}, "aggregator_a": {"sha256": "b"}}
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"date": "2026-10-04T23:00:00", "state": state}))
    monkeypatch.setattr(experiment, "frozen_state", lambda *a: dict(state))
    assert experiment.check_frozen(None, None, None, freeze)["state"] == state
    monkeypatch.setattr(experiment, "frozen_state", lambda *a: state | {"aggregator_a": {"sha256": "c"}})
    with pytest.raises(SystemExit, match="aggregator_a"):
        experiment.check_frozen(None, None, None, freeze)
    with pytest.raises(SystemExit, match="missing"):
        experiment.check_frozen(None, None, None, tmp_path / "none.json")


def test_config_digest_ignores_b_and_the_cascade_but_not_a_or_the_automaton():
    from sentinel.config import DEFAULT_CONFIG, load_config
    from sentinel.eval.experiment import config_digest

    config = load_config(DEFAULT_CONFIG)
    grey = config.model_copy(update={"aggregation": config.aggregation.model_copy(update={"g_low": 0.3})})
    assert config_digest(grey) == config_digest(config)
    theta = config.model_copy(update={"incidents": config.incidents.model_copy(update={"theta_low": 0.6})})
    assert config_digest(theta) != config_digest(config)
