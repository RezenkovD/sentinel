"""Pilot of aggregator B on stratified val windows (session 05, step 3)."""

import pytest

from sentinel.eval.b_pilot import consistency, pilot_sample, stratum
from sentinel.eval.cache import CachedEpisode
from sentinel.schema import Person, RiskResult, SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(t: float, people: int) -> SceneDescription:
    persons = tuple(Person(id=i, speed=0.2, limb_accel=None, holds=None, conf=0.8) for i in range(people))
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(
            n_persons=people, crowd_motion=0.2 if people else 0.0, motion_area=3.0, weapon_visible=False
        ),
        persons=persons,
    )


class FakeA:
    def assess(self, scene: SceneDescription) -> RiskResult:
        risk = 0.8 if scene.window.t_start >= 3 else 0.1
        return RiskResult(risk=risk, level="normal", reason_uk="—")


def _episode(i: int, label: int, part: str = "val") -> CachedEpisode:
    scenes = [_scene(float(t), people=0 if t == 0 else 2) for t in range(5)]
    return CachedEpisode({"episode_id": f"{part}/{i}", "part": part, "label": label}, scenes, [{}] * 5)


def test_windows_fall_into_label_people_and_a_risk_strata():
    assert stratum(1, _scene(0.0, 0), 0.9) == ("fight", "empty", "high")
    assert stratum(0, _scene(0.0, 2), 0.1) == ("calm", "people", "low")


def test_the_sample_is_stratified_seeded_and_from_val_only():
    episodes = [_episode(i, i % 2) for i in range(20)]
    sample = pilot_sample(episodes, FakeA(), per_stratum=2, seed=2026)
    strata = [w["stratum"] for w in sample]
    assert len(sample) == 2 * 6  # empty windows here are all low-risk: 6 strata exist, 2 windows each
    assert all(strata.count(s) == 2 for s in set(strata))
    assert sample == pilot_sample(episodes, FakeA(), per_stratum=2, seed=2026)
    first = pilot_sample(episodes, FakeA(), per_stratum=1, seed=2026)
    assert all(w in sample for w in first)  # a smaller pilot is a prefix of each stratum of a larger one
    with pytest.raises(ValueError, match="val"):
        pilot_sample([_episode(0, 0, "train")], FakeA(), per_stratum=1, seed=1)


def test_consistency_is_the_spread_of_repeated_answers_and_the_level_agreement():
    repeats = {
        "a": [(0.8, "alarm"), (0.7, "alarm"), (0.9, "alarm")],
        "b": [(0.2, "normal"), (0.5, "attention")],
    }
    result = consistency(repeats)
    assert result["windows"] == 2 and result["level_agreement"] == pytest.approx(0.5)
    assert result["risk_std_mean"] == pytest.approx((0.0816 + 0.15) / 2, abs=1e-3)
