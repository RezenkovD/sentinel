"""Usual values of the scene fields on calm train windows: the only data B gets (decision 01.10)."""

import pytest

from sentinel.eval.cache import CachedEpisode
from sentinel.eval.reference_stats import reference_stats
from sentinel.schema import Interaction, Person, SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(speeds: list[float | None], area: float | None, holds: str | None = None) -> SceneDescription:
    persons = tuple(
        Person(id=i, speed=s, limb_accel=None, holds=holds if i == 0 else None, conf=0.8)
        for i, s in enumerate(speeds)
    )
    pairs = (Interaction(pair=(0, 1), distance_h=1.5, approach_speed=None, facing=None, duration_s=0.0),)
    return SceneDescription(
        window=Window(t_start=0.0, t_end=1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(n_persons=len(persons), crowd_motion=None, motion_area=area, weapon_visible=False),
        persons=persons,
        interactions=pairs if len(persons) > 1 else (),
    )


def _episode(part: str, label: int, scenes: list[SceneDescription]) -> CachedEpisode:
    return CachedEpisode(
        {"episode_id": f"{part}/x", "part": part, "label": label}, scenes, [{} for _ in scenes]
    )


def test_quantiles_come_from_calm_train_windows_only_and_skip_nulls():
    calm = _episode("train", 0, [_scene([float(v), None], float(v)) for v in range(101)])
    fight = _episode("train", 1, [_scene([99.0, 99.0], 99.0)])  # never part of the "usual" values
    stats = reference_stats([calm, fight])
    assert stats["windows"] == 101
    speed = stats["fields"]["persons.speed"]
    assert (speed["p50"], speed["p90"], speed["p99"]) == (50.0, 90.0, 99.0) and speed["known"] == 101
    assert stats["fields"]["summary.motion_area"]["p90"] == pytest.approx(90.0)
    assert "summary.crowd_motion" not in stats["fields"]  # always null here: nothing to describe
    assert (
        stats["shares"]["persons.holds"] == 0.0 and stats["fields"]["interactions.distance_h"]["p50"] == 1.5
    )


def test_a_val_or_test_episode_is_refused():
    with pytest.raises(ValueError, match="train"):
        reference_stats([_episode("val", 0, [_scene([0.1], 1.0)])])
