"""The weapon confirmation rule evaluated on cached episodes (session 03, step 5)."""

import pytest

from sentinel.eval.cache import CachedEpisode
from sentinel.eval.weapon_rule import alarm_share, segment_recall, weapon_alarms
from sentinel.schema import Person, SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(t: float, holders: tuple[int, ...]) -> SceneDescription:
    persons = tuple(
        Person(id=i, speed=0.5, limb_accel=None, holds="gun" if i in holders else None, conf=0.9)
        for i in (1, 2)
    )
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(n_persons=2, crowd_motion=0.5, weapon_visible=bool(holders)),
        persons=persons,
    )


def _episode(
    holders: list[tuple[int, ...]], labels: list[int] | None = None, label: int = 1
) -> CachedEpisode:
    scenes = [_scene(float(t), h) for t, h in enumerate(holders)]
    aux = [{"label": None if labels is None else labels[t]} for t in range(len(holders))]
    return CachedEpisode({"episode_id": "e", "label": label, "part": "select"}, scenes, aux)


def test_alarm_needs_the_same_holder_in_consecutive_windows():
    scenes = _episode([(1,), (1,), (), (2,), (1,), (1,)]).scenes
    assert weapon_alarms(scenes, windows=2) == [False, True, False, False, False, True]
    assert weapon_alarms(scenes, windows=1) == [True, True, False, True, True, True]


def test_segment_recall_and_time_to_alarm():
    hit = _episode([(), (1,), (1,), (1,)], labels=[0, 1, 1, 1])  # weapon from t = 1, alarm at the end of 2
    missed = _episode([(), (1,), (), ()], labels=[0, 1, 1, 0])
    stats = segment_recall([hit, missed], windows=2)
    assert stats["segments"] == 2 and stats["recall"] == pytest.approx(0.5)
    # from the start of the first weapon window to the end of the alarm window
    assert stats["times_to_alarm_s"] == [pytest.approx(2.0)]
    assert stats["alarms_before_weapon"] == 0


def test_share_of_normal_clips_with_a_false_alarm():
    calm = [_episode([(), (), ()], label=0) for _ in range(3)]
    noisy = _episode([(2,), (2,), ()], label=0)
    assert alarm_share(calm + [noisy], windows=2) == pytest.approx(0.25)
    assert alarm_share(calm + [_episode([(2,), (), ()], label=0)], windows=2) == 0.0


def test_a_higher_detector_threshold_drops_the_holds_scored_below_it():
    """Decision 30.09 (report 03, item 4): the cache's aux keeps each holder's threshold, so one extraction
    answers every higher detector threshold."""
    from sentinel.eval.weapon_rule import at_threshold

    episode = _episode([(1,), (1,), (1, 2)], labels=[1, 1, 1])
    levels = [{"1": 0.9}, {"1": 0.55}, {"1": 0.8, "2": 0.45}]
    episode = CachedEpisode(
        episode.meta,
        episode.scenes,
        [a | {"hold_confs": h} for a, h in zip(episode.aux, levels, strict=True)],
    )
    holders = [[p.id for p in s.persons if p.holds] for s in at_threshold(episode, 0.5).scenes]
    assert holders == [[1], [1], [1]]
    holders = [[p.id for p in s.persons if p.holds] for s in at_threshold(episode, 0.6).scenes]
    assert holders == [[1], [], [1]]
    assert weapon_alarms(at_threshold(episode, 0.6).scenes, windows=2) == [False, False, False]


def test_a_cache_without_hold_thresholds_cannot_be_swept():
    from sentinel.eval.weapon_rule import at_threshold

    with pytest.raises(KeyError, match="hold_confs"):
        at_threshold(_episode([(1,)], labels=[1]), 0.5)
