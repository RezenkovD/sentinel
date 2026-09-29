from pathlib import Path

import pytest

from sentinel.config import IncidentsConfig
from sentinel.incidents import IncidentManager, State
from sentinel.schema import Interaction, Person, RiskResult, SceneDescription, Summary, Window, Zone

BASE = IncidentsConfig(
    theta_high=0.6,
    theta_low=0.4,
    k=3,
    n=5,
    quiet_s=10.0,
    merge_after_close_s=30.0,
    merge_participants_max_s=None,
    max_new_per_minute=6,
    update_interval_s=10.0,
    weapon_override_margin=0.1,
    blur_faces=True,
    output_dir=Path("results/incidents"),
)


class FakeKeyframes:
    def __init__(self) -> None:
        self.saved: list[tuple[str, str, object]] = []

    def __call__(self, incident_id: str, name: str, snapshot: object) -> Path:
        self.saved.append((incident_id, name, snapshot))
        return Path(f"/fake/{incident_id}/{name}.jpg")


def _scene(t: float, ids: tuple[int, ...] = (1, 2), holds: str | None = None) -> SceneDescription:
    persons = tuple(
        Person(id=i, speed=0.5, limb_accel=5.0, holds=holds if i == ids[0] else None, conf=0.8) for i in ids
    )
    pairs = (Interaction(pair=ids[:2], distance_h=0.5, approach_speed=0.5, facing=True, duration_s=1.0),)
    return SceneDescription(
        window=Window(t_start=t - 1.0, t_end=t, source="cam-1"),
        zone=Zone(type="street", hours="day", expected_density="medium"),
        summary=Summary(n_persons=len(ids), crowd_motion=0.5, weapon_visible=holds is not None),
        persons=persons,
        interactions=pairs if len(ids) > 1 else (),
    )


def _risk(r: float) -> RiskResult:
    level = "alarm" if r >= 0.6 else "attention" if r >= 0.4 else "normal"
    return RiskResult(risk=r, level=level, reason_uk=f"ризик {r}")


def _feed(manager: IncidentManager, risks: list[float], start: float = 1.0, ids=(1, 2), holds_at=()) -> list:
    events = []
    for i, r in enumerate(risks):
        t = start + i
        holds = "gun" if i in holds_at else None
        events += manager.on_window(t, _risk(r), _scene(t, ids, holds), snapshot=f"frame@{t}")
    return events


def _manager(**overrides) -> tuple[IncidentManager, FakeKeyframes]:
    keyframes = FakeKeyframes()
    return IncidentManager(BASE.model_copy(update=overrides), "cam-1", keyframes, run_id="r"), keyframes


def _kinds(events) -> list[str]:
    return [e.kind for e in events]


def test_risk_hovering_around_the_threshold_opens_one_incident_only():
    manager, _ = _manager()
    events = _feed(manager, [0.65, 0.45] * 15)
    assert _kinds(events).count("open") == 1 and "close" not in _kinds(events)


def test_single_high_window_only_raises_attention():
    manager, _ = _manager()
    assert _feed(manager, [0.7, 0.3, 0.3]) == []
    assert manager.state is State.NORMAL


def test_k_of_n_high_windows_open_an_alarm():
    manager, _ = _manager()
    events = _feed(manager, [0.7, 0.5, 0.7, 0.5, 0.7])
    assert _kinds(events) == ["open"]
    assert events[0].t == pytest.approx(5.0)  # the fifth window completes 3 of 5
    assert manager.state is State.ALARM


def test_weapon_in_hand_opens_an_alarm_at_once():
    manager, keyframes = _manager()
    events = _feed(manager, [0.7], holds_at={0})
    assert _kinds(events) == ["open"] and events[0].level == "alarm"
    assert keyframes.saved[0][1] == "keyframe"


def test_keyframe_is_the_highest_risk_window_before_opening():
    manager, keyframes = _manager()
    _feed(manager, [0.7, 0.9, 0.65])
    assert keyframes.saved == [(manager.last_incident_id, "keyframe", "frame@2.0")]


def test_incident_closes_after_m_seconds_of_calm():
    manager, _ = _manager()
    events = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 12)
    closes = [e for e in events if e.kind == "close"]
    assert len(closes) == 1
    assert closes[0].t == pytest.approx(4.0 + 10.0)  # calm since t = 4, closed 10 s later
    assert manager.state is State.NORMAL


def test_calm_timer_restarts_when_risk_comes_back_between_thresholds():
    manager, _ = _manager()
    events = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 5 + [0.5] + [0.2] * 9)
    assert "close" not in _kinds(events)


def test_new_high_risk_soon_after_closing_joins_the_old_incident():
    manager, _ = _manager()
    first = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 11)
    first_id = first[0].incident_id
    again = _feed(manager, [0.7, 0.7, 0.7], start=20.0, ids=(5, 6))  # other people, 6 s after closing
    assert "open" not in _kinds(again)
    assert again[0].incident_id == first_id and again[0].merged


def test_same_people_join_the_old_incident_even_later():
    manager, _ = _manager()
    first = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 11)
    again = _feed(manager, [0.7, 0.7, 0.7], start=100.0, ids=(2, 9))  # track 2 was a participant
    assert again[0].incident_id == first[0].incident_id and again[0].merged


def test_participant_merge_can_be_limited_in_time():
    manager, _ = _manager(merge_participants_max_s=60.0)
    first = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 11)
    again = _feed(manager, [0.7, 0.7, 0.7], start=100.0, ids=(2, 9))
    assert _kinds(again)[0] == "open" and again[0].incident_id != first[0].incident_id


def test_other_people_long_after_closing_open_a_new_incident():
    manager, _ = _manager()
    first = _feed(manager, [0.7, 0.7, 0.7] + [0.2] * 11)
    again = _feed(manager, [0.7, 0.7, 0.7], start=100.0, ids=(7, 8))
    assert _kinds(again) == ["open"] and again[0].incident_id != first[0].incident_id


def test_weapon_appearing_during_a_fight_escalates_with_a_new_keyframe():
    manager, keyframes = _manager()
    events = _feed(manager, [0.7, 0.7, 0.7, 0.8, 0.8], holds_at={4})
    assert _kinds(events) == ["open", "escalate"]
    assert [name for _, name, _ in keyframes.saved] == ["keyframe", "escalate_1"]


def test_updates_come_at_most_every_update_interval():
    manager, _ = _manager()
    events = _feed(manager, [0.7] * 25)
    updates = [e.t for e in events if e.kind == "update"]
    assert updates == [13.0, 23.0]  # opened at t = 3


def test_too_many_new_incidents_per_minute_are_folded_into_a_summary():
    manager, _ = _manager(quiet_s=1.0, merge_after_close_s=0.0, max_new_per_minute=2)
    events = []
    for n, start in enumerate([1.0, 10.0, 20.0]):
        events += _feed(manager, [0.7, 0.7, 0.7, 0.2, 0.2, 0.2], start=start, ids=(10 * n + 1, 10 * n + 2))
    assert _kinds(events).count("open") == 2
    summaries = [e for e in events if e.kind == "summary"]
    assert len(summaries) == 1 and summaries[0].suppressed == 1


def test_keyframe_comes_from_the_last_n_windows_not_a_stale_one():
    # a long ATTENTION spell must not keep every frame, nor pick an hour-old one when the alarm opens
    manager, keyframes = _manager()
    _feed(manager, [0.95] + [0.5] * 100 + [0.7, 0.8, 0.7])
    assert keyframes.saved[0][2] == "frame@103.0"  # the 0.8 window among the last n = 5


def test_end_of_stream_closes_the_open_incident():
    manager, _ = _manager()
    _feed(manager, [0.7, 0.7, 0.7, 0.8])
    events = manager.finish(t=4.5)
    assert [e.kind for e in events] == ["close"] and events[0].stream_ended
    assert manager.state is State.NORMAL and manager.finish(t=5.0) == []


def test_thresholds_must_leave_a_hysteresis_gap():
    import pytest as _pytest
    from pydantic import ValidationError

    with _pytest.raises(ValidationError):
        BASE.model_validate(BASE.model_dump() | {"theta_low": 0.7})
    with _pytest.raises(ValidationError):
        BASE.model_validate(BASE.model_dump() | {"k": 6})
