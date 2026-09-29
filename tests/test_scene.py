from datetime import datetime

import numpy as np
import pytest

from sentinel.agents.context import ZoneContext, ZoneProfile
from sentinel.agents.detector import Track
from sentinel.agents.features import FeatureConfig
from sentinel.agents.weapons import WeaponBox
from sentinel.scene import SceneBuilder
from sentinel.schema import Zone

CFG = FeatureConfig(
    min_keypoint_conf=0.5,
    hold_margin_h=0.1,
    hold_min_detections=2,
    pair_max_distance_h=2.0,
    max_pairs=5,
    contact_distance_h=1.0,
    facing_min_offset_h=0.03,
)
ZONE = Zone(type="street", hours="day", expected_density="medium")


def _track(tid: int, x: float, h: float = 100.0) -> Track:
    return Track(track_id=tid, box=(x, 0.0, x + 40.0, h), conf=0.8)


def _wrist_skeleton(x: float, y: float) -> np.ndarray:
    k = np.zeros((17, 3))
    k[10] = (x, y, 0.9)
    return k


def _run(builder: SceneBuilder, frames: list[tuple[float, list, dict, list]]) -> list:
    scenes = [s for t, tr, sk, w in frames if (s := builder.add(t, tr, sk, w)) is not None]
    if (last := builder.flush()) is not None:
        scenes.append(last)
    return scenes


def _builder() -> SceneBuilder:
    return SceneBuilder(CFG, window_s=1.0, source="cam-1", zone=ZONE)


def test_one_description_per_window_with_window_bounds():
    frames = [(round(0.1 * i, 1), [_track(1, 10.0 * i)], {}, []) for i in range(25)]  # 2.5 s at 10 fps
    scenes = _run(_builder(), frames)
    assert [(s.window.t_start, s.window.t_end) for s in scenes] == [(0.0, 1.0), (1.0, 2.0), (2.0, 2.4)]
    assert all(s.window.source == "cam-1" and s.zone == ZONE for s in scenes)


def test_window_without_people_is_a_valid_empty_scene():
    scenes = _run(_builder(), [(round(0.1 * i, 1), [], {}, []) for i in range(10)])
    s = scenes[0]
    assert (s.summary.n_persons, s.summary.crowd_motion, s.persons, s.interactions) == (0, 0.0, (), ())


def test_person_without_skeleton_gets_null_limb_features():
    frames = [(round(0.1 * i, 1), [_track(1, 10.0 * i)], {}, []) for i in range(10)]
    person = _run(_builder(), frames)[0].persons[0]
    assert person.limb_accel is None and person.speed == pytest.approx(1.0) and person.holds is None


def test_weapon_in_hand_versus_weapon_in_view():
    def gun_in_hand(i: int) -> WeaponBox:  # a fresh detection in every frame
        return WeaponBox("gun", (45.0, 45.0, 55.0, 55.0), 0.8 - 0.01 * i)

    gun_far = WeaponBox("gun", (400.0, 45.0, 410.0, 55.0), 0.8)
    held = [
        (round(0.1 * i, 1), [_track(1, 0.0)], {1: _wrist_skeleton(50.0, 50.0)}, [gun_in_hand(i)])
        for i in range(10)
    ]
    seen = [
        (round(0.1 * i, 1), [_track(1, 0.0)], {1: _wrist_skeleton(50.0, 50.0)}, [gun_far]) for i in range(10)
    ]
    s_held, s_seen = _run(_builder(), held)[0], _run(_builder(), seen)[0]
    assert s_held.persons[0].holds == "gun" and s_held.summary.weapon_visible
    assert s_seen.persons[0].holds is None and s_seen.summary.weapon_visible


def test_only_close_pairs_are_described_nearest_first_at_most_five():
    # 8 people in a row 60 px apart (heights 100 px): neighbours 0.6 heights, 3 apart 1.8, 4 apart 2.4
    tracks = [_track(i, 60.0 * i) for i in range(8)]
    scene = _run(_builder(), [(round(0.1 * k, 1), tracks, {}, []) for k in range(10)])[0]
    assert len(scene.interactions) == 5
    assert all(i.distance_h < 2.0 for i in scene.interactions)
    assert [i.distance_h for i in scene.interactions] == sorted(i.distance_h for i in scene.interactions)
    assert scene.interactions[0].distance_h == pytest.approx(0.6)


def test_contact_duration_spans_windows():
    # two people 0.5 heights apart for 2 s: the second window reports contact since t = 0
    frames = [(round(0.1 * i, 1), [_track(1, 0.0), _track(2, 50.0)], {}, []) for i in range(20)]
    first, second = _run(_builder(), frames)
    assert first.interactions[0].duration_s == pytest.approx(0.9)
    assert second.interactions[0].duration_s == pytest.approx(1.9)


def test_zone_of_a_source_falls_back_to_default_and_can_follow_the_clock():
    zones = ZoneContext(
        zones={
            "default": ZoneProfile(type="public_space", hours="day", expected_density="medium"),
            "stop": ZoneProfile(
                type="transport_stop", hours="auto", expected_density="low", day_hours=(7, 22)
            ),
        },
        sources={"cam-2": "stop"},
    )
    assert zones.zone_for("cam-9").type == "public_space"
    assert zones.zone_for("cam-2", now=datetime(2026, 10, 1, 23, 30)).hours == "night"
    assert zones.zone_for("cam-2", now=datetime(2026, 10, 1, 12, 0)).hours == "day"
