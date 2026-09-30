from datetime import datetime

import numpy as np
import pytest

from sentinel.agents.context import ZoneContext, ZoneProfile
from sentinel.agents.detector import Track
from sentinel.agents.features import FeatureConfig
from sentinel.agents.motion import FrameMotion
from sentinel.agents.weapons import WeaponBox
from sentinel.scene import SceneBuilder
from sentinel.schema import Zone

CFG = FeatureConfig(
    min_keypoint_conf=0.5,
    hold_margin_h=0.1,
    hold_min_share=0.4,
    weapon_max_height_h=0.5,
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


def _run(builder: SceneBuilder, frames: list[tuple]) -> list:
    """Frames are (t, tracks, skeletons, weapons[, weapon detector ran in this frame])."""
    scenes = [s for f in frames if (s := builder.add(*f)) is not None]
    if (last := builder.flush()) is not None:
        scenes.append(last)
    return scenes


def _builder() -> SceneBuilder:
    return SceneBuilder(CFG, window_s=1.0, source="cam-1", zone=ZONE)


def _upper_body(wrist_dx: float) -> np.ndarray:
    k = np.zeros((17, 3))
    k[5], k[6] = (10.0, 30.0, 0.9), (30.0, 30.0, 0.9)  # shoulders
    k[10] = (20.0 + wrist_dx, 50.0, 0.9)  # right wrist, relative to the shoulder midpoint
    return k


def test_limb_motion_is_measured_relative_to_the_torso():
    frames = [
        (round(0.1 * i, 1), [_track(1, 0.0)], {1: _upper_body(40.0 if i in (4, 5) else 0.0)}, [])
        for i in range(10)
    ]
    person = _run(_builder(), frames)[0].persons[0]
    assert person.limb_speed > 1.0 and person.limb_accel > 10.0 and person.pose_motion > 0.0


def test_the_last_frames_of_the_previous_window_help_smooth_the_next_one():
    # the arm moves just before the window boundary; the next window sees it through its first fits
    frames = [
        (round(0.1 * i, 1), [_track(1, 0.0)], {1: _upper_body(60.0 if i >= 9 else 0.0)}, [])
        for i in range(20)
    ]
    first, second = _run(_builder(), frames)
    assert second.persons[0].limb_accel > 5.0  # the fit at the first frame reaches back to the move


def test_pixel_motion_is_averaged_over_the_window():
    frames = []
    for i in range(10):
        motion = (
            None if i == 0 else FrameMotion(energy=4.0 + (i % 2) * 2, area=1.0, people={1: 10.0 * (i % 2)})
        )
        frames.append((round(0.1 * i, 1), [_track(1, 0.0)], {}, [], True, motion))
    scene = _run(_builder(), frames)[0]
    assert scene.summary.motion_energy == pytest.approx(5.0, abs=0.12)  # 9 frames with a previous one
    assert scene.summary.motion_area == pytest.approx(1.0)
    assert scene.persons[0].motion == pytest.approx(5.0, abs=0.6)


def test_without_pixel_motion_the_features_are_null():
    frames = [(round(0.1 * i, 1), [_track(1, 0.0)], {}, []) for i in range(10)]
    scene = _run(_builder(), frames)[0]
    assert scene.summary.motion_energy is None and scene.persons[0].motion is None


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


def test_a_detection_kept_between_detector_runs_counts_once():
    # the detector runs on even frames and its last boxes are repeated on odd ones
    gun = WeaponBox("gun", (45.0, 45.0, 55.0, 55.0), 0.8)

    def frames(detected_in_runs: set[int]) -> list[tuple]:
        out, boxes = [], []
        for i in range(10):
            ran = i % 2 == 0
            if ran:
                boxes = [gun] if i in detected_in_runs else []
            out.append((round(0.1 * i, 1), [_track(1, 0.0)], {1: _wrist_skeleton(50.0, 50.0)}, boxes, ran))
        return out

    assert _run(_builder(), frames({0}))[0].persons[0].holds is None  # 1 of 5 runs, shown in 2 frames
    assert _run(_builder(), frames({0, 2}))[0].persons[0].holds == "gun"  # 2 of 5 runs


def test_weapon_box_taller_than_half_the_tallest_person_is_not_in_view():
    def scene_with(box: tuple[float, float, float, float], tracks: list) -> bool:
        frames = [(round(0.1 * i, 1), tracks, {}, [WeaponBox("gun", box, 0.8)]) for i in range(10)]
        return _run(_builder(), frames)[0].summary.weapon_visible

    person = [_track(1, 0.0, h=100.0)]
    assert scene_with((200.0, 0.0, 240.0, 60.0), person) is False  # 0.6 of the person's height
    assert scene_with((200.0, 0.0, 240.0, 30.0), person) is True
    assert scene_with((0.0, 0.0, 500.0, 400.0), []) is True  # nobody to compare with: kept


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
