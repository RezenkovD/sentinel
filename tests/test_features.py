import numpy as np
import pytest

from sentinel.agents.features import (
    ContactTracker,
    FeatureConfig,
    Observation,
    approach_speed,
    distance_series,
    facing,
    holds,
    limb_accel,
    person_speed,
)
from sentinel.agents.weapons import WeaponBox

CFG = FeatureConfig(
    min_keypoint_conf=0.5,
    hold_margin_h=0.1,
    hold_min_detections=2,
    pair_max_distance_h=2.0,
    max_pairs=5,
    contact_distance_h=1.0,
    facing_min_offset_h=0.03,
)
H = 100.0  # person height in pixels: box 100 px tall


def _kps(wrist=(50.0, 50.0), wrist_conf=0.9, nose=None, shoulders=None) -> np.ndarray:
    k = np.zeros((17, 3))
    k[10] = (*wrist, wrist_conf)  # right wrist
    if nose is not None:
        k[0] = (*nose, 0.9)
    if shoulders is not None:
        k[5] = (*shoulders[0], 0.9)
        k[6] = (*shoulders[1], 0.9)
    return k


def _obs(t: float, x: float = 0.0, keypoints: np.ndarray | None = None, conf: float = 0.8) -> Observation:
    return Observation(t=t, box=(x, 0.0, x + 40.0, H), conf=conf, keypoints=keypoints)


def _wrist_track(times, xs, confs=None) -> list[Observation]:
    confs = confs or [0.9] * len(xs)
    return [
        _obs(t, 0.0, _kps(wrist=(x, 50.0), wrist_conf=c)) for t, x, c in zip(times, xs, confs, strict=True)
    ]


def test_strike_gives_a_sharp_wrist_acceleration():
    # wrist moves 10 px per 0.1 s, then stops dead: a = 2 (0 - 100 px/s) / 0.2 s = -1000 px/s² = 10 heights/s²
    assert limb_accel(_wrist_track([0.0, 0.1, 0.2], [0.0, 10.0, 10.0]), H, CFG) == pytest.approx(10.0)


def test_steady_wrist_motion_has_no_acceleration():
    assert limb_accel(_wrist_track([0.0, 0.1, 0.2], [0.0, 10.0, 20.0]), H, CFG) == pytest.approx(0.0)


def test_acceleration_uses_real_time_stamps_not_the_nominal_rate():
    # equal steps but uneven time: v1 = 100 px/s, v2 = 50 px/s, a = 2 (50 - 100) / 0.3 = -333 px/s²
    track = _wrist_track([0.0, 0.1, 0.3], [0.0, 10.0, 20.0])
    assert limb_accel(track, H, CFG) == pytest.approx(3.333, abs=1e-3)


def test_unconfident_wrist_points_are_not_used():
    track = _wrist_track([0.0, 0.1, 0.2], [0.0, 10.0, 10.0], confs=[0.9, 0.2, 0.9])
    assert limb_accel(track, H, CFG) is None


def test_person_without_skeleton_has_null_limb_features_but_a_speed():
    track = [_obs(0.0, 0.0), _obs(0.1, 10.0), _obs(0.2, 20.0)]  # box centre moves 100 px/s
    assert limb_accel(track, H, CFG) is None
    assert person_speed(track, H) == pytest.approx(1.0)


def test_track_seen_in_one_frame_has_null_speed():
    assert person_speed([_obs(0.5)], H) is None


def test_speed_is_the_median_over_the_window():
    track = [_obs(0.0, 0.0), _obs(0.1, 10.0), _obs(0.2, 20.0), _obs(0.3, 80.0)]  # 1, 1, 6 heights/s
    assert person_speed(track, H) == pytest.approx(1.0)


def _gun_at(x: float, y: float, conf: float = 0.8) -> WeaponBox:
    return WeaponBox("gun", (x - 5.0, y - 5.0, x + 5.0, y + 5.0), conf)


def test_weapon_at_the_wrist_in_two_frames_is_held():
    track = _wrist_track([0.0, 0.1, 0.2], [50.0, 50.0, 50.0])
    weapons = {0.0: [_gun_at(50.0, 50.0, 0.8)], 0.1: [_gun_at(50.0, 50.0, 0.7)], 0.2: []}  # two detections
    assert holds(track, weapons, H, CFG) == "gun"


def test_weapon_at_the_wrist_in_a_single_frame_is_not_held():
    track = _wrist_track([0.0, 0.1, 0.2], [50.0, 50.0, 50.0])
    assert holds(track, {0.0: [_gun_at(50.0, 50.0)]}, H, CFG) is None


def test_weapon_nearby_but_not_in_hand_is_not_held():
    track = _wrist_track([0.0, 0.1, 0.2], [50.0, 50.0, 50.0])
    far = {t: [_gun_at(50.0 + 0.5 * H, 50.0)] for t in (0.0, 0.1, 0.2)}  # half a height away
    assert holds(track, far, H, CFG) is None


def test_hold_margin_is_measured_in_person_heights():
    track = _wrist_track([0.0, 0.1], [50.0, 50.0])
    # box edge 8 px from the wrist; margin 0.1 height = 10 px reaches it
    near = {t: [WeaponBox("knife", (58.0, 45.0, 70.0, 55.0), 0.6 + t)] for t in (0.0, 0.1)}
    assert holds(track, near, H, CFG) == "knife"


def test_two_people_approaching():
    # centres 300 -> 200 -> 100 px apart in 1 s, heights 100 px: 3 -> 2 -> 1 heights
    a = [_obs(t, 0.0) for t in (0.0, 0.5, 1.0)]
    b = [_obs(t, x) for t, x in ((0.0, 300.0), (0.5, 200.0), (1.0, 100.0))]
    series = distance_series(a, b, H, H)
    assert [d for _, d in series] == pytest.approx([3.0, 2.0, 1.0])
    assert approach_speed(series) == pytest.approx(2.0)


def test_moving_apart_gives_negative_approach_speed():
    series = [(0.0, 1.0), (0.5, 1.5)]
    assert approach_speed(series) == pytest.approx(-1.0)


def test_single_common_frame_gives_null_approach_speed():
    assert approach_speed([(0.0, 1.0)]) is None


def _facing_person(x: float, nose_dx: float) -> Observation:
    shoulders = ((x + 10.0, 20.0), (x + 30.0, 20.0))  # shoulder midpoint at x + 20
    return _obs(0.0, x, _kps(nose=(x + 20.0 + nose_dx, 10.0), shoulders=shoulders))


def test_people_turned_to_each_other_are_facing():
    a = [_facing_person(0.0, +10.0)]  # nose to the right, B is on the right
    b = [_facing_person(100.0, -10.0)]  # nose to the left, A is on the left
    assert facing(a, b, H, H, CFG) is True


def test_person_turned_away_is_not_facing():
    a = [_facing_person(0.0, +10.0)]
    b = [_facing_person(100.0, +10.0)]  # B looks away from A
    assert facing(a, b, H, H, CFG) is False


def test_facing_is_null_without_nose_and_shoulders():
    a = [_obs(0.0, 0.0, _kps())]  # only a wrist
    b = [_facing_person(100.0, -10.0)]
    assert facing(a, b, H, H, CFG) is None


def test_contact_duration_carries_over_the_window_boundary_and_resets_on_separation():
    tracker = ContactTracker(contact_distance_h=1.0)
    pair = (1, 2)
    times = [round(0.1 * i, 1) for i in range(30)]
    near = {t for t in times if 0.5 <= t <= 1.8}
    windows = []
    for t in times:
        tracker.update(t, {pair: 0.5 if t in near else 1.5})
        if round(t + 0.1, 1) in (1.0, 2.0, 3.0):  # window ends
            windows.append(tracker.window_durations().get(pair, 0.0))
    assert windows == pytest.approx([0.4, 1.3, 0.0])


def test_one_detection_kept_over_frames_by_the_every_nth_schedule_counts_once():
    # the weapon detector runs every 2nd frame and keeps its last boxes in between: the same box twice
    track = _wrist_track([0.0, 0.1, 0.2, 0.3], [50.0, 50.0, 50.0, 50.0])
    same = _gun_at(50.0, 50.0)
    assert holds(track, {0.0: [same], 0.1: [same]}, H, CFG) is None
    fresh = WeaponBox("gun", (46.0, 45.0, 56.0, 55.0), 0.7)  # a new detection two frames later
    assert holds(track, {0.0: [same], 0.1: [same], 0.2: [fresh], 0.3: [fresh]}, H, CFG) == "gun"
