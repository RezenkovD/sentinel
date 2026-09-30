"""Smoothed keypoint kinematics relative to the torso (schema v3): limb_speed, limb_accel, pose_motion."""

import numpy as np
import pytest

from sentinel.agents.features import FeatureConfig, Observation, pose_kinematics

CFG = FeatureConfig(
    min_keypoint_conf=0.5,
    hold_margin_h=0.1,
    hold_min_share=0.4,
    weapon_max_height_h=0.5,
    pair_max_distance_h=2.0,
    max_pairs=5,
    contact_distance_h=1.0,
    facing_min_offset_h=0.03,
    smooth_points=5,
    max_gap_s=0.3,
)
H = 100.0
L_SHOULDER, R_SHOULDER, R_WRIST, L_HIP = 5, 6, 10, 11


def _skeleton(shift_x: float, wrist_rel: tuple[float, float], extra: dict | None = None) -> np.ndarray:
    """Shoulders at (40, 30) and (60, 30) moved by shift_x; right wrist relative to the shoulder midpoint."""
    k = np.zeros((17, 3))
    k[L_SHOULDER] = (40.0 + shift_x, 30.0, 0.9)
    k[R_SHOULDER] = (60.0 + shift_x, 30.0, 0.9)
    k[R_WRIST] = (50.0 + shift_x + wrist_rel[0], 30.0 + wrist_rel[1], 0.9)
    k[L_HIP] = (45.0 + shift_x, 70.0, 0.9)
    for index, point in (extra or {}).items():
        k[index] = (*point, 0.9)
    return k


def _obs(times, shifts, wrists) -> list[Observation]:
    return [
        Observation(t=t, box=(s, 0.0, s + 40.0, H), conf=0.9, keypoints=_skeleton(s, w))
        for t, s, w in zip(times, shifts, wrists, strict=True)
    ]


TIMES = [round(0.1 * i, 1) for i in range(10)]


def test_a_punch_moves_the_wrist_fast_relative_to_the_body():
    wrists = [(10.0 + (60.0 if 3 <= i <= 5 else 0.0), 20.0) for i in range(10)]  # out and back in 0.3 s
    k = pose_kinematics(_obs(TIMES, [0.0] * 10, wrists), H, CFG, t_min=0.0)
    assert k.limb_speed > 1.5  # heights/s
    assert k.limb_accel > 20.0


def test_running_moves_the_whole_body_but_not_the_limbs_relative_to_it():
    shifts = [30.0 * i for i in range(10)]  # 3 heights/s across the frame, arms still relative to the torso
    k = pose_kinematics(_obs(TIMES, shifts, [(10.0, 20.0)] * 10), H, CFG, t_min=0.0)
    assert k.limb_speed == pytest.approx(0.0, abs=1e-9)
    assert k.limb_accel == pytest.approx(0.0, abs=1e-9)
    assert k.pose_motion == pytest.approx(0.0, abs=1e-9)


def test_smoothing_suppresses_keypoint_jitter():
    rng = np.random.default_rng(0)
    wrists = [(10.0 + rng.normal(0, 2.0), 20.0 + rng.normal(0, 2.0)) for _ in range(10)]  # ±2 px of jitter
    obs = _obs(TIMES, [0.0] * 10, wrists)
    smooth = pose_kinematics(obs, H, CFG, t_min=0.0).limb_accel
    raw = max(
        np.linalg.norm((np.array(wrists[i + 1]) - 2 * np.array(wrists[i]) + np.array(wrists[i - 1])) / 0.1**2)
        / H
        for i in range(1, 9)
    )
    assert smooth < 0.35 * raw


def test_speed_uses_real_time_stamps():
    times = [0.0, 0.1, 0.25, 0.3, 0.45, 0.5]  # uneven steps, none longer than the gap limit
    wrists = [(10.0 + 100.0 * t, 20.0) for t in times]  # 1 height/s relative to the shoulders
    k = pose_kinematics(_obs(times, [0.0] * 6, wrists), H, CFG, t_min=0.0)
    assert k.limb_speed == pytest.approx(1.0, rel=1e-6)
    assert k.limb_accel == pytest.approx(0.0, abs=1e-6)


def test_no_interpolation_across_gaps_longer_than_the_limit():
    times = [0.0, 0.1, 0.2, 0.7, 0.8, 0.9]  # 0.5 s without the person
    wrists = [(10.0, 20.0)] * 3 + [(60.0, 20.0)] * 3  # the arm moved while we did not see it
    k = pose_kinematics(_obs(times, [0.0] * 6, wrists), H, CFG, t_min=0.0)
    assert k.limb_speed == pytest.approx(0.0, abs=1e-9)


def test_only_points_of_this_window_are_summarised_but_earlier_ones_help_the_fit():
    times = [round(0.1 * i, 1) for i in range(15)]  # 0.5 s of the previous window as context
    wrists = [(10.0 + (60.0 if i in (2, 3) else 0.0), 20.0) for i in range(15)]  # a punch before t_min
    k = pose_kinematics(_obs(times, [0.0] * 15, wrists), H, CFG, t_min=1.0)
    assert k.limb_speed == pytest.approx(0.0, abs=1e-9)


def test_unconfident_wrist_points_are_not_used():
    obs = _obs(TIMES, [0.0] * 10, [(10.0 + 10.0 * i, 20.0) for i in range(10)])
    for o in obs:
        o.keypoints[R_WRIST, 2] = 0.2
    k = pose_kinematics(obs, H, CFG, t_min=0.0)
    assert k.limb_speed is None and k.limb_accel is None


def test_limb_features_are_null_without_shoulders():
    obs = _obs(TIMES, [0.0] * 10, [(10.0, 20.0)] * 10)
    for o in obs:
        o.keypoints[[L_SHOULDER, R_SHOULDER], 2] = 0.1
    k = pose_kinematics(obs, H, CFG, t_min=0.0)
    assert k.limb_speed is None and k.limb_accel is None and k.pose_motion is None


def test_pose_motion_is_the_mean_keypoint_speed_relative_to_the_shoulders():
    # the hip swings 0.5 heights/s relative to the shoulders, the wrist stays: mean of the two speeds
    obs = []
    for t in TIMES:
        k = _skeleton(0.0, (10.0, 20.0))
        k[L_HIP] = (45.0 + 50.0 * t, 70.0, 0.9)
        obs.append(Observation(t=t, box=(0.0, 0.0, 40.0, H), conf=0.9, keypoints=k))
    motion = pose_kinematics(obs, H, CFG, t_min=0.0).pose_motion
    assert motion == pytest.approx(0.25, rel=1e-6)
