"""Raw wrist acceleration in the frame (the v2 feature) for figure 2.2 — on synthetic keypoints."""

import numpy as np
import pytest

from sentinel.agents.detector import Track
from sentinel.eval.raw_wrist import raw_wrist_accel, window_value


def _kp(x: float, y: float, conf: float = 0.9) -> np.ndarray:
    k = np.zeros((17, 3))
    k[9] = (x, y, conf)  # left wrist
    return k


def test_constant_acceleration_in_pixels_divided_by_the_median_height():
    # x = 50 t² px -> acceleration 100 px/s²; box height 100 px -> 1 height/s²
    obs = [(t, (0.0, 0.0, 10.0, 100.0), _kp(50 * t * t, 0.0)) for t in (0.0, 0.1, 0.2, 0.3)]
    assert raw_wrist_accel(obs, 0.5) == pytest.approx(1.0)


def test_unconfident_wrists_and_short_tracks_give_nothing():
    obs = [(t, (0.0, 0.0, 10.0, 100.0), _kp(t, 0.0, conf=0.2)) for t in (0.0, 0.1, 0.2)]
    assert raw_wrist_accel(obs, 0.5) is None
    assert raw_wrist_accel(obs[:2], 0.0) is None and raw_wrist_accel([], 0.5) is None


def test_window_value_is_the_largest_over_people():
    calm = [(t, [Track(1, (0, 0, 10, 100), 0.9)], {1: _kp(10.0, 0.0)}) for t in (0.0, 0.1, 0.2)]
    moving = [
        (
            t,
            [Track(1, (0, 0, 10, 100), 0.9), Track(2, (0, 0, 10, 100), 0.9)],
            {1: _kp(10, 0), 2: _kp(50 * t * t, 0)},
        )
        for t in (0.0, 0.1, 0.2)
    ]
    assert window_value(calm, 0.5) == pytest.approx(0.0)
    assert window_value(moving, 0.5) == pytest.approx(1.0)
    assert window_value([(0.0, [], {})], 0.5) is None
