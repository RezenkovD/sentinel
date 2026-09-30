"""Pixel motion agent (schema v3): frame differencing that does not depend on person detection."""

import numpy as np
import pytest

from sentinel.agents.detector import Track
from sentinel.agents.motion import PixelMotion


def _frame(w: int, h: int, square_at: float | None = None, size: float = 0.2) -> np.ndarray:
    """Dark frame with a bright square whose left edge is at `square_at` (share of the width)."""
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    if square_at is not None:
        x1, side = int(square_at * w), int(size * w)
        y1 = int(0.3 * h)
        img[y1 : y1 + side, x1 : x1 + side] = 220
    return img


def test_first_frame_has_no_motion_and_a_still_scene_has_none():
    motion = PixelMotion(max_width=160, threshold=15)
    assert motion.update(_frame(320, 240), []) is None
    still = motion.update(_frame(320, 240), [])
    assert still.energy == 0.0 and still.area == 0.0


def test_a_moving_object_shows_as_energy_area_and_motion_inside_its_box():
    motion = PixelMotion(max_width=160, threshold=15)
    w, h = 640, 480
    near = Track(1, (0.05 * w, 0.2 * h, 0.5 * w, 0.8 * h), 0.9)  # around the square
    far = Track(2, (0.7 * w, 0.2 * h, 0.95 * w, 0.8 * h), 0.9)
    motion.update(_frame(w, h, square_at=0.10), [near, far])
    m = motion.update(_frame(w, h, square_at=0.20), [near, far])
    assert m.energy > 5.0 and 3.0 < m.area < 15.0  # the square moved half its side: ~2 x 0.2 x 0.1 x 4/3
    assert m.people[1] > 20.0 and m.people[2] == pytest.approx(0.0, abs=0.5)


def test_motion_does_not_depend_on_the_input_resolution():
    energies = []
    for w, h in ((320, 240), (1280, 960), (1920, 1440)):
        motion = PixelMotion(max_width=160, threshold=15)
        motion.update(_frame(w, h, square_at=0.10), [])
        energies.append(motion.update(_frame(w, h, square_at=0.20), []).energy)
    assert max(energies) - min(energies) < 0.15 * max(energies)


def test_reset_forgets_the_previous_frame():
    motion = PixelMotion(max_width=160, threshold=15)
    motion.update(_frame(320, 240, square_at=0.1), [])
    motion.reset()
    assert motion.update(_frame(320, 240, square_at=0.5), []) is None


def test_a_change_of_frame_size_starts_over():
    motion = PixelMotion(max_width=160, threshold=15)
    motion.update(_frame(320, 240), [])
    assert motion.update(_frame(640, 360), []) is None
