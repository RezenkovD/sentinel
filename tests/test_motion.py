"""Pixel motion agent (schema v3, perception 3.1): frame differencing independent of person detection,
normalised to the nominal analysis interval."""

import numpy as np
import pytest

from sentinel.agents.detector import Track
from sentinel.agents.motion import PixelMotion, interval_scale


def _frame(w: int, h: int, square_at: float | None = None, size: float = 0.2) -> np.ndarray:
    """Dark frame with a bright square whose left edge is at `square_at` (share of the width)."""
    img = np.full((h, w, 3), 20, dtype=np.uint8)
    if square_at is not None:
        x1, side = int(square_at * w), int(size * w)
        y1 = int(0.3 * h)
        img[y1 : y1 + side, x1 : x1 + side] = 220
    return img


def test_first_frame_has_no_motion_and_a_still_scene_has_none():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    assert motion.update(_frame(320, 240), [], 0.0) is None
    still = motion.update(_frame(320, 240), [], 0.1)
    assert still.energy == 0.0 and still.area == 0.0


def test_a_moving_object_shows_as_energy_area_and_motion_inside_its_box():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    w, h = 640, 480
    near = Track(1, (0.05 * w, 0.2 * h, 0.5 * w, 0.8 * h), 0.9)  # around the square
    far = Track(2, (0.7 * w, 0.2 * h, 0.95 * w, 0.8 * h), 0.9)
    motion.update(_frame(w, h, square_at=0.10), [near, far], 0.0)
    m = motion.update(_frame(w, h, square_at=0.20), [near, far], 0.1)
    assert m.energy > 5.0 and 3.0 < m.area < 15.0  # the square moved half its side: ~2 x 0.2 x 0.1 x 4/3
    assert m.people[1] > 20.0 and m.people[2] == pytest.approx(0.0, abs=0.5)


def test_motion_does_not_depend_on_the_input_resolution():
    energies = []
    for w, h in ((320, 240), (1280, 960), (1920, 1440)):
        motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
        motion.update(_frame(w, h, square_at=0.10), [], 0.0)
        energies.append(motion.update(_frame(w, h, square_at=0.20), [], 0.1).energy)
    assert max(energies) - min(energies) < 0.15 * max(energies)


def test_reset_forgets_the_previous_frame():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    motion.update(_frame(320, 240, square_at=0.1), [], 0.0)
    motion.reset()
    assert motion.update(_frame(320, 240, square_at=0.5), [], 0.1) is None


def test_a_change_of_frame_size_starts_over():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    motion.update(_frame(320, 240), [], 0.0)
    assert motion.update(_frame(640, 360), [], 0.1) is None


def _shifted(level: int) -> np.ndarray:
    """Uniform grey frame: a change of brightness everywhere, like a moving crowd filling the view."""
    return np.full((240, 320, 3), level, dtype=np.uint8)


def test_motion_over_a_longer_gap_is_scaled_down_to_the_nominal_interval():
    """Live / paced runs drop frames: a change collected over two intervals counts as half per interval."""
    on_time = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    on_time.update(_shifted(100), [], 0.0)
    nominal = on_time.update(_shifted(120), [], 0.1)
    late = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    late.update(_shifted(100), [], 0.0)
    delayed = late.update(_shifted(120), [], 0.2)
    assert nominal.energy == pytest.approx(20.0) and delayed.energy == pytest.approx(10.0)
    # 20 grey levels in one interval is motion (> 15); spread over two intervals it is 10 per interval
    assert nominal.area == pytest.approx(100.0) and delayed.area == pytest.approx(0.0)


def test_a_shorter_gap_is_scaled_up():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    motion.update(_shifted(100), [], 0.0)
    assert motion.update(_shifted(110), [], 0.05).energy == pytest.approx(20.0)


def test_frames_on_the_nominal_grid_keep_their_values_exactly():
    """Offline caches (perception 3.0) were made on the nominal grid: 3.1 must give the very same numbers."""
    assert interval_scale(3 / 30 - 0 / 30, 0.1) == 1.0  # RWF: 30 frames/s, every third frame
    assert interval_scale(0.7 - 0.6, 0.1) == 1.0  # float noise in frame times
    assert interval_scale(0.5, 0.5) == 1.0  # Sevilla stills: 2 frames/s
    assert interval_scale(3 / 24, 0.1) == pytest.approx(0.8)  # XD at 24 frames/s: the grid alternates…
    assert interval_scale(2 / 24, 0.1) == pytest.approx(1.2, rel=0.01)  # …3 and 2 frames (Δt in ms)


def test_a_frame_that_is_not_later_than_the_previous_one_starts_over():
    motion = PixelMotion(max_width=160, threshold=15, interval_s=0.1)
    motion.update(_shifted(100), [], 1.0)
    assert motion.update(_shifted(120), [], 1.0) is None
