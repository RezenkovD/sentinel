"""Pixel motion agent: frame differencing on small grey frames, independent of person detection (schema v3).

It keeps a signal where people are hard to detect (dark or 320x240 clips). For consecutive analysed frames:
`energy` — the mean absolute difference over the frame, in grey levels (0-255); `area` — the percentage of
pixels that changed by more than `threshold` grey levels; per tracked person — the mean difference inside
their box. Frames are shrunk to at most `max_width` pixels (nearest-neighbour to 4x that, then area-averaged),
so the cost stays well below a millisecond and the values do not depend on the input resolution.

Perception 3.1 (decision 30.09, report 03): the change between two analysed frames is normalised to the
nominal analysis interval 1/fps — live and paced runs drop frames, and a change collected over two intervals
would otherwise look twice as strong as in the offline cache. Energy and per-person motion are multiplied by
interval / Δt, and a pixel counts in `area` if its change per interval exceeds the threshold. Δt is rounded to
milliseconds first, so frames on the nominal grid (RWF offline, Sevilla stills) keep exactly the 3.0 values.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from sentinel.agents.detector import Track


@dataclass(frozen=True)
class FrameMotion:
    energy: float  # mean |difference| over the frame, grey levels
    area: float  # % of pixels that changed by more than the threshold
    people: dict[int, float]  # track id -> mean |difference| inside the person's box, grey levels


def interval_scale(dt: float, interval_s: float) -> float:
    """Factor that brings a change collected over `dt` seconds to the nominal interval (1.0 on the grid)."""
    return interval_s / round(dt, 3)


class PixelMotion:
    def __init__(self, max_width: int, threshold: float, interval_s: float) -> None:
        self.max_width = max_width
        self.threshold = threshold
        self.interval_s = interval_s  # nominal time between analysed frames, 1 / analysis fps
        self.reset()

    def reset(self) -> None:
        """Forget the previous frame (call between unrelated clips)."""
        self._previous: np.ndarray | None = None
        self._previous_t: float | None = None

    def _shrink(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        width = min(self.max_width, w)
        height = max(1, round(h * width / w))
        if w > 4 * width:  # nearest-neighbour to 4x the target first: area-averaging a 1080p frame is slow
            image = cv2.resize(image, (4 * width, 4 * height), interpolation=cv2.INTER_NEAREST)
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small

    def update(self, image: np.ndarray, tracks: list[Track], t: float) -> FrameMotion | None:
        """Motion since the previous frame, per nominal interval; None for the first frame, after the frame
        size changed or when the frame is not later than the previous one."""
        small = self._shrink(image)
        previous, self._previous = self._previous, small
        previous_t, self._previous_t = self._previous_t, t
        if previous is None or previous.shape != small.shape or round(t - previous_t, 3) <= 0:
            return None
        scale = interval_scale(t - previous_t, self.interval_s)
        diff = cv2.absdiff(small, previous)
        sx, sy = small.shape[1] / image.shape[1], small.shape[0] / image.shape[0]
        people = {}
        for track in tracks:
            x1, y1 = max(0, int(track.box[0] * sx)), max(0, int(track.box[1] * sy))
            x2 = min(diff.shape[1], math.ceil(track.box[2] * sx))
            y2 = min(diff.shape[0], math.ceil(track.box[3] * sy))
            if x2 > x1 and y2 > y1:
                people[track.track_id] = float(diff[y1:y2, x1:x2].mean()) * scale
        return FrameMotion(
            energy=float(diff.mean()) * scale,
            area=float((diff > self.threshold / scale).mean() * 100),
            people=people,
        )
