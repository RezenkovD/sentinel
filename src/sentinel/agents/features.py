"""Person and pair features over one window (~1 s) of observations.

All time arithmetic uses the real frame timestamps. Distances and speeds are normalised by the person's
height: the median height of the track box over the window (units: heights, heights/s, heights/s²).
A feature that cannot be computed is None rather than an invented value.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from sentinel.agents.weapons import WeaponBox

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST = 0, 5, 6, 9, 10  # COCO keypoints

Pair = tuple[int, int]


@dataclass(frozen=True)
class FeatureConfig:
    min_keypoint_conf: float  # keypoints below this score are not used
    hold_margin_h: float  # weapon box expansion for "holds", in heights
    hold_min_frames: int  # frames of the window with a wrist in the (expanded) weapon box
    pair_max_distance_h: float  # only pairs closer than this go into the description
    max_pairs: int
    contact_distance_h: float  # a pair closer than this is "in contact"
    facing_min_offset_h: float  # nose-to-shoulder-midpoint horizontal offset that shows a direction


@dataclass(frozen=True, eq=False)
class Observation:
    """One person track in one analysed frame."""

    t: float
    box: tuple[float, float, float, float]
    conf: float
    keypoints: np.ndarray | None  # (17, 3) x, y, score; None if no skeleton was attached


def height(obs: Sequence[Observation]) -> float:
    """Person height in pixels: median box height over the window."""
    return float(np.median([o.box[3] - o.box[1] for o in obs]))


def _centre(box: tuple[float, float, float, float]) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def person_speed(obs: Sequence[Observation], h: float) -> float | None:
    """Median speed of the box centre over consecutive observations, heights/s."""
    speeds = [
        float(np.linalg.norm(_centre(b.box) - _centre(a.box))) / (b.t - a.t) / h
        for a, b in pairwise(obs)
        if b.t > a.t
    ]
    return float(np.median(speeds)) if speeds else None


def _point(o: Observation, index: int, cfg: FeatureConfig) -> np.ndarray | None:
    if o.keypoints is None or o.keypoints[index, 2] < cfg.min_keypoint_conf:
        return None
    return o.keypoints[index, :2]


def limb_accel(obs: Sequence[Observation], h: float, cfg: FeatureConfig) -> float | None:
    """Largest wrist acceleration, heights/s², from differences over three consecutive confident points."""
    best = None
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        track = [(o.t, p) for o in obs if (p := _point(o, wrist, cfg)) is not None]
        for (t1, p1), (t2, p2), (t3, p3) in zip(track, track[1:], track[2:], strict=False):
            if not (t1 < t2 < t3):
                continue
            accel = 2 * ((p3 - p2) / (t3 - t2) - (p2 - p1) / (t2 - t1)) / (t3 - t1)
            value = float(np.linalg.norm(accel)) / h
            best = value if best is None else max(best, value)
    return best


def holds(
    obs: Sequence[Observation], weapons_at: dict[float, list[WeaponBox]], h: float, cfg: FeatureConfig
) -> str | None:
    """Weapon kind whose box (expanded by hold_margin_h heights) contains a wrist in enough frames."""
    margin = cfg.hold_margin_h * h
    frames: Counter[str] = Counter()
    for o in obs:
        wrists = [p for w in (LEFT_WRIST, RIGHT_WRIST) if (p := _point(o, w, cfg)) is not None]
        kinds = {
            weapon.kind
            for weapon in weapons_at.get(o.t, [])
            for x, y in wrists
            if weapon.box[0] - margin <= x <= weapon.box[2] + margin
            and weapon.box[1] - margin <= y <= weapon.box[3] + margin
        }
        frames.update(kinds)
    held = [k for k, n in frames.items() if n >= cfg.hold_min_frames]
    if not held:
        return None
    return max(held, key=lambda k: (frames[k], k == "gun"))


def distance_series(
    a: Sequence[Observation], b: Sequence[Observation], h_a: float, h_b: float
) -> list[tuple[float, float]]:
    """(t, distance between box centres / mean height of the pair) over the frames both are seen in."""
    b_at = {o.t: o for o in b}
    scale = (h_a + h_b) / 2
    return [
        (o.t, float(np.linalg.norm(_centre(o.box) - _centre(b_at[o.t].box))) / scale)
        for o in a
        if o.t in b_at
    ]


def approach_speed(series: Sequence[tuple[float, float]]) -> float | None:
    """Largest rate at which the pair distance shrinks, heights/s (positive = approaching)."""
    rates = [-(d2 - d1) / (t2 - t1) for (t1, d1), (t2, d2) in pairwise(series) if t2 > t1]
    return max(rates) if rates else None


def _direction(o: Observation, h: float, cfg: FeatureConfig) -> int | None:
    """-1 / +1 when turned left / right in the image (nose off the shoulder midpoint), else None."""
    nose = _point(o, NOSE, cfg)
    left, right = _point(o, LEFT_SHOULDER, cfg), _point(o, RIGHT_SHOULDER, cfg)
    if nose is None or left is None or right is None:
        return None
    offset = nose[0] - (left[0] + right[0]) / 2
    if abs(offset) < cfg.facing_min_offset_h * h:
        return None
    return 1 if offset > 0 else -1


def facing(
    a: Sequence[Observation], b: Sequence[Observation], h_a: float, h_b: float, cfg: FeatureConfig
) -> bool | None:
    """Both turned towards each other (horizontal component), by majority over frames where it can be seen."""
    b_at = {o.t: o for o in b}
    votes = []
    for oa in a:
        ob = b_at.get(oa.t)
        if ob is None:
            continue
        dir_a, dir_b = _direction(oa, h_a, cfg), _direction(ob, h_b, cfg)
        if dir_a is None or dir_b is None:
            continue
        towards_b = 1 if _centre(ob.box)[0] >= _centre(oa.box)[0] else -1
        votes.append(dir_a == towards_b and dir_b == -towards_b)
    if not votes:
        return None
    return sum(votes) * 2 > len(votes)


class ContactTracker:
    """How long each pair has been continuously in contact, across window boundaries.

    A run starts at the first frame the pair is closer than `contact_distance_h` and lasts until a frame shows
    them apart; frames where the pair is not seen do not break the run. `window_durations` reports, for the
    window just finished, the longest run observed in it (counted from the run's start, even if that was in an
    earlier window).
    """

    def __init__(self, contact_distance_h: float) -> None:
        self.contact_distance_h = contact_distance_h
        self._start: dict[Pair, float] = {}
        self._window: dict[Pair, float] = {}

    def update(self, t: float, distances: dict[Pair, float]) -> None:
        for pair, d in distances.items():
            if d < self.contact_distance_h:
                start = self._start.setdefault(pair, t)
                self._window[pair] = max(self._window.get(pair, 0.0), t - start)
            else:
                self._start.pop(pair, None)

    def window_durations(self) -> dict[Pair, float]:
        durations, self._window = self._window, {}
        return durations
