"""Person and pair features over one window (~1 s) of observations.

All time arithmetic uses the real frame timestamps. Distances and speeds are normalised by the person's
height: the median height of the track box over the window (units: heights, heights/s, heights/s²).
A feature that cannot be computed is None rather than an invented value.
"""

import math
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
    hold_min_share: float  # share of the detector runs (while the person is seen) with a wrist in the box
    weapon_max_height_h: float  # a taller weapon box (in heights of the person) is implausible
    pair_max_distance_h: float  # only pairs closer than this go into the description
    max_pairs: int
    contact_distance_h: float  # a pair closer than this is "in contact"
    facing_min_offset_h: float  # nose-to-shoulder-midpoint horizontal offset that shows a direction
    smooth_points: int = 5  # keypoint trajectories: local least-squares fits over this many points…
    max_gap_s: float = 0.3  # …never across a gap longer than this (no interpolation)


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


def plausible(weapon: WeaponBox, h: float, cfg: FeatureConfig) -> bool:
    """A weapon box taller than weapon_max_height_h person heights is a false detection (a whole person)."""
    return weapon.box[3] - weapon.box[1] <= cfg.weapon_max_height_h * h


def holds(
    obs: Sequence[Observation], runs: dict[float, list[WeaponBox]], h: float, cfg: FeatureConfig
) -> str | None:
    """Weapon kind with a confident wrist inside its (expanded, plausible) box in enough detector runs.

    `runs` maps the times of the frames where the weapon detector ran to its boxes (boxes kept between runs
    are not repeated here). A weapon is held if it is at a wrist in at least `hold_min_share` of the runs made
    while the person was in view, and in at least one: 2 of 5 runs at 10 frames/s with the detector on every
    2nd frame, 1 of 2 on the 2-frames/s Sevilla sequences.
    """
    margin = cfg.hold_margin_h * h
    seen = [o for o in obs if o.t in runs]
    counts: dict[str, int] = {}
    for o in seen:
        wrists = [p for w in (LEFT_WRIST, RIGHT_WRIST) if (p := _point(o, w, cfg)) is not None]
        kinds = {
            weapon.kind
            for weapon in runs[o.t]
            if plausible(weapon, h, cfg)
            and any(
                weapon.box[0] - margin <= x <= weapon.box[2] + margin
                and weapon.box[1] - margin <= y <= weapon.box[3] + margin
                for x, y in wrists
            )
        }
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
    need = max(1, math.ceil(cfg.hold_min_share * len(seen) - 1e-9))
    held = [k for k, n in counts.items() if n >= need]
    if not held:
        return None
    return max(held, key=lambda k: (counts[k], k == "gun"))


@dataclass(frozen=True)
class Kinematics:
    """Smoothed keypoint motion relative to the torso over one window (schema v3)."""

    limb_speed: float | None  # p90 speed of the wrists relative to the shoulder midpoint, heights/s
    limb_accel: float | None  # largest wrist acceleration relative to the shoulder midpoint, heights/s²
    pose_motion: (
        float | None
    )  # mean speed of confident keypoints relative to the shoulder midpoint, heights/s


def _relative_series(
    obs: Sequence[Observation], index: int, cfg: FeatureConfig
) -> list[tuple[float, np.ndarray]]:
    """(t, keypoint - shoulder midpoint) in the frames where the keypoint and both shoulders are confident.

    The shoulder midpoint is the torso reference: hips are often hidden, and a reference whose points come and
    go from frame to frame would jump and fake motion.
    """
    series = []
    for o in obs:
        p, left, right = _point(o, index, cfg), _point(o, LEFT_SHOULDER, cfg), _point(o, RIGHT_SHOULDER, cfg)
        if p is not None and left is not None and right is not None:
            series.append((o.t, p - (left + right) / 2))
    return series


def _derivatives(
    series: list[tuple[float, np.ndarray]], cfg: FeatureConfig, with_acceleration: bool
) -> list[tuple[float, np.ndarray | None, np.ndarray | None]]:
    """(t, velocity, acceleration) at every point from local least-squares fits over up to smooth_points
    neighbours (a line for the velocity from 3 points, a parabola for the acceleration from 4), using the real
    time stamps and never reaching across a gap longer than max_gap_s. Vectorised over the points."""
    half = cfg.smooth_points // 2
    segments: list[list[tuple[float, np.ndarray]]] = []
    for t, p in series:
        if not segments or t - segments[-1][-1][0] > cfg.max_gap_s:
            segments.append([])
        segments[-1].append((t, p))
    out = []
    offsets = np.arange(-half, half + 1)
    for segment in segments:
        ts = np.array([t for t, _ in segment])
        ps = np.array([p for _, p in segment])
        n = len(segment)
        idx = np.arange(n)[:, None] + offsets[None, :]  # (n, w) neighbours of every point
        valid = (idx >= 0) & (idx < n)
        idx = np.clip(idx, 0, n - 1)
        w = valid.astype(float)
        count = valid.sum(axis=1)
        dt = np.where(valid, ts[idx] - ts[:, None], 0.0)
        pts = ps[idx] * w[..., None]  # (n, w, 2), zero where invalid
        # line: slope = cov(dt, p) / var(dt) over the valid neighbours
        mean_dt = dt.sum(axis=1) / count
        mean_p = pts.sum(axis=1) / count[:, None]
        ddt = (dt - mean_dt[:, None]) * w
        var = (ddt**2).sum(axis=1)
        cov = (ddt[..., None] * (pts - mean_p[:, None, :] * w[..., None])).sum(axis=1)
        velocity_ok = (count >= 3) & (var > 0)
        velocity = np.zeros((n, 2))
        velocity[velocity_ok] = cov[velocity_ok] / var[velocity_ok, None]
        acceleration = np.zeros((n, 2))
        acceleration_ok = np.zeros(n, dtype=bool)
        if with_acceleration:
            design = np.stack([w, dt, dt**2], axis=-1)  # rows of invalid neighbours are zero
            acceleration_ok = count >= 4
            if acceleration_ok.any():
                a = design[acceleration_ok]
                normal = a.transpose(0, 2, 1) @ a
                rhs = a.transpose(0, 2, 1) @ pts[acceleration_ok]
                acceleration[acceleration_ok] = 2 * np.linalg.solve(normal, rhs)[:, 2, :]
        for j in range(n):
            out.append(
                (
                    ts[j],
                    velocity[j] if velocity_ok[j] else None,
                    acceleration[j] if acceleration_ok[j] else None,
                )
            )
    return out


def pose_kinematics(obs: Sequence[Observation], h: float, cfg: FeatureConfig, t_min: float) -> Kinematics:
    """Limb and body motion relative to the shoulders, summarised over the points at or after `t_min`.

    `obs` may start before the window (`t_min`): earlier points only help the fits at the window's start.
    Relative motion tells a fight (fists move around the torso) from running (the whole body moves).
    """
    wrist_speeds, wrist_accels, speeds = [], [], []
    for index in range(17):
        if index in (LEFT_SHOULDER, RIGHT_SHOULDER):
            continue
        wrist = index in (LEFT_WRIST, RIGHT_WRIST)
        for t, velocity, acceleration in _derivatives(_relative_series(obs, index, cfg), cfg, wrist):
            if t < t_min - 1e-9:
                continue
            if velocity is not None:
                speed = float(np.linalg.norm(velocity)) / h
                speeds.append(speed)
                if index in (LEFT_WRIST, RIGHT_WRIST):
                    wrist_speeds.append(speed)
            if acceleration is not None and index in (LEFT_WRIST, RIGHT_WRIST):
                wrist_accels.append(float(np.linalg.norm(acceleration)) / h)
    return Kinematics(
        limb_speed=float(np.percentile(wrist_speeds, 90)) if wrist_speeds else None,
        limb_accel=max(wrist_accels) if wrist_accels else None,
        pose_motion=float(np.mean(speeds)) if speeds else None,
    )


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
