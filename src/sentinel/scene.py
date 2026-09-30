"""Scene description assembly: frames of one window (~1 s) -> SceneDescription (SPEC §5).

Windows are aligned to multiples of window_s on the frame clock. A frame at or past the end of the current
window closes it and opens the next one; `flush` closes the last, possibly shorter, window. Pair contact
duration is tracked across windows.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from sentinel.agents.detector import Track
from sentinel.agents.features import (
    ContactTracker,
    FeatureConfig,
    Observation,
    approach_speed,
    distance_series,
    facing,
    height,
    holds,
    limb_accel,
    person_speed,
    plausible,
)
from sentinel.agents.weapons import WeaponBox
from sentinel.schema import Interaction, Person, SceneDescription, Summary, Window, Zone

_EPS = 1e-9


@dataclass(frozen=True, eq=False)
class FrameRecord:
    t: float
    tracks: list[Track]
    skeletons: dict[int, np.ndarray]
    weapons: list[WeaponBox]
    weapon_run: bool = True  # the weapon detector ran on this frame (else `weapons` are its kept boxes)


class SceneBuilder:
    def __init__(self, cfg: FeatureConfig, window_s: float, source: str, zone: Zone) -> None:
        self.cfg = cfg
        self.window_s = window_s
        self.source = source
        self.zone = zone  # may be reassigned between windows (e.g. day / night)
        self._frames: list[FrameRecord] = []
        self._window_start: float | None = None
        self._contacts = ContactTracker(cfg.contact_distance_h)

    def add(
        self,
        t: float,
        tracks: list[Track],
        skeletons: dict[int, np.ndarray],
        weapons: list[WeaponBox],
        weapon_run: bool = True,
    ) -> SceneDescription | None:
        """Add an analysed frame; returns the description of the window this frame closed, if any."""
        scene = None
        if self._window_start is None:
            self._window_start = math.floor(t / self.window_s + _EPS) * self.window_s
        elif t >= self._window_start + self.window_s - _EPS:
            scene = self._build(self._window_start + self.window_s)
            self._window_start = math.floor(t / self.window_s + _EPS) * self.window_s
        self._frames.append(FrameRecord(t, tracks, skeletons, weapons, weapon_run))
        return scene

    def flush(self) -> SceneDescription | None:
        """Close the current window at its last frame (end of a file)."""
        if not self._frames:
            return None
        return self._build(self._frames[-1].t)

    def _build(self, t_end: float) -> SceneDescription:
        frames, self._frames = self._frames, []
        cfg = self.cfg
        observations: dict[int, list[Observation]] = defaultdict(list)
        runs = {f.t: f.weapons for f in frames if f.weapon_run}
        for f in frames:
            for tr in f.tracks:
                observations[tr.track_id].append(
                    Observation(f.t, tr.box, tr.conf, f.skeletons.get(tr.track_id))
                )
        heights = {tid: height(obs) for tid, obs in observations.items()}

        persons = tuple(
            Person(
                id=tid,
                speed=person_speed(obs, heights[tid]),
                limb_accel=limb_accel(obs, heights[tid], cfg),
                holds=holds(obs, runs, heights[tid], cfg),
                conf=float(np.mean([o.conf for o in obs])),
            )
            for tid, obs in sorted(observations.items())
        )

        series = {}
        for a, b in combinations(sorted(observations), 2):
            s = distance_series(observations[a], observations[b], heights[a], heights[b])
            if s:
                series[(a, b)] = s
        at_time: dict[float, dict[tuple[int, int], float]] = defaultdict(dict)
        for pair, s in series.items():
            for t, d in s:
                at_time[t][pair] = d
        for f in frames:
            self._contacts.update(f.t, at_time.get(f.t, {}))
        durations = self._contacts.window_durations()

        closest = sorted((min(d for _, d in s), pair) for pair, s in series.items())
        close = [(d, pair) for d, pair in closest if d < cfg.pair_max_distance_h][: cfg.max_pairs]
        interactions = tuple(
            Interaction(
                pair=(a, b),
                distance_h=d,
                approach_speed=approach_speed(series[(a, b)]),
                facing=facing(observations[a], observations[b], heights[a], heights[b], cfg),
                duration_s=durations.get((a, b), 0.0),
            )
            for d, (a, b) in close
        )

        speeds = [p.speed for p in persons if p.speed is not None]
        crowd_motion = (float(np.mean(speeds)) if speeds else None) if persons else 0.0
        summary = Summary(
            n_persons=len(persons),
            crowd_motion=crowd_motion,
            weapon_visible=any(self._weapon_in_view(f) for f in frames),
        )
        window = Window(t_start=self._window_start, t_end=t_end, source=self.source)
        return SceneDescription(
            window=window, zone=self.zone, summary=summary, persons=persons, interactions=interactions
        )

    def _weapon_in_view(self, f: FrameRecord) -> bool:
        """A plausible weapon box in the frame: not taller than allowed for the tallest person in view."""
        if not f.tracks:
            return bool(f.weapons)  # nobody to compare the box with
        tallest = max(tr.box[3] - tr.box[1] for tr in f.tracks)
        return any(plausible(w, tallest, self.cfg) for w in f.weapons)
