"""Pipeline: frame -> people + tracks -> pose -> weapons -> scene per window -> A -> incidents -> notifier.

Each closed window writes one journal line (results/run_<time>.jsonl): the scene description, A's risk, the
incident state, the age of the window's last frame and the processing time spent on the window.
"""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from sentinel.agents.context import ZoneContext
from sentinel.agents.detector import Track
from sentinel.agents.weapons import WeaponBox
from sentinel.config import Config
from sentinel.incidents import IncidentEvent, IncidentManager, KeyframeSaver
from sentinel.io.source import Frame
from sentinel.keyframe import Snapshot
from sentinel.notify import Notifier
from sentinel.scene import SceneBuilder
from sentinel.schema import RiskResult, SceneDescription, Zone


class Aggregator(Protocol):
    def assess(self, scene: SceneDescription) -> RiskResult: ...


@dataclass
class Agents:
    people: Any  # update(image) -> list[Track]; reset()
    poser: Any  # estimate(image, tracks) -> {track_id: keypoints}
    weapons: Any  # detect(image, people boxes) -> list[WeaponBox]; reset()


@dataclass
class _Window:
    """What the pipeline keeps about the window being filled."""

    snapshot: Snapshot | None = None
    rank: tuple[int, int] = (-1, -1)
    timing_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    frames: int = 0
    last_age_s: float | None = None

    def add(
        self,
        frame: Frame,
        tracks: list[Track],
        skeletons: dict[int, np.ndarray],
        weapons: list[WeaponBox],
        timing_ms: dict[str, float],
        age_s: float | None,
        others: list | None = None,
    ) -> None:
        rank = (int(bool(weapons)), len(tracks))  # prefer a frame showing a weapon, then the most people
        if rank >= self.rank:
            self.rank = rank
            self.snapshot = Snapshot(frame.image, tracks, skeletons, weapons, list(others or []))
        for stage, ms in timing_ms.items():
            self.timing_ms[stage] += ms
        self.frames += 1
        self.last_age_s = age_s


class Pipeline:
    def __init__(
        self,
        agents: Agents,
        aggregator: Aggregator,
        notifier: Notifier,
        config: Config,
        source_id: str,
        run_id: str,
        journal: Path | None,
        keyframes: KeyframeSaver,
        zone: Zone | None = None,
    ) -> None:
        self.agents, self.aggregator, self.notifier, self.config = agents, aggregator, notifier, config
        self.source_id, self.keyframes = source_id, keyframes
        self.journal = Path(journal) if journal is not None else None
        if self.journal is not None:
            self.journal.parent.mkdir(parents=True, exist_ok=True)
        self.zone = zone or ZoneContext.from_yaml(config.context.zones).zone_for(source_id)
        self.reset(run_id)

    def reset(self, run_id: str) -> None:
        """Start a new stream (e.g. the next clip): fresh tracks, windows and incident state."""
        self.agents.people.reset()
        self.agents.weapons.reset()
        cfg = self.config
        self.builder = SceneBuilder(
            cfg.features.to_feature_config(), cfg.analysis.window_s, self.source_id, self.zone
        )
        self.incidents = IncidentManager(cfg.incidents, self.source_id, self.keyframes, run_id)
        self._window = _Window()
        self.windows = 0
        self.last_timing_ms: dict[str, float] = {}
        self._last_t = 0.0

    def process(self, frame: Frame, age_s: float | None = None) -> list[IncidentEvent]:
        timing: dict[str, float] = {}
        started = time.perf_counter()
        tracks = self.agents.people.update(frame.image)
        timing["people"] = _ms_since(started)
        started = time.perf_counter()
        skeletons = self.agents.poser.estimate(frame.image, tracks)
        timing["pose"] = _ms_since(started)
        started = time.perf_counter()
        weapons = self.agents.weapons.detect(frame.image, [t.box for t in tracks])
        weapon_run = getattr(self.agents.weapons, "ran", True)
        timing["weapons"] = _ms_since(started)
        started = time.perf_counter()
        scene = self.builder.add(frame.t, tracks, skeletons, weapons, weapon_run)
        timing["scene"] = _ms_since(started)
        events = []
        started = time.perf_counter()
        if scene is not None:
            events = self._close_window(scene)
            self._window = _Window()  # this frame opens the next window
        self.last_timing_ms = timing | {"decision": _ms_since(started) if scene is not None else 0.0}
        others = getattr(self.agents.poser, "last_people", [])
        self._window.add(frame, tracks, skeletons, weapons, timing, age_s, others)
        self._last_t = frame.t
        return events

    def finish(self) -> list[IncidentEvent]:
        """End of the stream: close the last window and any incident still open."""
        scene = self.builder.flush()
        events = self._close_window(scene) if scene is not None else []
        t_end = scene.window.t_end if scene is not None else self._last_t
        closing = self.incidents.finish(t_end)
        for event in closing:
            self.notifier.on_event(event)
        if self.journal is not None and closing:
            line = {
                "t": t_end,
                "stream_end": True,
                "state": self.incidents.state.value,
                "events": _events(closing),
            }
            with self.journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return events + closing

    def _close_window(self, scene: SceneDescription) -> list[IncidentEvent]:
        started = time.perf_counter()
        risk = self.aggregator.assess(scene)
        events = self.incidents.on_window(scene.window.t_end, risk, scene, self._window.snapshot)
        decision_ms = _ms_since(started)
        for event in events:
            self.notifier.on_event(event)
        self.windows += 1
        if self.journal is not None:
            timing = {k: round(v, 1) for k, v in self._window.timing_ms.items()}
            timing |= {"decision": round(decision_ms, 1), "frames": self._window.frames}
            line = {
                "t": scene.window.t_end,
                "scene": scene.model_dump(mode="json"),
                "risk": risk.model_dump(mode="json"),
                "state": self.incidents.state.value,
                "frame_age_s": None if self._window.last_age_s is None else round(self._window.last_age_s, 3),
                "timing_ms": timing,
                "events": _events(events),
            }
            with self.journal.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return events


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _events(events: list[IncidentEvent]) -> list[dict[str, str]]:
    return [{"kind": e.kind, "incident_id": e.incident_id} for e in events]


@dataclass
class DriveResult:
    frames: int
    incidents: set[str]
    error: str | None  # the source failed (e.g. a camera read error)
    source_lost: bool  # a live stream stopped by itself (not by Ctrl+C)


def drive(reader: Any, pipeline: Any, mode: str) -> DriveResult:
    """Feed frames to the pipeline until the source ends, fails or Ctrl+C; always closes open incidents."""
    frames, incidents, error, interrupted = 0, set(), None, False
    started = time.monotonic()  # the reader's own clock starts on the first read, right below
    try:
        for frame in reader:
            age = None if mode == "offline" else time.monotonic() - started - frame.t
            incidents |= {e.incident_id for e in pipeline.process(frame, age)}
            frames += 1
    except KeyboardInterrupt:
        interrupted = True
    except RuntimeError as exc:  # LatestFrameReader: reading the source failed
        error = f"{exc}: {exc.__cause__}" if exc.__cause__ else str(exc)
    finally:
        incidents |= {e.incident_id for e in pipeline.finish()}
    lost = mode == "live" and not interrupted
    return DriveResult(frames, incidents, error, lost)


def model_files(config: Config) -> dict[str, Path]:
    """People, pose and weapon model files for the configured backend (OpenVINO: up-to-date exports)."""
    from sentinel.models import resolve_weights

    m = config.models
    paths = {"people": m.people.weights, "pose": m.pose.weights, "weapons": config.weapons.weights}
    return {role: resolve_weights(path, m.backend) for role, path in paths.items()}


def people_agents(config: Config, files: dict[str, Path], device: str) -> tuple[Any, Any]:
    """People and pose agents: a detector + a pose model matched to its tracks, or one tracked pose model."""
    from sentinel.agents.detector import PersonTracker, PoseTracker
    from sentinel.agents.pose_features import MatchRules, PoseEstimator

    m, pose = config.models, config.models.pose
    if m.people.source == "pose":
        tracker = PoseTracker(files["pose"], m.imgsz, m.people.conf, m.people.tracker, device)
        return tracker, tracker
    people = PersonTracker(files["people"], m.imgsz, m.people.conf, m.people.tracker, device)
    rules = MatchRules(pose.min_iou, pose.min_inside, pose.keypoint_conf, pose.box_margin)
    return people, PoseEstimator(files["pose"], m.imgsz, pose.conf, rules, device)


def build_pipeline(
    config: Config, source_id: str, run_id: str, journal: Path | None, notifier: Notifier
) -> Pipeline:
    """The real pipeline: YOLO people + ByteTrack, YOLO-pose, weapon detector, aggregator A, key frames."""
    from sentinel.agents.weapons import WeaponDetector
    from sentinel.aggregators.rules import RulesAggregator
    from sentinel.keyframe import KeyframeWriter

    m, w = config.models, config.weapons
    files, device = model_files(config), m.runtime_device
    people, poser = people_agents(config, files, device)
    agents = Agents(
        people=people,
        poser=poser,
        weapons=WeaponDetector.from_weights(
            files["weapons"], w.imgsz, w.conf, w.every_n_frames, w.class_map, w.mode, w.crop_pad, device
        ),
    )
    keyframes = KeyframeWriter(config.incidents.output_dir, config.incidents.blur_faces)
    aggregator = RulesAggregator.from_config(config)
    return Pipeline(agents, aggregator, notifier, config, source_id, run_id, journal, keyframes)


def run_source(config: Config, source: str, paced: bool, results: Path = Path("results")) -> dict:
    """`sentinel run`: a camera / RTSP stream (live) or a file (paced or offline) until it ends or Ctrl+C."""
    import os
    import sys

    from dotenv import find_dotenv, load_dotenv

    from sentinel.io.source import LatestFrameReader
    from sentinel.notify import make_notifier

    load_dotenv(find_dotenv(usecwd=True))  # .env of the working directory (also in Docker)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    live = source.isdigit() or "://" in source
    mode = "live" if live else "paced" if paced else "offline"
    notifier = make_notifier(os.environ, config.notify.incidents_jsonl, config.notify.edit_interval_s)
    print(f"notifications: {type(notifier).__name__}", file=sys.stderr)
    journal = results / f"run_{run_id}.jsonl"
    pipeline = build_pipeline(config, config.source.camera_id, run_id, journal, notifier)
    reader = LatestFrameReader(source, mode, target_fps=config.analysis.fps)
    started = time.monotonic()
    try:
        result = drive(reader, pipeline, mode)
    finally:
        reader.close()
        notifier.close()
    wall = time.monotonic() - started
    stats = reader.stats
    summary = {
        "mode": mode,
        "journal": str(journal),
        "frames_analysed": result.frames,
        "windows": pipeline.windows,
        "incidents": sorted(result.incidents),
        "wall_s": round(wall, 1),
        "skipped_share": round(stats.skipped / stats.received, 3) if stats.received else None,
        "source_error": result.error,
        "source_lost": result.source_lost,
    }
    if not live and reader.source_fps > 0 and stats.received:
        summary["rtf"] = round(wall / (stats.received / reader.source_fps), 2)
    if result.error or result.source_lost:
        print(f"camera / stream unavailable: {result.error or 'the stream ended'}", file=sys.stderr)
    return summary
