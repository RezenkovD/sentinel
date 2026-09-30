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

# Version of the perception code (agents, features, scene assembly). The episode cache is keyed by it and by a
# hash of the perception settings and files, but code is not hashed: any change to perception code after the
# freeze (session 03, step 2) must bump this version, or cached and live scenes would silently differ.
PERCEPTION_VERSION = "3.0"


def perception_hash(config: Config) -> str:
    """Hash of everything that shapes the scene descriptions: perception settings, zone and tracker files,
    model checkpoints, schema version. Aggregator and incident settings are downstream and not included."""
    import hashlib

    from sentinel.models import sha256
    from sentinel.schema import SCHEMA_VERSION

    dump = config.model_dump(mode="json")
    settings = {k: dump[k] for k in ("analysis", "models", "weapons", "features", "motion", "context")}
    m = config.models
    files = [
        m.people.weights,
        m.pose.weights,
        config.weapons.weights,
        Path(m.people.tracker),
        config.context.zones,
    ]
    digest = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "files": {str(p): sha256(Path(p)) for p in files},
    }
    return hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()[:16]


class Aggregator(Protocol):
    def assess(self, scene: SceneDescription) -> RiskResult: ...


@dataclass
class Agents:
    people: Any  # update(image) -> list[Track]; reset()
    poser: Any  # estimate(image, tracks) -> {track_id: keypoints}
    weapons: Any  # detect(image, people boxes) -> list[WeaponBox]; reset()
    motion: Any = None  # update(image, tracks) -> FrameMotion | None; reset() — pixel motion (schema v3)


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


@dataclass
class FrameResult:
    """What the perception agents found in one frame (kept by the pipeline for key frames and timing)."""

    tracks: list[Track]
    skeletons: dict[int, np.ndarray]
    weapons: list[WeaponBox]
    others: list  # every person the pose model saw (face blurring)
    timing_ms: dict[str, float]


class Perception:
    """Lower layer: frame -> people, pose, weapons, pixel motion -> one scene description per window.

    Shared by `sentinel run` and the episode cache (`sentinel eval extract`), so both see exactly the same
    scenes. Next to each scene comes its `aux`: service data outside the scene description that aggregators
    never see — the frames in the window and the best raw weapon score (for the baseline detector).
    """

    def __init__(self, agents: Agents, config: Config, source_id: str, zone: Zone) -> None:
        self.agents, self.config, self.source_id, self.zone = agents, config, source_id, zone
        self.reset()

    def reset(self) -> None:
        """Start a new stream: fresh tracks, windows and pixel history."""
        self.agents.people.reset()
        self.agents.weapons.reset()
        if self.agents.motion is not None:
            self.agents.motion.reset()
        cfg = self.config
        self.builder = SceneBuilder(
            cfg.features.to_feature_config(), cfg.analysis.window_s, self.source_id, self.zone
        )
        self._aux = self._new_aux()

    @staticmethod
    def _new_aux() -> dict:
        return {"frames": 0, "weapon_max_conf": 0.0}

    def process(self, frame: Frame) -> tuple[SceneDescription | None, dict | None, FrameResult]:
        """Analyse a frame; returns the scene (and aux) of the window this frame closed, if any."""
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
        motion = None if self.agents.motion is None else self.agents.motion.update(frame.image, tracks)
        timing["motion"] = _ms_since(started)
        started = time.perf_counter()
        scene = self.builder.add(frame.t, tracks, skeletons, weapons, weapon_run, motion)
        timing["scene"] = _ms_since(started)
        aux = None
        if scene is not None:  # this frame opens the next window
            aux, self._aux = self._aux, self._new_aux()
        self._aux["frames"] += 1
        if weapon_run:
            self._aux["weapon_max_conf"] = max(
                self._aux["weapon_max_conf"], float(getattr(self.agents.weapons, "last_max_conf", 0.0))
            )
        others = getattr(self.agents.poser, "last_people", [])
        return scene, aux, FrameResult(tracks, skeletons, weapons, others, timing)

    def flush(self) -> tuple[SceneDescription | None, dict | None]:
        """End of the stream: the last, possibly shorter, window."""
        scene = self.builder.flush()
        aux, self._aux = (self._aux, self._new_aux()) if scene is not None else (None, self._aux)
        return scene, aux


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
        self.perception = Perception(agents, config, source_id, self.zone)
        self.reset(run_id)

    def reset(self, run_id: str) -> None:
        """Start a new stream (e.g. the next clip): fresh tracks, windows and incident state."""
        self.perception.reset()
        self.incidents = IncidentManager(self.config.incidents, self.source_id, self.keyframes, run_id)
        self._window = _Window()
        self.windows = 0
        self.last_timing_ms: dict[str, float] = {}
        self._last_t = 0.0

    def process(self, frame: Frame, age_s: float | None = None) -> list[IncidentEvent]:
        scene, _, found = self.perception.process(frame)
        events = []
        started = time.perf_counter()
        if scene is not None:
            events = self._close_window(scene)
            self._window = _Window()  # this frame opens the next window
        self.last_timing_ms = found.timing_ms | {"decision": _ms_since(started) if scene is not None else 0.0}
        self._window.add(
            frame, found.tracks, found.skeletons, found.weapons, found.timing_ms, age_s, found.others
        )
        self._last_t = frame.t
        return events

    def finish(self) -> list[IncidentEvent]:
        """End of the stream: close the last window and any incident still open."""
        scene, _ = self.perception.flush()
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


def build_agents(config: Config, record_weapons: bool = False) -> Agents:
    """The perception agents with real models; `record_weapons` scores weapons down to weapons.record_conf."""
    from sentinel.agents.motion import PixelMotion
    from sentinel.agents.weapons import WeaponDetector

    m, w = config.models, config.weapons
    files, device = model_files(config), m.runtime_device
    people, poser = people_agents(config, files, device)
    return Agents(
        people=people,
        poser=poser,
        weapons=WeaponDetector.from_weights(
            files["weapons"],
            w.imgsz,
            w.conf,
            w.every_n_frames,
            w.class_map,
            w.mode,
            w.crop_pad,
            device,
            record_conf=w.record_conf if record_weapons else None,
        ),
        motion=PixelMotion(config.motion.max_width, config.motion.threshold),
    )


def build_pipeline(
    config: Config, source_id: str, run_id: str, journal: Path | None, notifier: Notifier
) -> Pipeline:
    """The real pipeline: YOLO people + ByteTrack, YOLO-pose, weapon detector, aggregator A, key frames."""
    from sentinel.aggregators.rules import make_aggregator
    from sentinel.keyframe import KeyframeWriter

    agents = build_agents(config)
    keyframes = KeyframeWriter(config.incidents.output_dir, config.incidents.blur_faces)
    aggregator = make_aggregator(config)
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
