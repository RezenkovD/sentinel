"""Episode cache (SPEC §9; session 03, step 3): one perception pass per episode, then A, B, the cascade and
the baselines all read the same files.

    uv run sentinel eval extract --plan all       # RWF train + val -> RWF test -> Sevilla -> XD subset
    uv run sentinel eval extract --dataset rwf2000 --parts train --per-class 100   # a quick sample

Each episode becomes data/episodes/<dataset>/<episode_id>.jsonl: a first line {"meta": {...}} and one line per
window {"scene": <SceneDescription>, "aux": {...}}. `aux` lies outside the scene description and no aggregator
sees it: the frames in the window, the best raw weapon score (for the baseline detector) and the window label
where the dataset has frame-level annotation (Sevilla, XD-Violence); RWF has clip labels only (in `meta`).

Resumable: finished files are skipped; an interrupted episode leaves nothing behind (written to a temporary
file, then renamed); a file made by another perception (version or settings hash) is refused, never mixed in.
Runs in the background with nice 19, pinned to a few CPU cores (the efficiency cores of this laptop), and is
paused by the quiet-window bench while it measures (the PID goes to data/episodes/.extract.pid).
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from sentinel.config import DEFAULT_CONFIG, Config, load_config
from sentinel.pipeline import perception_hash  # noqa: F401  (re-exported: the cache's key)
from sentinel.schema import SCHEMA_VERSION, SceneDescription

OUT = Path("data/episodes")
XD_SCENARIO = {"B1", "B2"}  # Fighting and Shooting; riots, abuse, accidents and explosions are out of scope


@dataclass
class Episode:
    dataset: str
    part: str
    episode_id: str  # also the file path under data/episodes/<dataset>/
    label: int  # 1 = dangerous (fight / weapon), 0 = normal
    duration_s: float
    frames: Callable[[], Iterator]  # a fresh iterator of frames (with .t and .image) on every call
    events: list[tuple[float, float]] | None = None  # annotated dangerous intervals, seconds (XD-Violence)
    frame_labels: dict[float, int] | None = None  # label of each frame by its time (Sevilla)
    source: str = ""


# --- labels ---


def window_label(t_start: float, t_end: float, events: list[tuple[float, float]]) -> int:
    """1 if at least half of the window lies inside annotated events."""
    length = t_end - t_start
    inside = sum(max(0.0, min(t_end, b) - max(t_start, a)) for a, b in events)
    return int(length > 0 and inside >= 0.5 * length - 1e-9)


def xd_labels(name: str) -> list[str]:
    """Class labels of an XD-Violence video from its name: ..._label_B2-G-0 -> ["B2", "G"]."""
    return [label for label in name.rsplit("_label_", 1)[1].split("-") if label != "0"]


def select_xd_subset(videos: list[dict], budget_hours: float, seed: int) -> dict:
    """Test subset of about `budget_hours`: equal time for Fighting (B1), Shooting (B2) and normal (A) videos.

    Videos whose labels leave the scenario (riots, abuse, accidents, explosions — alone or mixed in) are left
    out.
    """
    strata: dict[str, list[dict]] = {"B1": [], "B2": [], "A": []}
    out_of_scope = 0
    for video in sorted(videos, key=lambda v: v["name"]):
        labels = xd_labels(video["name"])
        if labels == ["A"]:
            strata["A"].append(video)
        elif set(labels) <= XD_SCENARIO:
            strata[labels[0]].append(video)
        else:
            out_of_scope += 1
    share = budget_hours * 3600 / len(strata)
    rng = random.Random(seed)
    chosen = []
    for stratum, pool in strata.items():
        pool = list(pool)
        rng.shuffle(pool)
        total = 0.0
        for video in pool:
            if total >= share:
                break
            chosen.append({**video, "stratum": stratum, "labels": xd_labels(video["name"])})
            total += video["duration_s"]
    return {
        "budget_hours": budget_hours,
        "seed": seed,
        "rule": "equal time for B1 / B2 / A; labels outside {B1, B2} excluded (scenario limit)",
        "videos": chosen,
        "excluded": {"out_of_scope": out_of_scope},
    }


def sevilla_frames(images: Path, segment: str) -> list[tuple[float, Path, int]]:
    """(time, image, label) of a Sevilla segment: frames numbered at 2 frames/s; label 1 = a weapon box."""
    frames = []
    for path in images.glob(f"{segment}_frame_*.jpg"):
        number = int(re.search(r"_frame_(\d+)\.jpg$", path.name).group(1))
        xml = path.with_suffix(".xml")
        label = int(xml.exists() and ET.parse(xml).getroot().find("object") is not None)
        frames.append((number, path, label))
    frames.sort()
    first = frames[0][0] if frames else 0
    return [((number - first) / 2.0, path, label) for number, path, label in frames]


# --- perception settings ---


# --- writing ---


def _labels_of(episode: Episode, scene: SceneDescription, frame_times: list[float]) -> int | None:
    if episode.events is not None:
        return window_label(scene.window.t_start, scene.window.t_end, episode.events)
    if episode.frame_labels is not None:
        return max((episode.frame_labels.get(t, 0) for t in frame_times), default=0)
    return None


def _read_meta(path: Path) -> dict | None:
    """Meta of an episode file, or None if the file is unreadable (e.g. cut short by a power loss)."""
    try:
        with path.open(encoding="utf-8") as f:
            return json.loads(f.readline())["meta"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def is_current(path: Path, meta: dict) -> bool:
    """The file exists and was made by this perception: the same settings hash, and the same version or one
    this code reads as its own (`COMPATIBLE_PERCEPTIONS`)."""
    from sentinel.pipeline import compatible_perceptions

    existing = _read_meta(path) if path.exists() else None
    return (
        existing is not None
        and existing.get("perception_hash") == meta["perception_hash"]
        and compatible_perceptions({existing.get("perception_version"), meta["perception_version"]})
    )


def extract_episode(episode: Episode, perception, out: Path, meta: dict, redo: bool = False) -> bool:
    """Run perception over the episode and write its file; False if a file of this perception exists.

    A file of another perception is an error unless `redo`; an unreadable file is simply made again.
    """
    if out.exists():
        if is_current(out, meta):
            return False
        existing = _read_meta(out)
        if existing is not None and not redo:
            raise RuntimeError(
                f"{out} was made by another perception ({existing.get('perception_version')}, "
                f"{existing.get('perception_hash')}): remove it or run with --redo"
            )
    perception.reset()
    started = time.perf_counter()
    lines: list[dict] = []
    in_window: list[float] = []
    for frame in episode.frames():
        scene, aux, _ = perception.process(frame)
        if scene is not None:
            lines.append({"scene": scene, "aux": aux | {"label": _labels_of(episode, scene, in_window)}})
            in_window = []
        in_window.append(frame.t)
    scene, aux = perception.flush()
    if scene is not None:
        lines.append({"scene": scene, "aux": aux | {"label": _labels_of(episode, scene, in_window)}})
    header = meta | {
        "dataset": episode.dataset,
        "part": episode.part,
        "episode_id": episode.episode_id,
        "label": episode.label,
        "duration_s": round(episode.duration_s, 2),
        "windows": len(lines),
        "source": episode.source,
        "perception_s": round(time.perf_counter() - started, 1),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"meta": header}, ensure_ascii=False) + "\n")
            for line in lines:
                record = {"scene": line["scene"].model_dump(mode="json"), "aux": line["aux"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # the data reaches the disk before the rename (a power loss leaves no stub)
        os.replace(tmp, out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return True


# --- datasets ---


@dataclass(frozen=True)
class _ImageFrame:
    image: object
    t: float
    index: int


def rwf_episodes(
    split: Path, root: Path, parts: list[str], per_class: int | None, seed: int, fps: float
) -> list[Episode]:
    """Clip = episode, clip label; parts of data/splits/rwf2000.json (train / val / test)."""
    from sentinel.io.source import LatestFrameReader

    data = json.loads(split.read_text())
    episodes = []
    for part in parts:
        clips = data[part]
        if per_class is not None:  # a balanced sample (the quick check before freezing perception)
            rng = random.Random(seed)
            by_class = {label: [c for c in clips if f"/{label}/" in c] for label in ("Fight", "NonFight")}
            clips = [c for label in by_class for c in rng.sample(sorted(by_class[label]), per_class)]
        for clip in clips:
            label = "Fight" if "/Fight/" in clip else "NonFight"
            path = root / clip

            def frames(path: Path = path) -> Iterator:
                with LatestFrameReader(path, mode="offline", target_fps=fps) as reader:
                    yield from reader

            episodes.append(
                Episode(
                    dataset="rwf2000",
                    part=part,
                    episode_id=f"{part}/{label}/{Path(clip).stem}",
                    label=int(label == "Fight"),
                    duration_s=5.0,
                    frames=frames,
                    source=clip,
                )
            )
    return episodes


def sevilla_episodes(split: Path, images: Path, parts: list[str]) -> list[Episode]:
    """Segment = episode (2 frames/s, frame labels from the XML); Cam1 + Cam5 = select (dev), Cam7 = test."""
    data = json.loads(split.read_text())
    episodes = []
    for part in parts:
        for segment in data["segments"][part]:
            frames = sevilla_frames(images, segment)
            if not frames:
                continue

            def iterate(frames: list = frames) -> Iterator:
                for i, (t, path, _) in enumerate(frames):
                    yield _ImageFrame(image=cv2.imread(str(path)), t=t, index=i)

            episodes.append(
                Episode(
                    dataset="us_gun",
                    part=part,
                    episode_id=f"{part}/{segment}",
                    label=int(any(label for _, _, label in frames)),
                    duration_s=frames[-1][0] + 0.5,
                    frames=iterate,
                    frame_labels={t: label for t, _, label in frames},
                    source=segment,
                )
            )
    return episodes


def xd_episodes(subset: Path, videos: Path, annotations: Path, fps: float) -> list[Episode]:
    """Test videos of the XD-Violence subset; window labels from the frame-level annotation."""
    from sentinel.io.source import LatestFrameReader

    events_by_name: dict[str, list[int]] = {}
    for line in annotations.read_text().splitlines():
        if not line.strip():
            continue
        name, *numbers = line.split()
        events_by_name[name.removesuffix(".mp4")] = [int(n) for n in numbers]  # 14 names carry the extension
    episodes = []
    for video in json.loads(subset.read_text())["videos"]:
        path = videos / f"{video['name']}.mp4"
        cap = cv2.VideoCapture(str(path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # annotation frames are frames of the video itself
        video_fps = video_fps if video_fps > 0 else 24.0  # XD-Violence videos are 24 frames/s
        cap.release()
        if video["stratum"] != "A" and video["name"] not in events_by_name:
            raise ValueError(f"{video['name']}: a violent test video without frame annotation")
        numbers = events_by_name.get(video["name"], [])
        events = [(numbers[i] / video_fps, numbers[i + 1] / video_fps) for i in range(0, len(numbers) - 1, 2)]

        def frames(path: Path = path) -> Iterator:
            with LatestFrameReader(path, mode="offline", target_fps=fps) as reader:
                yield from reader

        episodes.append(
            Episode(
                dataset="xd",
                part="test",
                episode_id=f"test/{video['name']}",
                label=int(video["stratum"] != "A"),
                duration_s=video["duration_s"],
                frames=frames,
                events=events,
                source=video["name"],
            )
        )
    return episodes


def extract_all(
    episodes: list[Episode], perception, out_dir: Path, meta: dict, redo: bool, log=print
) -> dict[str, int]:
    """Extract every episode; one that fails (e.g. an unreadable video) is logged and skipped, not fatal."""
    counts = {"done": 0, "skipped": 0, "failed": 0}
    hours = sum(e.duration_s for e in episodes) / 3600
    done_s, started = 0.0, time.monotonic()
    for i, episode in enumerate(episodes, 1):
        out = out_dir / episode.dataset / f"{episode.episode_id}.jsonl"
        try:
            made = extract_episode(episode, perception, out, meta, redo=redo)
            counts["done" if made else "skipped"] += 1
        except Exception as exc:  # KeyboardInterrupt / SystemExit stop the whole run
            counts["failed"] += 1
            log(f"FAILED {episode.dataset}/{episode.episode_id}: {type(exc).__name__}: {exc}")
        done_s += episode.duration_s
        if i % 20 == 0 or i == len(episodes):
            elapsed = time.monotonic() - started
            eta_h = (hours * 3600 - done_s) * elapsed / done_s / 3600 if done_s else math.nan
            log(
                f"{time.strftime('%H:%M:%S')} {episode.dataset} {i}/{len(episodes)}: {done_s / 3600:.2f} of "
                f"{hours:.2f} h of video in {elapsed / 3600:.2f} h, ETA {eta_h:.1f} h, {counts}"
            )
    return counts


def stage_meta(cfg: Config) -> dict:
    """Metadata every episode file of a stage carries: versions, perception hash, weight hashes, libraries."""
    import importlib.metadata as md

    import numpy as np

    from sentinel.models import sha256
    from sentinel.pipeline import PERCEPTION_VERSION

    def version(package: str) -> str | None:
        try:
            return md.version(package)
        except md.PackageNotFoundError:
            return None

    m = cfg.models
    return {
        "schema_version": SCHEMA_VERSION,
        "perception_version": PERCEPTION_VERSION,
        "perception_hash": perception_hash(cfg),
        "backend": m.backend,
        "analysis_fps": cfg.analysis.fps,
        "weights_sha256": {
            "people": sha256(m.people.weights),
            "pose": sha256(m.pose.weights),
            "weapons": sha256(cfg.weapons.weights),
        },
        "versions": {
            "ultralytics": version("ultralytics"),
            "openvino": version("openvino"),
            "torch": version("torch"),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }


# --- the command ---


def with_weapon_variant(config: Config, variant: str, conf: float | None = None) -> Config:
    """Weapon detector variant (e.g. crops320:2) and threshold for a separate cache (step 5 comparison)."""
    from sentinel.bench import WeaponVariant

    v = WeaponVariant.parse(variant)
    update = {"mode": v.mode, "imgsz": v.imgsz, "every_n_frames": v.every_n_frames}
    if conf is not None:
        update["conf"] = conf
    return config.model_copy(update={"weapons": config.weapons.model_copy(update=update)})


def dataset_config(config: Config, dataset: str) -> Config:
    """Sevilla is 2 frames/s of stills: analyse every frame and run the weapon detector on each of them."""
    if dataset != "us_gun":
        return config
    return config.model_copy(
        update={
            "analysis": config.analysis.model_copy(update={"fps": 2.0}),
            "weapons": config.weapons.model_copy(update={"every_n_frames": 1}),
        }
    )


def pick_cores(available: list[int], threads: int, cpus: list[int] | None) -> list[int]:
    """The given CPUs, or the last `threads` available ones (the efficiency cores of this laptop)."""
    if cpus:
        missing = set(cpus) - set(available)
        if missing:
            raise ValueError(f"CPUs {sorted(missing)} are not available to this process")
        return sorted(cpus)
    return sorted(available)[-threads:]


def _limit_cpu(threads: int, cpus: list[int] | None = None) -> list[int]:
    """Pin the process to a few CPUs (efficiency cores by default) and be as polite as possible."""
    cores = pick_cores(sorted(os.sched_getaffinity(0)), threads, cpus)
    os.sched_setaffinity(0, cores)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.nice(19)
    return cores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--plan", choices=["all"], help="every dataset in the order the experiment needs them"
    )
    parser.add_argument("--dataset", choices=["rwf2000", "us_gun", "xd"])
    parser.add_argument("--parts", nargs="+", help="rwf2000: train val test; us_gun: select test")
    parser.add_argument(
        "--per-class", type=int, help="rwf2000: a balanced sample of this many clips per class"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4, help="CPU cores (and model threads) to use")
    parser.add_argument(
        "--cpus", type=int, nargs="+", help="exact CPUs to pin to (default: the last --threads)"
    )
    parser.add_argument(
        "--backend", choices=["torch", "openvino"], help="default: models.backend of the config"
    )
    parser.add_argument("--weapons", help="weapon detector variant for a separate cache, e.g. crops320:2")
    parser.add_argument("--weapon-conf", type=float, help="weapon score threshold with --weapons")
    parser.add_argument(
        "--classes", nargs="+", choices=["Fight", "NonFight"], help="rwf2000: only these classes"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--redo", action="store_true", help="replace files made by another perception")
    args = parser.parse_args(argv)
    if not args.plan and not args.dataset:
        parser.error("give --plan all or --dataset")
    if args.dataset and not args.parts:
        parser.error("--dataset needs --parts (rwf2000: train val test; us_gun: select test; xd: test)")
    if args.weapons and args.out == OUT:
        parser.error("--weapons makes a separate cache: give it its own --out (e.g. data/episodes_crops)")

    import signal

    from sentinel.quiet import register_extraction, unregister_extraction

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # `systemctl stop` still runs the cleanup below
    cores = _limit_cpu(args.threads, args.cpus)
    config = load_config(args.config)
    if args.backend:
        config = config.model_copy(
            update={"models": config.models.model_copy(update={"backend": args.backend})}
        )
    if args.weapons:
        config = with_weapon_variant(config, args.weapons, args.weapon_conf)
    stages = (
        [
            ("rwf2000", ["train", "val"]),
            ("rwf2000", ["test"]),
            ("us_gun", ["select", "test"]),
            ("xd", ["test"]),
        ]
        if args.plan
        else [(args.dataset, args.parts)]
    )
    register_extraction()
    failed = 0
    try:
        for dataset, parts in stages:
            failed += _run_stage(config, dataset, parts, args, cores)
    finally:
        unregister_extraction()
    if failed:
        print(f"{failed} episodes failed (see FAILED lines above); run the same command again to retry them")
        raise SystemExit(1)


def _episodes(dataset: str, parts: list[str], args, fps: float) -> list[Episode]:
    if dataset == "rwf2000":
        episodes = rwf_episodes(
            Path("data/splits/rwf2000.json"), Path("data/raw/rwf2000"), parts, args.per_class, args.seed, fps
        )
        classes = {"Fight": 1, "NonFight": 0}
        wanted = {classes[c] for c in args.classes} if args.classes else {0, 1}
        return [e for e in episodes if e.label in wanted]
    if dataset == "us_gun":
        return sevilla_episodes(Path("data/splits/us_gun.json"), Path("data/raw/us-gun/Images"), parts)
    return xd_episodes(
        Path("data/splits/xd_subset.json"),
        Path("data/raw/xd-violence/videos"),
        Path("data/raw/xd-violence/annotations.txt"),
        fps,
    )


def _run_stage(config: Config, dataset: str, parts: list[str], args, cores: list[int]) -> int:
    """Extract one dataset stage; returns the number of failed episodes."""
    import torch

    from sentinel.agents.context import ZoneContext
    from sentinel.pipeline import Perception, build_agents

    cfg = dataset_config(config, dataset)
    episodes = _episodes(dataset, parts, args, cfg.analysis.fps)
    meta = stage_meta(cfg)
    todo = [e for e in episodes if not is_current(args.out / e.dataset / f"{e.episode_id}.jsonl", meta)]
    hours = sum(e.duration_s for e in todo) / 3600
    print(
        f"{dataset} {parts}: {len(episodes)} episodes, {len(todo)} to do ({hours:.2f} h of video); "
        f"cores {cores}; perception {meta['perception_version']} / {meta['perception_hash']}",
        flush=True,
    )
    if not todo:
        return 0
    zone = ZoneContext.from_yaml(cfg.context.zones).zone_for(cfg.source.camera_id)
    perception = Perception(build_agents(cfg, record_weapons=True), cfg, cfg.source.camera_id, zone)
    if cfg.models.backend == "torch":
        _warm = next(iter(todo[0].frames()))  # Ultralytics sets 8 threads when its predictor starts
        perception.reset()
        perception.process(_warm)
        torch.set_num_threads(len(cores))
    counts = extract_all(todo, perception, args.out, meta, args.redo, log=lambda m: print(m, flush=True))
    return counts["failed"]


if __name__ == "__main__":
    main()
