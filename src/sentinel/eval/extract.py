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
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from sentinel.config import DEFAULT_CONFIG, Config, load_config
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


def perception_hash(config: Config) -> str:
    """Hash of everything that shapes the scene descriptions: perception settings, zone and tracker files,
    model checkpoints, schema version. Aggregator and incident settings are downstream and not included."""
    from sentinel.models import sha256

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
    import hashlib

    return hashlib.sha256(json.dumps(digest, sort_keys=True).encode()).hexdigest()[:16]


# --- writing ---


def _labels_of(episode: Episode, scene: SceneDescription, frame_times: list[float]) -> int | None:
    if episode.events is not None:
        return window_label(scene.window.t_start, scene.window.t_end, episode.events)
    if episode.frame_labels is not None:
        return max((episode.frame_labels.get(t, 0) for t in frame_times), default=0)
    return None


def extract_episode(episode: Episode, perception, out: Path, meta: dict) -> bool:
    """Run perception over the episode and write its file; False if an identical file already exists."""
    if out.exists():
        with out.open(encoding="utf-8") as f:
            existing = json.loads(f.readline())["meta"]
        same = ("perception_version", "perception_hash")
        if all(existing.get(k) == meta[k] for k in same):
            return False
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


def _limit_cpu(threads: int) -> list[int]:
    """Pin the process to the last `threads` CPUs (efficiency cores here) and be as polite as possible."""
    cores = sorted(os.sched_getaffinity(0))[-threads:]
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

    from sentinel.quiet import EXTRACT_PIDFILE

    cores = _limit_cpu(args.threads)
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
        else [(args.dataset, args.parts or ["test"])]
    )
    EXTRACT_PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    EXTRACT_PIDFILE.write_text(str(os.getpid()))
    try:
        for dataset, parts in stages:
            _run_stage(config, dataset, parts, args, cores)
    finally:
        EXTRACT_PIDFILE.unlink(missing_ok=True)


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


def _run_stage(config: Config, dataset: str, parts: list[str], args, cores: list[int]) -> None:
    import torch

    from sentinel.agents.context import ZoneContext
    from sentinel.pipeline import PERCEPTION_VERSION, Perception, build_agents

    cfg = dataset_config(config, dataset)
    episodes = _episodes(dataset, parts, args, cfg.analysis.fps)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "perception_version": PERCEPTION_VERSION,
        "perception_hash": perception_hash(cfg),
        "backend": cfg.models.backend,
        "analysis_fps": cfg.analysis.fps,
    }
    todo = [e for e in episodes if not (args.out / e.dataset / f"{e.episode_id}.jsonl").exists() or args.redo]
    hours = sum(e.duration_s for e in todo) / 3600
    print(
        f"{dataset} {parts}: {len(episodes)} episodes, {len(todo)} to do ({hours:.2f} h of video); "
        f"cores {cores}; perception {PERCEPTION_VERSION} / {meta['perception_hash']}",
        flush=True,
    )
    if not todo:
        return
    zone = ZoneContext.from_yaml(cfg.context.zones).zone_for(cfg.source.camera_id)
    perception = Perception(build_agents(cfg, record_weapons=True), cfg, cfg.source.camera_id, zone)
    done_s = 0.0
    started = time.monotonic()
    for i, episode in enumerate(todo, 1):
        out = args.out / episode.dataset / f"{episode.episode_id}.jsonl"
        if args.redo:
            out.unlink(missing_ok=True)
        extract_episode(episode, perception, out, meta)
        if cfg.models.backend == "torch":
            torch.set_num_threads(len(cores))  # Ultralytics sets 8 threads when its predictor starts
        done_s += episode.duration_s
        elapsed = time.monotonic() - started
        eta_h = (hours * 3600 - done_s) * elapsed / done_s / 3600 if done_s else math.nan
        if i % 20 == 0 or i == len(todo):
            print(
                f"{time.strftime('%H:%M:%S')} {dataset} {i}/{len(todo)}: {done_s / 3600:.2f} of "
                f"{hours:.2f} h of video in {elapsed / 3600:.2f} h, ETA {eta_h:.1f} h",
                flush=True,
            )


if __name__ == "__main__":
    main()
