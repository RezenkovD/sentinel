"""Speed benchmark (SPEC, section 6): ms/frame for detection+tracking and pose, analysis rate, RTF.

Clips are read in offline mode and sampled on the analysis grid (8 or 10 frames/s), exactly the frames the
pipeline would analyse. Per-frame times exclude a separate warm-up pass. The measured RTF is the wall time of
the whole clip (decoding every source frame + analysing the sampled ones) divided by the clip duration.
"""

import json
import platform
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from sentinel.agents.detector import Track
from sentinel.agents.pose_features import MatchRules
from sentinel.io.source import LatestFrameReader


class Tracker(Protocol):
    def update(self, frame: np.ndarray) -> list[Track]: ...

    def reset(self) -> None: ...


class Poser(Protocol):
    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]: ...


@dataclass(frozen=True)
class PowerState:
    """Power source and profile during a run: on battery the same laptop is about twice as slow."""

    source: str  # "ac", "battery" or "unknown"
    profile: str  # power-profiles-daemon profile (performance / balanced / power-saver) or "unknown"

    def __str__(self) -> str:
        return f"{self.source}, {self.profile}"


UNKNOWN_POWER = PowerState("unknown", "unknown")


def read_power_state(
    supply_root: Path = Path("/sys/class/power_supply"),
    profile_cmd: Sequence[str] = ("powerprofilesctl", "get"),
) -> PowerState:
    source = "unknown"
    supplies = sorted(supply_root.iterdir()) if supply_root.is_dir() else []
    for supply in supplies:
        try:
            if (supply / "type").read_text().strip() == "Mains":
                source = "ac" if (supply / "online").read_text().strip() == "1" else "battery"
                break
        except OSError:
            continue
    try:
        done = subprocess.run(list(profile_cmd), capture_output=True, text=True, timeout=5, check=True)
        profile = done.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        profile = "unknown"
    return PowerState(source, profile)


@dataclass(frozen=True)
class Timing:
    mean_ms: float
    p95_ms: float


def timing(values_ms: Sequence[float]) -> Timing:
    return Timing(mean_ms=float(np.mean(values_ms)), p95_ms=float(np.percentile(values_ms, 95)))


@dataclass(frozen=True)
class Measurement:
    det: Timing  # detection + tracking
    pose: Timing  # pose model + attaching skeletons to tracks
    total: Timing
    frames: int  # measured (analysed) frames
    pose_coverage: float  # share of person tracks that got a skeleton
    rtf: dict[float, float]  # analysis fps -> measured real-time factor


@dataclass(frozen=True)
class BenchRow:
    model: str
    backend: str
    device: str
    imgsz: int
    threads: int | None
    measurement: Measurement
    clips: list[str] = field(default_factory=list)
    warmup: int = 10
    power: PowerState = UNKNOWN_POWER

    @property
    def max_fps(self) -> float:
        """Analysis rate reachable if frames are processed back to back."""
        return 1000.0 / self.measurement.total.mean_ms


def _warm_up(
    source: Path, tracker: Tracker, poser: Poser, frames: int, after: Callable[[], None] | None
) -> None:
    with LatestFrameReader(source, mode="offline") as reader:
        for _, frame in zip(range(frames), reader, strict=False):
            poser.estimate(frame.image, tracker.update(frame.image))
    if after is not None:
        after()


def run_measurement(
    sources: Sequence[Path],
    tracker: Tracker,
    poser: Poser,
    fps_list: Iterable[float],
    warmup: int,
    clock: Callable[[], float] = time.perf_counter,
    after_warmup: Callable[[], None] | None = None,
) -> Measurement:
    _warm_up(sources[0], tracker, poser, warmup, after_warmup)
    det_ms: list[float] = []
    pose_ms: list[float] = []
    n_tracks = n_skeletons = 0
    rtf: dict[float, float] = {}
    for fps in fps_list:
        wall = duration = 0.0
        for source in sources:
            tracker.reset()
            started = clock()
            with LatestFrameReader(source, mode="offline", target_fps=fps) as reader:
                for frame in reader:
                    t0 = clock()
                    tracks = tracker.update(frame.image)
                    t1 = clock()
                    skeletons = poser.estimate(frame.image, tracks)
                    t2 = clock()
                    det_ms.append((t1 - t0) * 1000)
                    pose_ms.append((t2 - t1) * 1000)
                    n_tracks += len(tracks)
                    n_skeletons += len(skeletons)
                wall += clock() - started
                duration += reader.stats.received / reader.source_fps
        rtf[fps] = wall / duration
    total_ms = [d + p for d, p in zip(det_ms, pose_ms, strict=True)]
    return Measurement(
        det=timing(det_ms),
        pose=timing(pose_ms),
        total=timing(total_ms),
        frames=len(total_ms),
        pose_coverage=n_skeletons / n_tracks if n_tracks else 0.0,
        rtf=rtf,
    )


@dataclass(frozen=True)
class PacedResult:
    """Real-time run: clips played at their own speed, analysis limited to target_fps (SPEC, section 6)."""

    target_fps: float
    delivered: int  # analysed frames
    received: int  # frames the reader got from the source
    wall_s: float
    duration_s: float  # video duration
    latency: Timing  # from the frame's appearance to the end of its analysis

    @property
    def effective_fps(self) -> float:
        return self.delivered / self.wall_s

    @property
    def skipped_share(self) -> float:
        return 1.0 - self.delivered / self.received


@dataclass(frozen=True)
class PacedRow:
    model: str
    backend: str
    device: str
    imgsz: int
    threads: int | None
    result: PacedResult
    clips: list[str] = field(default_factory=list)
    warmup: int = 10
    power: PowerState = UNKNOWN_POWER


def run_paced(
    sources: Sequence[Path],
    tracker: Tracker,
    poser: Poser,
    fps: float,
    warmup: int,
    after_warmup: Callable[[], None] | None = None,
) -> PacedResult:
    _warm_up(sources[0], tracker, poser, warmup, after_warmup)
    delivered = received = 0
    wall = duration = 0.0
    latency_ms: list[float] = []
    for source in sources:
        tracker.reset()
        with LatestFrameReader(source, mode="paced", target_fps=fps) as reader:
            started = time.monotonic()  # the reader starts its clock on the first read, right below
            for frame in reader:
                poser.estimate(frame.image, tracker.update(frame.image))
                latency_ms.append((time.monotonic() - started - frame.t) * 1000)
            wall += time.monotonic() - started
            stats = reader.stats
            delivered += stats.delivered
            received += stats.received
            duration += stats.received / reader.source_fps
    return PacedResult(fps, delivered, received, wall, duration, timing(latency_ms))


def render_table(rows: Sequence[BenchRow]) -> str:
    fps_keys = sorted({fps for row in rows for fps in row.measurement.rtf})
    header = (
        [
            "model",
            "backend",
            "device",
            "imgsz",
            "threads",
            "power",
            "det+track, ms",
            "pose, ms",
            "total, ms",
            "max fps",
        ]
        + [f"RTF@{fps:g}" for fps in fps_keys]
        + ["pose coverage", "frames"]
    )
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        m = row.measurement
        cells = [
            row.model,
            row.backend,
            row.device,
            str(row.imgsz),
            "—" if row.threads is None else str(row.threads),
            str(row.power),
            f"{m.det.mean_ms:.1f} / {m.det.p95_ms:.1f}",
            f"{m.pose.mean_ms:.1f} / {m.pose.p95_ms:.1f}",
            f"{m.total.mean_ms:.1f} / {m.total.p95_ms:.1f}",
            f"{row.max_fps:.1f}",
            *[f"{m.rtf[fps]:.2f}" if fps in m.rtf else "—" for fps in fps_keys],
            f"{m.pose_coverage:.0%}",
            str(m.frames),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_paced_table(rows: Sequence[PacedRow]) -> str:
    header = [
        "model",
        "backend",
        "imgsz",
        "threads",
        "power",
        "target fps",
        "effective fps",
        "analysed / received",
        "skipped",
        "latency, ms",
        "wall / video, s",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        r = row.result
        cells = [
            row.model,
            row.backend,
            str(row.imgsz),
            "—" if row.threads is None else str(row.threads),
            str(row.power),
            f"{r.target_fps:g}",
            f"{r.effective_fps:.1f}",
            f"{r.delivered} / {r.received}",
            f"{r.skipped_share:.0%}",
            f"{r.latency.mean_ms:.0f} / {r.latency.p95_ms:.0f}",
            f"{r.wall_s:.1f} / {r.duration_s:.1f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def paced_row_from_json(data: dict) -> PacedRow:
    r = data.pop("result")
    latency = Timing(**r.pop("latency"))
    power = PowerState(**data.pop("power")) if "power" in data else UNKNOWN_POWER
    return PacedRow(result=PacedResult(latency=latency, **r), power=power, **data)


def _append_jsonl(jsonl: Path, row: BenchRow | PacedRow) -> list[dict]:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return _read_jsonl(jsonl)


def _read_jsonl(jsonl: Path) -> list[dict]:
    if not jsonl.exists():
        return []
    return [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]


def row_from_json(data: dict) -> BenchRow:
    m = data.pop("measurement")
    measurement = Measurement(
        det=Timing(**m["det"]),
        pose=Timing(**m["pose"]),
        total=Timing(**m["total"]),
        frames=m["frames"],
        pose_coverage=m["pose_coverage"],
        rtf={float(k): v for k, v in m["rtf"].items()},
    )
    power = PowerState(**data.pop("power")) if "power" in data else UNKNOWN_POWER
    return BenchRow(measurement=measurement, power=power, **data)


def check_same_setup(logs: Sequence[Path], clips: list[str], warmup: int, power: PowerState) -> None:
    """Refuse to add a run to a report whose rows were measured on other clips, warm-up or power state."""
    for log in logs:
        for row in _read_jsonl(log):
            logged = (row.get("clips"), row.get("warmup", 10), row.get("power", asdict(UNKNOWN_POWER)))
            if logged != (clips, warmup, asdict(power)):
                raise ValueError(
                    f"{log} already holds rows measured on {logged[0]} with warm-up {logged[1]} and power "
                    f"{logged[2]}; mixing incomparable runs in one table — pass another --out"
                )


def append_row(jsonl: Path, row: BenchRow) -> list[BenchRow]:
    """Append a row to the JSONL log and return every row logged so far."""
    return [row_from_json(d) for d in _append_jsonl(jsonl, row)]


def append_paced_row(jsonl: Path, row: PacedRow) -> list[PacedRow]:
    """Append a real-time row to its JSONL log and return every row logged so far."""
    return [paced_row_from_json(d) for d in _append_jsonl(jsonl, row)]


def environment_notes() -> list[str]:
    import torch
    import ultralytics

    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    return [f"CPU: {cpu}", f"torch {torch.__version__}", f"ultralytics {ultralytics.__version__}"]


def clip_info(path: Path) -> str:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = 0  # count decodable frames: the header's frame count is wrong for truncated files
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return f"{path.name}: {w}×{h}, {fps:g} fps, {n / fps:.1f} s" if fps > 0 else f"{path.name}: {w}×{h}"


def render_report(
    rows: Sequence[BenchRow], paced: Sequence[PacedRow], sources: Sequence[Path], warmup: int
) -> str:
    lines = [
        f"# Bench {time.strftime('%Y-%m-%d')}",
        "",
        "Середовище: " + "; ".join(environment_notes()) + ".",
        "",
        "Кліпи останнього прогону:",
        *[f"- {clip_info(s)}" for s in sources],
        "",
        f"Прогрів: перші {warmup} кадрів (окремий прохід, у заміри не входить).",
        "",
        "## Швидкість моделей (offline)",
        "",
        "Кліпи читаються в режимі `offline` з вибіркою кадрів на сітці частоти аналізу. "
        "Час — мс/кадр, `середнє / p95`. `max fps` = 1000 / середній total. "
        "`RTF@f` — виміряний: весь час обробки кліпу (декодування всіх кадрів джерела + аналіз вибраних) "
        "/ тривалість відео. `pose coverage` — частка треків людей, до яких прив'язано скелет.",
        "",
        render_table(rows),
        "",
    ]
    if paced:
        lines += [
            "## Реальний час (paced)",
            "",
            "Кліпи відтворюються з власною швидкістю, як жива камера; читач тримає лише найсвіжіший кадр, "
            "аналіз — не частіше за `target fps`. `effective fps` — проаналізовано кадрів за секунду; "
            "`skipped` — частка кадрів джерела, що не аналізувалися; `latency` — від появи кадру до кінця "
            "його аналізу, мс (`середнє / p95`).",
            "",
            render_paced_table(paced),
            "",
        ]
    return "\n".join(lines)


def run_bench(
    sources: Sequence[Path],
    det_weights: Path,
    pose_weights: Path,
    backend: str,
    device: str,
    imgsz_list: Sequence[int],
    fps_list: Sequence[float],
    warmup: int,
    threads: int | None,
    person_conf: float,
    tracker_cfg: str,
    pose_conf: float,
    pose_rules: MatchRules,
    out: Path,
    paced: bool = False,
) -> Path:
    import torch

    from sentinel.agents.detector import PersonTracker
    from sentinel.agents.pose_features import PoseEstimator

    jsonl, paced_jsonl = out.with_suffix(".jsonl"), out.with_suffix(".paced.jsonl")
    clips = [s.name for s in sources]
    power = read_power_state()
    check_same_setup([jsonl, paced_jsonl], clips, warmup, power)

    model = (
        det_weights.stem
        if pose_weights.stem == f"{det_weights.stem}-pose"
        else f"{det_weights.stem}+{pose_weights.stem}"
    )
    label = backend
    if backend == "openvino":
        import openvino

        label = f"openvino {openvino.__version__.split('-')[0]}"
        from sentinel.models import resolve_weights

        det_weights, pose_weights = (
            resolve_weights(det_weights, backend),
            resolve_weights(pose_weights, backend),
        )

    def after_warmup() -> None:
        # Ultralytics resets torch threads when a predictor is set up (during warm-up), so set them after.
        if backend == "torch" and threads is not None:
            torch.set_num_threads(threads)

    def threads_used() -> int | None:
        return torch.get_num_threads() if backend == "torch" else None

    for imgsz in imgsz_list:
        for paced_fps in fps_list if paced else [None]:
            tracker = PersonTracker(
                det_weights, imgsz=imgsz, conf=person_conf, tracker=tracker_cfg, device=device
            )
            poser = PoseEstimator(pose_weights, imgsz=imgsz, conf=pose_conf, rules=pose_rules, device=device)
            if paced_fps is None:
                m = run_measurement(sources, tracker, poser, fps_list, warmup, after_warmup=after_warmup)
                row = BenchRow(model, label, device, imgsz, threads_used(), m, clips, warmup, power)
                append_row(jsonl, row)
                print(render_table([row]), flush=True)
            else:
                r = run_paced(sources, tracker, poser, paced_fps, warmup, after_warmup=after_warmup)
                paced_row = PacedRow(model, label, device, imgsz, threads_used(), r, clips, warmup, power)
                append_paced_row(paced_jsonl, paced_row)
                print(render_paced_table([paced_row]), flush=True)
            rows = [row_from_json(d) for d in _read_jsonl(jsonl)]
            paced_rows = [paced_row_from_json(d) for d in _read_jsonl(paced_jsonl)]
            out.write_text(render_report(rows, paced_rows, sources, warmup), encoding="utf-8")
    return out


# --- full pipeline (session 02, step 7) ---


@dataclass(frozen=True)
class WeaponVariant:
    mode: str  # "frame" or "crops"
    imgsz: int
    every_n_frames: int

    @classmethod
    def parse(cls, text: str) -> "WeaponVariant":
        """ "480:1", "640:2" (whole frame) or "crops320:2" (person crops): input size : every n-th frame."""
        size, sep, every = text.partition(":")
        if not sep or not every.isdigit():
            raise ValueError(f"weapon variant must look like 640:1 or crops320:2, got {text!r}")
        mode = "crops" if size.startswith("crops") else "frame"
        return cls(mode, int(size.removeprefix("crops")), int(every))

    @property
    def label(self) -> str:
        return f"{'crops' if self.mode == 'crops' else ''}{self.imgsz}:{self.every_n_frames}"


class _NoNotifier:
    def on_event(self, event: object) -> None:
        pass

    def close(self) -> None:
        pass


def measure_pipeline(
    cfg, sources: Sequence[Path], warmup: int, during: Callable[[], None] | None = None
) -> dict:
    """One configuration: warm-up, an offline pass (stage times, RTF) and a paced pass (like a live camera).

    `during` is called after every processed frame (e.g. to sample the machine load).
    """
    from sentinel.pipeline import build_pipeline

    fps = cfg.analysis.fps
    tick = during or (lambda: None)
    pipeline = build_pipeline(cfg, cfg.source.camera_id, "bench", None, _NoNotifier())
    with LatestFrameReader(sources[0], mode="offline") as reader:
        for _, frame in zip(range(warmup), reader, strict=False):
            pipeline.process(frame)
    stages: dict[str, list[float]] = {}
    totals: list[float] = []
    wall = duration = 0.0
    for source in sources:  # offline: every frame on the analysis grid, decoding included
        pipeline.reset(run_id="bench")
        started = time.perf_counter()
        with LatestFrameReader(source, mode="offline", target_fps=fps) as reader:
            for frame in reader:
                pipeline.process(frame)
                for stage, ms in pipeline.last_timing_ms.items():
                    stages.setdefault(stage, []).append(ms)
                totals.append(sum(pipeline.last_timing_ms.values()))
                tick()
            pipeline.finish()
            wall += time.perf_counter() - started
            duration += reader.stats.received / reader.source_fps
    latency: list[float] = []
    delivered = received = 0
    paced_wall = 0.0
    for source in sources:  # paced: like a live camera
        pipeline.reset(run_id="bench")
        with LatestFrameReader(source, mode="paced", target_fps=fps) as reader:
            started = time.monotonic()
            for frame in reader:
                pipeline.process(frame)
                latency.append((time.monotonic() - started - frame.t) * 1000)
                tick()
            pipeline.finish()
            paced_wall += time.monotonic() - started
            delivered += reader.stats.delivered
            received += reader.stats.received
    return {
        "stages_ms": {k: asdict(timing(v)) for k, v in stages.items()},
        "total_ms": asdict(timing(totals)),
        "rtf": wall / duration,
        "paced": {
            "effective_fps": delivered / paced_wall,
            "skipped_share": 1 - delivered / received,
            "latency_ms": asdict(timing(latency)),
        },
        "frames": len(totals),
    }


def with_weapons(config, variant: "WeaponVariant", out: Path):
    weapons = config.weapons.model_copy(
        update={"mode": variant.mode, "imgsz": variant.imgsz, "every_n_frames": variant.every_n_frames}
    )
    incidents = config.incidents.model_copy(update={"output_dir": out.parent / f"{out.stem}_keyframes"})
    return config.model_copy(update={"weapons": weapons, "incidents": incidents})


def run_pipeline_bench(
    config, sources: Sequence[Path], variants: Sequence[WeaponVariant], warmup: int, out: Path
) -> Path:
    """Full pipeline per weapon-detector variant: offline RTF + stage times, then a paced pass."""
    power = read_power_state()
    rows = []
    for variant in variants:
        rows.append(
            {"variant": variant.label} | measure_pipeline(with_weapons(config, variant, out), sources, warmup)
        )
        print(json.dumps(rows[-1]), flush=True)
    report = {
        "power": asdict(power),
        "device": config.models.runtime_device,
        "fps": config.analysis.fps,
        "clips": [s.name for s in sources],
        "warmup": warmup,
        "rows": rows,
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    out.write_text(render_pipeline_report(report, sources), encoding="utf-8")
    return out


def render_pipeline_report(report: dict, sources: Sequence[Path]) -> str:
    fps = report["fps"]
    lines = [
        f"# Bench: повний конвеєр — {time.strftime('%Y-%m-%d')}",
        "",
        "Середовище: " + "; ".join(environment_notes()) + f"; живлення: {PowerState(**report['power'])}.",
        f"Бекенд: {report.get('device', 'cpu')} (cpu = PyTorch, intel:cpu = OpenVINO); "
        f"частота аналізу {fps:g} кадрів/с; прогрів {report['warmup']} кадрів; кліпів {len(sources)}:",
        *[f"- {clip_info(s)}" for s in sources],
        "",
        "Конвеєр: люди + трек (YOLO26n) → поза (YOLO26n-pose) → зброя (варіант) → ознаки й опис сцени → "
        "A → автомат інцидентів. Мс/кадр — `середнє / p95`. `RTF` — offline: весь час обробки "
        "(з декодуванням) / тривалість відео. Paced — кліпи в темпі камери: ефективна частота, частка "
        "пропущених кадрів джерела, затримка від появи кадру до кінця його обробки.",
        "",
        "| зброя | люди | поза | зброя, мс | сцена | рішення | разом, мс | RTF@"
        + f"{fps:g}"
        + " | paced: кадр/с | пропущено | затримка, мс |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def cell(stages: dict, k: str) -> str:
        return f"{stages[k]['mean_ms']:.1f} / {stages[k]['p95_ms']:.1f}"

    for row in report["rows"]:
        s = row["stages_ms"]
        p = row["paced"]
        lines.append(
            f"| {row['variant']} | {cell(s, 'people')} | {cell(s, 'pose')} | {cell(s, 'weapons')} | "
            f"{cell(s, 'scene')} | {cell(s, 'decision')} | "
            f"{row['total_ms']['mean_ms']:.1f} / {row['total_ms']['p95_ms']:.1f} | "
            f"{row['rtf']:.2f} | {p['effective_fps']:.1f} | {p['skipped_share']:.0%} | "
            f"{p['latency_ms']['mean_ms']:.0f} / {p['latency_ms']['p95_ms']:.0f} |"
        )
    return "\n".join(lines) + "\n"


# --- quiet-window bench (session 03, step 1) ---


@dataclass(frozen=True)
class PeopleVariant:
    source: str  # "detect" (detector + pose model) or "pose" (the pose model tracked itself)
    weights: Path  # the detector, or the pose model for source "pose"

    @classmethod
    def parse(cls, text: str) -> "PeopleVariant":
        """ "detect:weights/yolo11n.pt" or "pose:weights/yolo26n-pose.pt"."""
        source, sep, weights = text.partition(":")
        if not sep or source not in ("detect", "pose") or not weights:
            raise ValueError(
                f"people variant must look like detect:<weights> or pose:<weights>, got {text!r}"
            )
        return cls(source, Path(weights))

    @property
    def label(self) -> str:
        return f"{self.weights.stem} {'track' if self.source == 'pose' else 'detect'}"


def with_people(config, variant: PeopleVariant):
    m = config.models
    people = m.people.model_copy(update={"source": variant.source})
    pose = m.pose
    if variant.source == "detect":
        people = people.model_copy(update={"weights": variant.weights})
    else:
        pose = pose.model_copy(update={"weights": variant.weights})
    return config.model_copy(update={"models": m.model_copy(update={"people": people, "pose": pose})})


def with_backend(config, backend: str):
    return config.model_copy(update={"models": config.models.model_copy(update={"backend": backend})})


def _cell(stage: dict) -> str:
    return f"{stage['mean_ms']:.1f} / {stage['p95_ms']:.1f}"


def render_quiet_report(rows: Sequence[dict], sources: Sequence[Path]) -> str:
    """Latest clean (not noisy) measurement per combination; noisy attempts are only counted."""
    latest: dict[tuple, dict] = {}
    for row in rows:
        if not row["noisy"]:
            latest[(row["people"], row["weapons"], row["backend"])] = row
    noisy = sum(row["noisy"] for row in rows)
    lines = [
        f"# Bench: тихий замір повного конвеєра — {time.strftime('%Y-%m-%d')}",
        "",
        "Середовище: " + "; ".join(environment_notes()) + ".",
        f"Кліпів {len(sources)}; частота аналізу 10 кадрів/с. Замір стартує, коли 1-хвилинне середнє чужого "
        "навантаження (задачі, що виконуються або чекають введення-виведення, без наших процесів) < 1,5 "
        "протягом 2 хв, від мережі, профіль `performance`; прогін, під час якого воно перевищило 2, — шумний "
        "і повторюється. Наше фонове вилучення на час заміру призупинено.",
        f"Шумних прогонів: {noisy}.",
        "",
        "| джерело людей | зброя | бекенд | люди, мс | поза, мс | зброя, мс | разом, мс | RTF@10 | "
        "paced: кадр/с | пропущено | затримка p95, мс | чуже навантаження сер. / макс. | load avg | час |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (people, weapons, backend), row in sorted(latest.items()):
        s, p = row["stages_ms"], row["paced"]
        lines.append(
            f"| {people} | {weapons} | {backend} | {_cell(s['people'])} | {_cell(s['pose'])} | "
            f"{_cell(s['weapons'])} | {_cell(row['total_ms'])} | {row['rtf']:.2f} | "
            f"{p['effective_fps']:.1f} | {p['skipped_share']:.0%} | {p['latency_ms']['p95_ms']:.0f} | "
            f"{row['foreign_load_mean']:.2f} / {row['foreign_load_max']:.2f} | "
            f"{row['load_start']:.2f} → {row['load_end']:.2f} | {row['time'][11:16]} |"
        )
    lines += [
        "",
        "`load avg` — системний 1-хвилинний load average на початку й наприкінці прогону (з нашим заміром).",
        "Мс/кадр — `середнє / p95`; RTF — offline (з декодуванням); paced — кліпи в темпі камери.",
    ]
    return "\n".join(lines) + "\n"


def run_quiet_bench(
    config,
    sources: Sequence[Path],
    people: Sequence[PeopleVariant],
    weapons: Sequence[WeaponVariant],
    backends: Sequence[str],
    warmup: int,
    out: Path,
    deadline_h: float,
) -> Path:
    """Every people source x weapon variant x backend, each measured in its own quiet window."""
    import os
    from datetime import datetime

    import sentinel.agents.context  # noqa: F401  (imported now: the code measured is the code at launch)
    import sentinel.agents.detector  # noqa: F401
    import sentinel.agents.features  # noqa: F401
    import sentinel.agents.pose_features  # noqa: F401
    import sentinel.agents.weapons  # noqa: F401
    import sentinel.aggregators.rules  # noqa: F401
    import sentinel.incidents  # noqa: F401
    import sentinel.keyframe  # noqa: F401
    import sentinel.pipeline  # noqa: F401
    import sentinel.scene  # noqa: F401
    from sentinel.models import export_openvino
    from sentinel.quiet import (
        QuietRunner,
        extraction_pids,
        foreign_load,
        machine_load_average,
        pause_extraction,
        resume_extraction,
    )

    if "openvino" in backends:
        for pt in {config.weapons.weights, config.models.pose.weights, *(v.weights for v in people)}:
            export_openvino(pt)
    resume_extraction()  # a run killed mid-measurement may have left the extraction paused
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    jsonl = out.with_suffix(".jsonl")

    def own() -> set[int]:
        return {os.getpid(), *extraction_pids()}

    def measure(combo: tuple, during: Callable[[], None]) -> dict:
        backend, people_variant, weapon_variant = combo
        cfg = with_backend(with_people(with_weapons(config, weapon_variant, out), people_variant), backend)
        last = [0.0]

        def tick() -> None:  # sample the load about once a second
            if time.monotonic() - last[0] >= 1.0:
                last[0] = time.monotonic()
                during()

        load_start = machine_load_average()
        row = measure_pipeline(cfg, sources, warmup, tick)
        return row | {
            "backend": backend,
            "people": people_variant.label,
            "weapons": weapon_variant.label,
            "power": asdict(read_power_state()),
            "load_start": load_start,
            "load_end": machine_load_average(),
            "time": datetime.now().isoformat(timespec="seconds"),
            "commit": commit,
            "clips": [s.name for s in sources],
        }

    def save(row: dict) -> None:
        _append_jsonl_row(jsonl, row)
        out.write_text(render_quiet_report(_read_jsonl(jsonl), sources), encoding="utf-8")

    def power() -> tuple[bool, str]:
        state = read_power_state()
        return state.source == "ac", state.profile

    runner = QuietRunner(
        sample=lambda: foreign_load(own),
        measure=measure,
        power=power,
        pause=pause_extraction,
        resume=resume_extraction,
        log=lambda message: print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True),
        on_row=save,
    )
    combos = [(b, p, w) for p in people for w in weapons for b in backends]  # the first people source first
    runner.run(combos, deadline_s=deadline_h * 3600)
    return out


def _append_jsonl_row(jsonl: Path, row: dict) -> None:
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
