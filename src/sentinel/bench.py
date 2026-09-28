"""Speed benchmark (SPEC, section 6): ms/frame for detection+tracking and pose, analysis rate, RTF.

Clips are read in offline mode and sampled on the analysis grid (8 or 10 frames/s), exactly the frames the
pipeline would analyse. Per-frame times exclude a separate warm-up pass. The measured RTF is the wall time of
the whole clip (decoding every source frame + analysing the sampled ones) divided by the clip duration.
"""

import json
import platform
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from sentinel.agents.detector import Track
from sentinel.io.source import LatestFrameReader


class Tracker(Protocol):
    def update(self, frame: np.ndarray) -> list[Track]: ...

    def reset(self) -> None: ...


class Poser(Protocol):
    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]: ...


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

    @property
    def max_fps(self) -> float:
        """Analysis rate reachable if frames are processed back to back."""
        return 1000.0 / self.measurement.total.mean_ms

    def model_rtf(self, fps: float) -> float:
        """RTF from model time alone (no decoding): fps × mean ms per frame."""
        return fps * self.measurement.total.mean_ms / 1000.0


def run_measurement(
    sources: Sequence[Path],
    tracker: Tracker,
    poser: Poser,
    fps_list: Iterable[float],
    warmup: int,
    clock: Callable[[], float] = time.perf_counter,
    after_warmup: Callable[[], None] | None = None,
) -> Measurement:
    with LatestFrameReader(sources[0], mode="offline") as reader:
        for _, frame in zip(range(warmup), reader, strict=False):
            poser.estimate(frame.image, tracker.update(frame.image))
    if after_warmup is not None:
        after_warmup()

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


def render_table(rows: Sequence[BenchRow]) -> str:
    fps_keys = sorted({fps for row in rows for fps in row.measurement.rtf})
    header = (
        [
            "model",
            "backend",
            "device",
            "imgsz",
            "threads",
            "det+track, ms",
            "pose, ms",
            "total, ms",
            "max fps",
        ]
        + [f"RTF@{fps:g}" for fps in fps_keys]
        + [f"model RTF@{'/'.join(f'{fps:g}' for fps in fps_keys)}", "pose coverage", "frames"]
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
            f"{m.det.mean_ms:.1f} / {m.det.p95_ms:.1f}",
            f"{m.pose.mean_ms:.1f} / {m.pose.p95_ms:.1f}",
            f"{m.total.mean_ms:.1f} / {m.total.p95_ms:.1f}",
            f"{row.max_fps:.1f}",
            *[f"{m.rtf[fps]:.2f}" if fps in m.rtf else "—" for fps in fps_keys],
            " / ".join(f"{row.model_rtf(fps):.2f}" for fps in fps_keys),
            f"{m.pose_coverage:.0%}",
            str(m.frames),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    return BenchRow(measurement=measurement, **data)


def append_row(jsonl: Path, row: BenchRow) -> list[BenchRow]:
    """Append a row to the JSONL log and return every row logged so far."""
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return [row_from_json(json.loads(line)) for line in jsonl.read_text(encoding="utf-8").splitlines()]


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
    notes = [f"CPU: {cpu}", f"torch {torch.__version__}", f"ultralytics {ultralytics.__version__}"]
    try:
        import openvino

        notes.append(f"openvino {openvino.__version__}")
    except ImportError:
        pass
    return notes


def clip_info(path: Path) -> str:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return f"{path.name}: {w}×{h}, {fps:g} fps, {n / fps:.1f} s" if fps > 0 else f"{path.name}: {w}×{h}"


def render_report(rows: Sequence[BenchRow], sources: Sequence[Path], warmup: int) -> str:
    return "\n".join(
        [
            f"# Bench {time.strftime('%Y-%m-%d')}",
            "",
            "Середовище: " + "; ".join(environment_notes()) + ".",
            "",
            "Кліпи останнього прогону:",
            *[f"- {clip_info(s)}" for s in sources],
            "",
            f"Прогрів: перші {warmup} кадрів (окремий прохід, у заміри не входить). "
            "Кліпи читаються в режимі `offline` з вибіркою кадрів на сітці частоти аналізу. "
            "Час — мс/кадр, `середнє / p95`.",
            "`max fps` = 1000 / середній total. "
            "`RTF@f` — виміряний: весь час обробки кліпу (декодування всіх кадрів джерела + аналіз вибраних) "
            "/ тривалість відео. "
            "`model RTF` = f × середній total / 1000 (лише моделі, без декодування). "
            "`pose coverage` — частка треків людей, до яких прив'язано скелет.",
            "",
            render_table(rows),
            "",
        ]
    )


def export_openvino(weights: Path) -> Path:
    """Export an Ultralytics .pt checkpoint to OpenVINO IR once (dynamic input shape, so any imgsz works)."""
    from ultralytics import YOLO

    out = weights.with_name(f"{weights.stem}_openvino_model")
    if not (out / f"{weights.stem}.xml").exists():
        YOLO(str(weights)).export(format="openvino", dynamic=True, half=False)
    return out


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
    min_iou: float,
    out: Path,
) -> Path:
    import torch

    from sentinel.agents.detector import PersonTracker
    from sentinel.agents.pose_features import PoseEstimator

    model = (
        det_weights.stem
        if pose_weights.stem == f"{det_weights.stem}-pose"
        else f"{det_weights.stem}+{pose_weights.stem}"
    )
    if backend == "openvino":
        det_weights, pose_weights = export_openvino(det_weights), export_openvino(pose_weights)

    def after_warmup() -> None:
        # Ultralytics resets torch threads when a predictor is set up (during warm-up), so set them after.
        if backend == "torch" and threads is not None:
            torch.set_num_threads(threads)

    jsonl = out.with_suffix(".jsonl")
    for imgsz in imgsz_list:
        tracker = PersonTracker(
            det_weights, imgsz=imgsz, conf=person_conf, tracker=tracker_cfg, device=device
        )
        poser = PoseEstimator(pose_weights, imgsz=imgsz, conf=pose_conf, min_iou=min_iou, device=device)
        m = run_measurement(sources, tracker, poser, fps_list, warmup, after_warmup=after_warmup)
        threads_used = torch.get_num_threads() if backend == "torch" else None
        row = BenchRow(model, backend, device, imgsz, threads_used, m, clips=[s.name for s in sources])
        rows = append_row(jsonl, row)
        out.write_text(render_report(rows, sources, warmup), encoding="utf-8")
        print(render_table([row]), flush=True)
    return out
