"""Trials protocol (session 06, step 2): a clean-container run recording what was tested and how it behaved.

    scripts/protocol.sh        # on the host: this in the Docker image, the protocol copied to docs/reports/
    sentinel eval protocol --out results/protocol_<date>.md   # inside the container

Steps: versions and hashes (image, libraries, models, config, A); the test suite (pytest); then a few test
clips per scenario — fights, a weapon (shooting), calm — each run twice through the real pipeline: offline
(RTF, ms/frame) and paced like a live camera (analysis rate, skipped frames, latency, incidents and their
messages). Clips come from the test parts only and are fixed below; nothing is tuned on them. Notifications
go to the console unless `--telegram` is given (then the bot set up in .env gets the real messages).
"""

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, Config, load_config

SEED = 2026
COLUMNS = (
    "Сценарій",
    "Кліп",
    "Тривалість, с",
    "RTF",
    "мс/кадр (сер. / p95)",
    "Аналіз, кадр/с",
    "Пропущено",
    "Затримка p50 / p95, мс",
    "Інцидентів",
    "Очікувано",
    "Відповідає",
)
XD = Path("data/raw/xd-violence/videos")
# (scenario, clip, an incident is expected); RWF clips are drawn from the test part with the seed
XD_CLIPS = (
    ("бійка", XD / "Rush.Hour.3.2007.BluRay__#01-12-16_01-12-36_label_B1-0-0.mp4", True),
    ("зброя", XD / "Mission.Impossible.Ghost.Protocol.2011__#00-01-18_00-01-40_label_B2-0-0.mp4", True),
    ("зброя", XD / "City.Of.Men.2007__#01-35-30_01-36-00_label_B2-0-0.mp4", True),
    ("спокій", XD / "v=GnN9VTh_s2A__#1_label_A.mp4", False),
)


def protocol_clips(
    splits: Path = Path("data/splits/rwf2000.json"), per_class: int = 3, chain: int = 3
) -> list[tuple[str, list[Path], bool]]:
    """(scenario, clips played back to back, an incident is expected). An RWF clip lasts 5 s — too short for
    the automaton (k of n windows), so each RWF case is the first `chain` test segments of one source video,
    15 s of the same scene; sources are drawn with the seed (trial of 04.10: single 5-s fights opened
    nothing)."""
    import re

    test = json.loads(splits.read_text(encoding="utf-8"))["test"]
    rng = random.Random(SEED)
    rwf = Path("data/raw/rwf2000")

    def chains(kind: str) -> list[list[Path]]:
        by: dict[str, list[tuple[int, str]]] = {}
        for path in test:
            if m := re.search(rf"/{kind}/(.+)_(\d+)\.avi$", path):
                by.setdefault(m.group(1), []).append((int(m.group(2)), path))
        sources = sorted(k for k, v in by.items() if len(v) >= chain)
        return [[rwf / p for _, p in sorted(by[k])[:chain]] for k in rng.sample(sources, per_class)]

    return (
        [("бійка", c, True) for c in chains("Fight")]
        + [(XD_CLIPS[0][0], [XD_CLIPS[0][1]], XD_CLIPS[0][2])]
        + [(s, [c], e) for s, c, e in XD_CLIPS[1:3]]
        + [("спокій", c, False) for c in chains("NonFight")]
        + [(XD_CLIPS[3][0], [XD_CLIPS[3][1]], XD_CLIPS[3][2])]
    )


def joined(clips: list[Path], folder: Path) -> Path:
    """One clip, or several played back to back as one video (the first clip's size and frame rate), written
    losslessly (FFV1): a lossy re-encode changes the motion the pipeline sees (trial 2 of 04.10)."""
    import cv2

    if len(clips) == 1:
        return clips[0]
    out = folder / f"{clips[0].stem.rsplit('_', 1)[0]}_x{len(clips)}.mkv"
    writer = None
    for clip in clips:
        capture = cv2.VideoCapture(str(clip))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                size = frame.shape[1::-1]
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
            writer.write(cv2.resize(frame, size))
        capture.release()
    writer.release()
    return out


def environment(config: Config) -> dict:
    from sentinel.eval.experiment import config_digest, sha256
    from sentinel.pipeline import PERCEPTION_VERSION, perception_hash

    versions = {}
    for name in ("ultralytics", "openvino", "torch", "cv2", "numpy", "sklearn", "pydantic"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing library is reported, not fatal
            versions[name] = "—"
    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if "model name" in line
        ),
        platform.processor(),
    )
    return {
        "image": os.environ.get("SENTINEL_IMAGE", "— (не в Docker або не передано)"),
        "image_created": os.environ.get("SENTINEL_IMAGE_CREATED", "—"),
        "git": os.environ.get("SENTINEL_GIT", "—"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu,
        "cpus": os.cpu_count(),
        "versions": versions,
        "backend": config.models.backend,
        "perception": f"{PERCEPTION_VERSION}:{perception_hash(config)}",
        "config_sha256": sha256(DEFAULT_CONFIG),
        "frozen_config_sha256": config_digest(config),
        "a_sha256": sha256(Path(config.aggregator_a.model)),
        "weights_sha256": {
            "people": sha256(Path(config.models.people.weights)),
            "pose": sha256(Path(config.models.pose.weights)),
            "weapons": sha256(Path(config.weapons.weights)),
        },
    }


def run_tests() -> dict:
    started = time.monotonic()
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    return {
        "ok": done.returncode == 0,
        "summary": lines[-1] if lines else "",
        "s": round(time.monotonic() - started, 1),
    }


class Collect:
    """Keeps every event and passes it on (to the console or the bot)."""

    def __init__(self, inner) -> None:
        self.inner, self.events = inner, []

    def on_event(self, event) -> None:
        self.events.append(event)
        self.inner.on_event(event)

    def close(self) -> None:
        pass


def run_clip(config: Config, clip: Path, notifier, origin: float) -> dict:
    """The clip offline (RTF, ms/frame), then paced (rate, skipped share, latency, incidents)."""
    from sentinel.io.source import LatestFrameReader
    from sentinel.notify import caption
    from sentinel.pipeline import build_pipeline

    fps = config.analysis.fps
    collect = Collect(notifier)
    pipeline = build_pipeline(config, config.source.camera_id, "protocol", None, collect)
    totals, started = [], time.perf_counter()
    with LatestFrameReader(clip, mode="offline", target_fps=fps) as reader:
        for frame in reader:
            pipeline.process(frame)
            totals.append(sum(pipeline.last_timing_ms.values()))
        pipeline.finish()
        duration = reader.stats.received / reader.source_fps
    rtf = (time.perf_counter() - started) / duration
    collect.events.clear()
    pipeline.reset(run_id="protocol")
    latency = []
    with LatestFrameReader(clip, mode="paced", target_fps=fps) as reader:
        began = time.monotonic()
        for frame in reader:
            pipeline.process(frame)
            latency.append((time.monotonic() - began - frame.t) * 1000)
        pipeline.finish()
        wall = time.monotonic() - began
        delivered, received = reader.stats.delivered, reader.stats.received
    events = list(collect.events)
    return {
        "duration_s": round(duration, 1),
        "rtf": round(rtf, 2),
        "ms_mean": round(float(np.mean(totals)), 1),
        "ms_p95": round(float(np.percentile(totals, 95)), 1),
        "analysis_fps": round(delivered / wall, 1),
        "skipped_share": round(1 - delivered / received, 3) if received else None,
        "latency_p50_ms": round(float(np.percentile(latency, 50)), 0),
        "latency_p95_ms": round(float(np.percentile(latency, 95)), 0),
        "incidents": sum(e.kind == "open" for e in events),
        "messages": [
            caption(e, config.source.label, origin) for e in events if e.kind in ("open", "escalate", "close")
        ],
    }


def _n(value, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def render(env: dict, tests: dict, rows: list[dict], started: str, telegram: bool) -> str:
    ok = [r for r in rows if (r["incidents"] > 0) == r["expected"]]
    lines = [
        f"# Протокол випробувань програмного продукту Sentinel — {date.today().strftime('%d.%m.%Y')}",
        "",
        "## 1. Мета",
        "",
        "Перевірити, що програмний продукт, зібраний у Docker-образ із зафіксованими версіями, з чистого "
        "середовища проходить автоматичні тести, обробляє відео в реальному часі та створює інциденти з "
        "повідомленнями для оператора у сценаріях «бійка», «зброя» й «спокій».",
        "",
        "## 2. Умови",
        "",
        f"- Дата й час початку: {started}.",
        f"- Процесор: {env['cpu']} ({env['cpus']} логічних ядер); ОС контейнера: {env['os']}; "
        "GPU не використовується.",
        f"- Образ: `{env['image']}` (створено {env['image_created']}); код — коміт `{env['git']}`.",
        "- Режим: відео з файлу в темпі камери (`paced`), частота аналізу 10 кадрів/с; повідомлення — "
        + ("бот Telegram." if telegram else "консоль (Telegram вимкнено для протоколу)."),
        "",
        "## 3. Засоби",
        "",
        f"- Python {env['python']}; " + ", ".join(f"{k} {v}" for k, v in env["versions"].items()) + ".",
        f"- Бекенд моделей: {env['backend']}; сприйняття `{env['perception']}`.",
        f"- Конфіг `configs/default.yaml`: SHA-256 `{env['config_sha256']}` "
        f"(заморожені секції — `{env['frozen_config_sha256'][:16]}…`).",
        f"- Агрегатор A v1.1: SHA-256 `{env['a_sha256']}`.",
        "- Ваги: " + "; ".join(f"{k} `{v[:16]}…`" for k, v in env["weights_sha256"].items()) + ".",
        "- Тестові відео: тестова частина RWF-2000 — 3 бійки й 3 спокійні сцени, кожна — 3 сегменти одного "
        "відео поспіль (15 с; відео вибрано з seed 2026), і XD-Violence (бійка, 2 стрілянини, нормальне "
        "відео); "
        "жоден параметр на них не підбирався.",
        "",
        "## 4. Порядок",
        "",
        "1. Зібрати образ і запустити контейнер без доступу до коду хоста (ваги й дані — томи лише для "
        "читання).",
        "2. Зафіксувати версії й хеші (розділ 3).",
        "3. Запустити автоматичні тести (`pytest`).",
        "4. Для кожного кліпу: прогін offline (RTF, мс/кадр), потім у темпі камери (частота аналізу, "
        "пропущені кадри, затримка від появи кадру до кінця його аналізу, інциденти й тексти повідомлень).",
        "5. Порівняти наявність інциденту з очікуваною для сценарію.",
        "",
        "## 5. Результати",
        "",
        f"**Автоматичні тести:** {'пройдено' if tests['ok'] else 'НЕ пройдено'} — `{tests['summary']}`.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for r in rows:
        lines.append(
            f"| {r['scenario']} | `{r['clip']}` | {_n(r['duration_s'], 1)} | {_n(r['rtf'])} | "
            f"{_n(r['ms_mean'], 1)} / {_n(r['ms_p95'], 1)} | {_n(r['analysis_fps'], 1)} | "
            f"{r['skipped_share']:.0%} | {r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} | "
            f"{r['incidents']} | {'є' if r['expected'] else 'немає'} | "
            f"{'так' if (r['incidents'] > 0) == r['expected'] else 'ні'} |"
        )
    rtf = [r["rtf"] for r in rows]
    lat = [r["latency_p95_ms"] for r in rows]
    lines += [
        "",
        f"Разом: RTF {_n(min(rtf))}–{_n(max(rtf))} (сер. {_n(float(np.mean(rtf)))}); затримка p95 до "
        f"{max(lat):.0f} мс; очікуваний результат — у {len(ok)} з {len(rows)} кліпів.",
        "",
        "**Повідомлення оператору (тексти, як у Telegram):**",
        "",
    ]
    for r in rows:
        for m in r["messages"]:
            lines.append(f"- `{r['clip']}`: " + m.replace("\n", " · "))
    verdict = tests["ok"] and max(rtf) < 1.0
    lines += [
        "",
        "## 6. Висновок",
        "",
        (
            "Програмний продукт з чистого середовища проходить автоматичні тести й обробляє відео швидше за "
            f"реальний час (RTF ≤ {_n(max(rtf))}) з частотою аналізу ~10 кадрів/с. "
            if verdict
            else "Вимоги не виконано повністю (див. таблицю): "
        )
        + f"Наявність інциденту відповідає сценарію в {len(ok)} з {len(rows)} тестових кліпів; розбіжності — "
        + (
            ", ".join(
                f"`{r['clip']}` ({r['scenario']})" for r in rows if (r["incidents"] > 0) != r["expected"]
            )
            or "немає"
        )
        + ". Якість на повних тестових наборах — `docs/bench/experiment_test_*.md`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    from dotenv import find_dotenv, load_dotenv

    from sentinel.notify import make_notifier

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=Path(f"results/protocol_{date.today().isoformat()}.md"))
    parser.add_argument("--telegram", action="store_true", help="send the real messages to the bot in .env")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    started = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    env = environment(config)
    tests = {"ok": True, "summary": "пропущено (--skip-tests)", "s": 0} if args.skip_tests else run_tests()
    print(f"tests: {tests['summary']}", flush=True)
    if args.telegram:
        load_dotenv(find_dotenv(usecwd=True))
    origin = time.time()
    notifier = make_notifier(
        os.environ if args.telegram else {},
        Path("results/protocol_incidents.jsonl"),
        place=config.source.label,
        origin=origin,
    )
    rows = []
    folder = Path(tempfile.mkdtemp(prefix="protocol_"))
    try:
        for scenario, clips, expected in protocol_clips():
            clip = joined(clips, folder)
            row = {"scenario": scenario, "clip": clip.name, "expected": expected} | run_clip(
                config, clip, notifier, origin
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        notifier.close()
        shutil.rmtree(folder, ignore_errors=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(env, tests, rows, started, args.telegram), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"environment": env, "tests": tests, "clips": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()
