"""Sanity check of the whole pipeline on our RWF-2000 train split (not the experiment, SPEC §9 / session 02).

    uv run python -m eval.sanity --per-class 20        # (a) N Fight + N NonFight clips in offline mode

Loads the models once and resets tracks, windows and incident state between clips. Writes everything
under results/sanity_<time>/: journal.jsonl (every window), incidents.jsonl, key frames, summary.json.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.io.source import LatestFrameReader
from sentinel.notify import ConsoleNotifier
from sentinel.pipeline import build_pipeline

SEED = 2026


def window_records(lines: list[str]) -> list[dict]:
    """Parsed journal lines of analysis windows (the end-of-stream line carries no scene or risk)."""
    records = [json.loads(line) for line in lines]
    return [r for r in records if not r.get("stream_end")]


def pick_clips(split: Path, per_class: int, seed: int = SEED) -> list[str]:
    """N Fight + N NonFight clips from our train part (fixed seed)."""
    train = json.loads(split.read_text())["train"]
    rng = random.Random(seed)
    picked = []
    for label in ("Fight", "NonFight"):
        picked += sorted(rng.sample([p for p in train if f"/{label}/" in p], per_class))
    return picked


def concat_plan(split: Path, gaps: list[int], seed: int = SEED) -> list[tuple[str, list[str]]]:
    """NonFight / Fight segments from our train part; a Fight segment = two consecutive clips."""
    train = json.loads(split.read_text())["train"]
    rng = random.Random(seed)
    by_video: dict[str, list[str]] = {}
    for p in train:
        if "/Fight/" in p:
            stem = Path(p).stem
            video, _, index = stem.rpartition("_")
            if index.isdigit():
                by_video.setdefault(video, []).append(p)
    videos = sorted(v for v, clips in by_video.items() if len(clips) >= 2)
    calm = [p for p in train if "/NonFight/" in p]
    rng.shuffle(calm)
    fights = rng.sample(videos, len(gaps) - 1)
    plan = []
    for i, gap in enumerate(gaps):
        plan.append(("NonFight", [calm.pop() for _ in range(gap)]))
        if i < len(fights):
            clips = sorted(by_video[fights[i]], key=lambda p: int(Path(p).stem.rpartition("_")[2]))[:2]
            plan.append(("Fight", clips))
    return plan


def make_video(plan: list[tuple[str, list[str]]], root: Path, out: Path) -> list[dict]:
    """ffmpeg concat at 1280x720, 30 fps; returns the segments with their start / end in seconds."""
    import subprocess

    clips = [c for _, seg in plan for c in seg]
    inputs = [arg for c in clips for arg in ("-i", str(root / c))]
    scale = (
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"
    )
    chains = "".join(f"[{i}:v]{scale}[v{i}];" for i in range(len(clips)))
    joined = "".join(f"[v{i}]" for i in range(len(clips)))
    graph = f"{chains}{joined}concat=n={len(clips)}:v=1:a=0[out]"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(out)]
    subprocess.run(cmd, check=True)
    segments, t = [], 0.0
    for label, seg in plan:
        segments.append({"label": label, "start_s": t, "end_s": t + 5.0 * len(seg), "clips": seg})
        t += 5.0 * len(seg)
    return segments


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=Path("data/splits/rwf2000.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/rwf2000"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED, help="clip sample (2026 = the tuning sample)")
    parser.add_argument(
        "--make-videos", action="store_true", help="(b) build the concatenated test videos only"
    )
    args = parser.parse_args(argv)

    if args.make_videos:
        out_dir = Path("results/sanity_videos")
        out_dir.mkdir(parents=True, exist_ok=True)
        variants = {"one_minute": [2, 3, 3], "long_gaps": [2, 9, 9]}  # NonFight clips (5 s) around two fights
        manifest = {}
        for name, gaps in variants.items():
            plan = concat_plan(args.split, gaps)
            manifest[name] = make_video(plan, args.root, out_dir / f"{name}.mp4")
        (out_dir / "segments.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        print(
            json.dumps({k: [(s["label"], s["start_s"], s["end_s"]) for s in v] for k, v in manifest.items()})
        )
        return

    config = load_config(args.config)
    out = Path("results") / f"sanity_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    config = config.model_copy(
        update={"incidents": config.incidents.model_copy(update={"output_dir": out / "incidents"})}
    )
    notifier = ConsoleNotifier(out / "incidents.jsonl")
    pipeline = build_pipeline(config, config.source.camera_id, "sanity", out / "journal.jsonl", notifier)

    clips = pick_clips(args.split, args.per_class, args.seed)
    rows = []
    for clip in clips:
        pipeline.reset(run_id=f"sanity-{Path(clip).stem}")
        events = []
        journal_path = out / "journal.jsonl"
        before = len(journal_path.read_text(encoding="utf-8").splitlines()) if journal_path.exists() else 0
        started = time.perf_counter()
        with LatestFrameReader(args.root / clip, mode="offline", target_fps=config.analysis.fps) as reader:
            for frame in reader:
                events += pipeline.process(frame)
            events += pipeline.finish()
            duration = reader.stats.received / reader.source_fps
        wall = time.perf_counter() - started
        windows = window_records(journal_path.read_text(encoding="utf-8").splitlines()[before:])
        risks = [w["risk"]["risk"] for w in windows]
        rows.append(
            {
                "clip": clip,
                "label": "Fight" if "/Fight/" in clip else "NonFight",
                "incident": any(e.kind == "open" for e in events),
                "events": [e.kind for e in events],
                "risks": risks,
                "weapon_windows": sum(w["scene"]["summary"]["weapon_visible"] for w in windows),
                "max_risk": max(risks, default=0.0),
                "rtf": round(wall / duration, 2),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    notifier.close()

    summary = {"clips": len(rows), "seed": args.seed}
    for label in ("Fight", "NonFight"):
        sub = [r for r in rows if r["label"] == label]
        summary[label] = {
            "with_incident": sum(r["incident"] for r in sub),
            "of": len(sub),
            "max_risk_median": float(np.median([r["max_risk"] for r in sub])),
            "windows_high": sum(sum(x >= config.incidents.theta_high for x in r["risks"]) for r in sub),
            "windows": sum(len(r["risks"]) for r in sub),
        }
    (out / "summary.json").write_text(
        json.dumps({"summary": summary, "clips": rows}, ensure_ascii=False, indent=1)
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
