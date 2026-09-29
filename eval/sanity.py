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


def pick_clips(split: Path, per_class: int, seed: int = SEED) -> list[str]:
    """N Fight + N NonFight clips from our train part (fixed seed)."""
    train = json.loads(split.read_text())["train"]
    rng = random.Random(seed)
    picked = []
    for label in ("Fight", "NonFight"):
        picked += sorted(rng.sample([p for p in train if f"/{label}/" in p], per_class))
    return picked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=Path("data/splits/rwf2000.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/rwf2000"))
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out = Path("results") / f"sanity_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    config = config.model_copy(
        update={"incidents": config.incidents.model_copy(update={"output_dir": out / "incidents"})}
    )
    notifier = ConsoleNotifier(out / "incidents.jsonl")
    pipeline = build_pipeline(config, config.source.camera_id, "sanity", out / "journal.jsonl", notifier)

    clips = pick_clips(args.split, args.per_class)
    rows = []
    for clip in clips:
        pipeline.reset(run_id=f"sanity-{Path(clip).stem}")
        events, risks = [], []
        before = pipeline.windows
        started = time.perf_counter()
        with LatestFrameReader(args.root / clip, mode="offline", target_fps=config.analysis.fps) as reader:
            for frame in reader:
                events += pipeline.process(frame)
            events += pipeline.finish()
            duration = reader.stats.received / reader.source_fps
        wall = time.perf_counter() - started
        journal = (out / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        risks = [json.loads(line)["risk"]["risk"] for line in journal[before:]]
        rows.append(
            {
                "clip": clip,
                "label": "Fight" if "/Fight/" in clip else "NonFight",
                "incident": any(e.kind == "open" for e in events),
                "events": [e.kind for e in events],
                "risks": risks,
                "max_risk": max(risks, default=0.0),
                "rtf": round(wall / duration, 2),
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    notifier.close()

    summary = {"clips": len(rows), "seed": SEED}
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
