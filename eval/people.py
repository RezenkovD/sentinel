"""Person detector check: how many people each model finds on the same clips (RWF has no person boxes, so
the count of confirmed tracks per frame is a proxy for recall).

    uv run python -m eval.people --per-class 20
"""

import argparse
import json
from pathlib import Path

import numpy as np

from eval.sanity import pick_clips
from sentinel.agents.detector import PersonTracker
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.io.source import LatestFrameReader

MODELS = ("weights/yolo11n.pt", "weights/yolo26n.pt")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=Path("data/splits/rwf2000.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/rwf2000"))
    parser.add_argument("--per-class", type=int, default=20)
    args = parser.parse_args(argv)
    cfg = load_config(args.config).models
    clips = pick_clips(args.split, args.per_class)
    results = {}
    for weights in MODELS:
        tracker = PersonTracker(Path(weights), cfg.imgsz, cfg.person.conf, cfg.person.tracker, cfg.device)
        per_clip = []
        for clip in clips:
            tracker.reset()
            counts = []
            with LatestFrameReader(args.root / clip, mode="offline", target_fps=10) as reader:
                counts = [len(tracker.update(frame.image)) for frame in reader]
            per_clip.append(
                {
                    "clip": clip,
                    "tracks_per_frame": float(np.mean(counts)),
                    "empty_share": float(np.mean(np.array(counts) == 0)),
                }
            )
        results[weights] = {
            label: {
                "tracks_per_frame": float(
                    np.mean([c["tracks_per_frame"] for c in per_clip if f"/{label}/" in c["clip"]])
                ),
                "frames_without_people": float(
                    np.mean([c["empty_share"] for c in per_clip if f"/{label}/" in c["clip"]])
                ),
            }
            for label in ("Fight", "NonFight")
        }
        results[weights]["clips"] = per_clip
        print(weights, json.dumps({k: v for k, v in results[weights].items() if k != "clips"}), flush=True)
    out = Path("results/people_check.json")
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
