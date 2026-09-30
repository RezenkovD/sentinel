"""People source check: how many people each detector / tracked pose model finds on the same clips.

RWF has no person boxes, so the number of confirmed tracks per frame is a proxy for recall (session 02:
the same 20 Fight + 20 NonFight clips of our train split, seed 2026). Counts do not depend on machine load.

    uv run sentinel eval people --per-class 20
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.agents.detector import PersonTracker, PoseTracker
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.sanity import pick_clips
from sentinel.io.source import LatestFrameReader
from sentinel.models import resolve_weights

# label -> (people source, checkpoint)
CANDIDATES = {
    "YOLO11n detect": ("detect", "weights/yolo11n.pt"),
    "YOLO26n detect": ("detect", "weights/yolo26n.pt"),
    "YOLO11n-pose track": ("pose", "weights/yolo11n-pose.pt"),
    "YOLO26n-pose track": ("pose", "weights/yolo26n-pose.pt"),
}


def summarise(per_clip: list[dict]) -> dict:
    """Mean tracks per frame and share of frames without anyone, per class."""
    out = {}
    for label in ("Fight", "NonFight"):
        rows = [c for c in per_clip if f"/{label}/" in c["clip"]]
        out[label] = {
            "tracks_per_frame": float(np.mean([c["tracks_per_frame"] for c in rows])),
            "frames_without_people": float(np.mean([c["empty_share"] for c in rows])),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=Path("data/splits/rwf2000.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/rwf2000"))
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=Path(f"results/people_check_{date.today().isoformat()}.json")
    )
    args = parser.parse_args(argv)
    m = load_config(args.config).models
    clips = pick_clips(args.split, args.per_class)
    results = {"backend": m.backend, "imgsz": m.imgsz, "conf": m.people.conf, "fps": 10, "models": {}}
    for label, (source, weights) in CANDIDATES.items():
        cls = PoseTracker if source == "pose" else PersonTracker
        tracker = cls(
            resolve_weights(Path(weights), m.backend),
            m.imgsz,
            m.people.conf,
            m.people.tracker,
            m.runtime_device,
        )
        per_clip = []
        for clip in clips:
            tracker.reset()
            with LatestFrameReader(args.root / clip, mode="offline", target_fps=10) as reader:
                counts = [len(tracker.update(frame.image)) for frame in reader]
            per_clip.append(
                {
                    "clip": clip,
                    "tracks_per_frame": float(np.mean(counts)),
                    "empty_share": float(np.mean(np.array(counts) == 0)),
                }
            )
        results["models"][label] = {**summarise(per_clip), "clips": per_clip}
        print(label, json.dumps(summarise(per_clip)), flush=True)
    args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
