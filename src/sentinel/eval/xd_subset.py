"""XD-Violence test subset (session 03, step 3): equal time of Fighting (B1), Shooting (B2) and normal (A).

    uv run sentinel eval xd-subset --budget-hours 4      # -> data/splits/xd_subset.json

Only the test part of XD-Violence is used, and only for the experiment. Videos with labels outside the
scenario (riots B4, abuse B5, accidents B6, explosions G — alone or mixed with B1 / B2) are left out: a
coverage limit.
"""

import argparse
import json
from pathlib import Path

import cv2

from sentinel.eval.extract import select_xd_subset


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frames / fps if fps > 0 else 0.0
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--videos", type=Path, default=Path("data/raw/xd-violence/videos"))
    parser.add_argument("--budget-hours", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("data/splits/xd_subset.json"))
    args = parser.parse_args(argv)
    videos = [
        {"name": path.stem, "duration_s": round(video_duration(path), 2)}
        for path in sorted(args.videos.glob("*.mp4"))
    ]
    subset = select_xd_subset(videos, args.budget_hours, args.seed)
    subset["dataset"] = "XD-Violence test (videos folder of the Kaggle mirror, see data/SOURCES.md)"
    subset["pool"] = len(videos)
    args.out.write_text(json.dumps(subset, ensure_ascii=False, indent=1), encoding="utf-8")
    hours: dict[str, float] = {}
    for v in subset["videos"]:
        hours[v["stratum"]] = hours.get(v["stratum"], 0.0) + v["duration_s"] / 3600
    counts = {s: sum(v["stratum"] == s for v in subset["videos"]) for s in hours}
    print(
        f"{len(subset['videos'])} videos: "
        + ", ".join(f"{s} {counts[s]} ({h:.2f} h)" for s, h in hours.items())
    )
    print(f"excluded (outside the scenario): {subset['excluded']['out_of_scope']} of {len(videos)}")


if __name__ == "__main__":
    main()
