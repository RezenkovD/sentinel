"""Raw wrist acceleration in the frame on the windows of the episode cache — the v2 feature that v3 replaced.

    uv run sentinel eval raw-wrist      # -> results/raw_wrist_accel_train.jsonl (resumable, a line per clip)

The cache keeps scene descriptions, not keypoints, so people and poses are run again over our RWF train clips
with the cache's own perception (same models, backend, tracker and frame grid; perception hash
`ad09b92cb6f6d8ed`). For every window and person: the largest wrist acceleration from finite differences of
three consecutive confident wrist points in frame pixels, divided by the median box height of the person in
the window (heights/s²) — no smoothing and no subtraction of the shoulders, exactly the v2 `limb_accel`. The
window value is the largest over its people, or null. Figure 2.2 compares its AUC with v3's smoothed
acceleration relative to the torso on the same windows.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, load_config

OUT = Path("results/raw_wrist_accel_train.jsonl")
LEFT_WRIST, RIGHT_WRIST = 9, 10

Observation = tuple[float, tuple[float, float, float, float], np.ndarray | None]  # t, box, keypoints (17, 3)


def raw_wrist_accel(obs: Sequence[Observation], min_conf: float) -> float | None:
    """v2 `limb_accel` of one person over a window: heights/s², from raw wrist points in the frame."""
    if not obs:
        return None
    h = float(np.median([box[3] - box[1] for _, box, _ in obs]))
    if h <= 0:
        return None
    best = None
    for wrist in (LEFT_WRIST, RIGHT_WRIST):
        track = [(t, k[wrist, :2]) for t, _, k in obs if k is not None and k[wrist, 2] >= min_conf]
        for (t1, p1), (t2, p2), (t3, p3) in zip(track, track[1:], track[2:], strict=False):
            if not (t1 < t2 < t3):
                continue
            accel = 2 * ((p3 - p2) / (t3 - t2) - (p2 - p1) / (t2 - t1)) / (t3 - t1)
            value = float(np.linalg.norm(accel)) / h
            best = value if best is None else max(best, value)
    return best


def window_value(frames: list[tuple[float, list, dict]], min_conf: float) -> float | None:
    """Largest raw wrist acceleration over the people of one window (frames: t, tracks, skeletons)."""
    people: dict[int, list[Observation]] = {}
    for t, tracks, skeletons in frames:
        for track in tracks:
            people.setdefault(track.track_id, []).append((t, track.box, skeletons.get(track.track_id)))
    values = [v for obs in people.values() if (v := raw_wrist_accel(obs, min_conf)) is not None]
    return max(values) if values else None


def main(argv: list[str] | None = None) -> None:
    from sentinel.agents.context import ZoneContext
    from sentinel.eval.extract import rwf_episodes
    from sentinel.pipeline import Perception, build_agents, perception_hash

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--parts", nargs="+", default=["train"])
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    episodes = rwf_episodes(
        Path("data/splits/rwf2000.json"), Path("data/raw/rwf2000"), args.parts, None, 0, cfg.analysis.fps
    )
    done = set()
    if args.out.exists():
        done = {json.loads(line)["episode_id"] for line in args.out.read_text(encoding="utf-8").splitlines()}
    todo = [e for e in episodes if e.episode_id not in done]
    print(f"{len(episodes)} clips, {len(todo)} to do; perception {perception_hash(cfg)}", flush=True)
    zone = ZoneContext.from_yaml(cfg.context.zones).zone_for(cfg.source.camera_id)
    perception = Perception(build_agents(cfg), cfg, cfg.source.camera_id, zone)
    min_conf = cfg.features.min_keypoint_conf
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n, episode in enumerate(todo, 1):
        perception.reset()
        values, windows, frames = [], [], []
        for frame in episode.frames():  # the frame closing a window belongs to the next one (as in extract)
            scene, _, found = perception.process(frame)
            if scene is not None:
                values.append(window_value(frames, min_conf))
                windows.append([scene.window.t_start, scene.window.t_end])
                frames = []
            frames.append((frame.t, found.tracks, found.skeletons))
        scene, _ = perception.flush()
        if scene is not None:
            values.append(window_value(frames, min_conf))
            windows.append([scene.window.t_start, scene.window.t_end])
        record = {
            "episode_id": episode.episode_id,
            "label": episode.label,
            "perception_hash": perception_hash(cfg),
            "windows": windows,
            "raw_wrist_accel": [None if v is None else round(v, 3) for v in values],
        }
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        if n % 50 == 0:
            print(f"{n}/{len(todo)}", flush=True)
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
