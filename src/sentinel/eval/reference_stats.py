"""Usual values of the scene fields on calm windows of our RWF train — the reference given to aggregator B.

    uv run sentinel eval reference-stats    # -> src/sentinel/aggregators/prompts/reference_stats.json

B is zero-shot with respect to labels (decision 01.10): it never sees labelled examples, only what a calm
scene usually looks like — p50 / p90 / p99 of every numeric field over the NonFight windows of our train
part, and the share of windows / people / pairs where a flag is set. A, for comparison, is fitted on the
labelled train windows.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, check_complete, load_episodes

OUT = Path(__file__).parents[1] / "aggregators" / "prompts" / "reference_stats.json"
NUMERIC = {
    "summary": ("n_persons", "crowd_motion", "motion_energy", "motion_area"),
    "persons": ("speed", "limb_speed", "limb_accel", "pose_motion", "motion", "conf"),
    "interactions": ("distance_h", "approach_speed", "duration_s"),
}
FLAGS = {"summary": ("weapon_visible",), "persons": ("holds",), "interactions": ("facing",)}


def reference_stats(episodes: list[CachedEpisode]) -> dict:
    """Quantiles of every numeric field (nulls skipped) and shares of flags over calm train windows."""
    values: dict[str, list[float]] = {f"{g}.{n}": [] for g, names in NUMERIC.items() for n in names}
    flags: dict[str, list[bool]] = {f"{g}.{n}": [] for g, names in FLAGS.items() for n in names}
    windows = 0
    for episode in episodes:
        if episode.meta["part"] != "train":
            raise ValueError(f"{episode.episode_id}: reference values come from the train part only")
        if episode.label != 0:
            continue
        for scene in episode.scenes:
            windows += 1
            groups = {
                "summary": [scene.summary],
                "persons": list(scene.persons),
                "interactions": list(scene.interactions),
            }
            for group, items in groups.items():
                for item in items:
                    for name in NUMERIC[group]:
                        if (v := getattr(item, name)) is not None:
                            values[f"{group}.{name}"].append(float(v))
                    for name in FLAGS[group]:
                        v = getattr(item, name)
                        flags[f"{group}.{name}"].append(v is not None and v is not False)
    fields = {}
    for key, vs in values.items():
        if vs:
            p50, p90, p99 = (round(float(q), 2) for q in np.quantile(vs, [0.5, 0.9, 0.99]))
            fields[key] = {"p50": p50, "p90": p90, "p99": p99, "known": len(vs)}
    shares = {key: round(float(np.mean(vs)), 4) if vs else 0.0 for key, vs in flags.items()}
    return {"windows": windows, "fields": fields, "shares": shares}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    train = load_episodes(args.episodes, "rwf2000", ["train"])
    check_complete(train, "rwf2000", ["train"])
    stats = reference_stats(train) | {
        "source": "calm (NonFight) windows of our RWF train",
        "perception": [train[0].meta["perception_version"], train[0].meta["perception_hash"]],
        "created": date.today().isoformat(),
    }
    args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
