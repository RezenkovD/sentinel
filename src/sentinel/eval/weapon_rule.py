"""The weapon rule on cached episodes (session 03, step 5): does the confirmation keep real weapons and drop
false ones?

    uv run sentinel eval weapon-rule [--crops-episodes data/episodes_crops]

- Sevilla Cam1 + Cam5 (dev): recall per segment (a segment with a weapon window gets at least one alarm) and
  the time from the first weapon window to the alarm; alarms before any weapon is annotated are counted apart.
- All NonFight clips of our RWF train: share of clips with a false weapon alarm.
Both for the confirmed rule (the same track in two consecutive windows) and, for comparison, a single window;
for the full frame (the main cache) and, when extracted, for person crops. Choice rule of the task: the
largest Sevilla recall with at most 1 % false alarms on RWF train NonFight.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, check_complete, load_episodes
from sentinel.incidents import WeaponConfirmation
from sentinel.schema import SceneDescription


def weapon_alarms(scenes: list[SceneDescription], windows: int) -> list[bool]:
    """Window by window: does the weapon rule raise an alarm (the incident automaton's rule)?"""
    rule = WeaponConfirmation(windows)
    return [rule.update(scene) for scene in scenes]


def segment_recall(episodes: list[CachedEpisode], windows: int) -> dict:
    """Recall over segments with an annotated weapon, times to the first alarm, alarms before any weapon."""
    hits, times, early = 0, [], 0
    segments = [e for e in episodes if any(a.get("label") for a in e.aux)]
    for episode in segments:
        alarms = weapon_alarms(episode.scenes, windows)
        first_weapon = next(i for i, a in enumerate(episode.aux) if a.get("label"))
        early += any(alarms[:first_weapon])
        later = [i for i in range(first_weapon, len(alarms)) if alarms[i]]
        if later:
            hits += 1
            times.append(episode.scenes[later[0]].window.t_end - episode.scenes[first_weapon].window.t_start)
    return {
        "segments": len(segments),
        "recall": hits / len(segments) if segments else None,
        "times_to_alarm_s": times,
        "alarms_before_weapon": early,
    }


def alarm_share(episodes: list[CachedEpisode], windows: int) -> float:
    """Share of episodes with at least one weapon alarm."""
    return float(np.mean([any(weapon_alarms(e.scenes, windows)) for e in episodes])) if episodes else 0.0


def evaluate(root: Path, allow_partial: bool = False) -> dict:
    sevilla = load_episodes(root, "us_gun", ["select"])
    check_complete(sevilla, "us_gun", ["select"], allow_partial=allow_partial)
    normal = [e for e in load_episodes(root, "rwf2000", ["train"]) if e.label == 0]
    if len(normal) < 640 and not allow_partial:  # a crops cache holds NonFight clips only
        raise RuntimeError(
            f"{root}: {len(normal)} of 640 NonFight train clips — wait or pass --allow-partial"
        )
    rows = {}
    for windows in (2, 1):
        seg = segment_recall(sevilla, windows)
        rows[f"confirm{windows}"] = seg | {
            "median_time_to_alarm_s": float(np.median(seg["times_to_alarm_s"]))
            if seg["times_to_alarm_s"]
            else None,
            "rwf_nonfight_clips": len(normal),
            "rwf_nonfight_false_alarm_share": alarm_share(normal, windows),
        }
    return rows


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}".replace(".", ",") if isinstance(value, float) else str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--crops-episodes", type=Path, help="cache extracted with weapons.mode crops (dev parts)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path(f"results/weapon_rule_{date.today().isoformat()}.md")
    )
    parser.add_argument("--allow-partial", action="store_true", help="run on a cache still being extracted")
    args = parser.parse_args(argv)
    modes = {"кадр 480:2": args.episodes}
    if args.crops_episodes is not None:
        modes["вирізки 320:2"] = args.crops_episodes
    results = {mode: evaluate(root, args.allow_partial) for mode, root in modes.items()}
    lines = [
        f"# Правило «зброя в руці» на кеші — {date.today().isoformat()}",
        "",
        "Севілья Cam1 + Cam5 (dev, 2 кадри/с, детектор на кожному кадрі) і всі NonFight нашого train RWF.",
        "",
        "| Режим | Підтвердження | Сегментів зі зброєю | Повнота сегментів | Медіана часу до тривоги, с "
        "| Тривог до появи зброї | NonFight train з хибною тривогою |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, rows in results.items():
        for key, row in rows.items():
            lines.append(
                f"| {mode} | {'2 вікна поспіль' if key == 'confirm2' else '1 вікно'} | {row['segments']} | "
                f"{_fmt(row['recall'])} | {_fmt(row['median_time_to_alarm_s'])} | "
                f"{row['alarms_before_weapon']} | "
                f"{row['rwf_nonfight_false_alarm_share']:.1%} з {row['rwf_nonfight_clips']} |"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
