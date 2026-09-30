"""Baselines on the episode cache (session 03, step 6; SPEC §9 — five experiment configurations).

    uv run sentinel eval baselines

- `baseline_weapon` — as in the bachelor's work: an alarm when a window has a weapon detection scoring at
  least τ (the best raw score of the window, `aux.weapon_max_conf`); no pose, no confirmation. τ on dev: the
  largest Sevilla Cam1 + Cam5 window recall with at most 1 % of our RWF train NonFight clips raising an alarm
  (the same trade-off as the weapon rule of step 5).
- `baseline_motion` — a classic simple violence heuristic: an alarm when a window's `motion_energy` is at
  least τ; the episode score is its largest window value; τ for the best episode F1 on our RWF val.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, _auc, load_episodes
from sentinel.eval.train_a import f1_threshold

WEAPON_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))


def weapon_baseline(
    sevilla: list[CachedEpisode],
    normal: list[CachedEpisode],
    max_false_alarm: float = 0.01,
    grid: tuple[float, ...] = WEAPON_GRID,
) -> dict:
    """The lowest threshold whose false alarms on normal clips stay within `max_false_alarm`."""
    windows = [(a["weapon_max_conf"], a.get("label")) for e in sevilla for a in e.aux]
    positive = np.array([c for c, label in windows if label])
    rows = []
    for tau in sorted(grid):
        segments = [e for e in sevilla if any(a.get("label") for a in e.aux)]
        hit_segments = sum(
            any(a["weapon_max_conf"] >= tau and a.get("label") for a in e.aux) for e in segments
        )
        false = float(np.mean([any(a["weapon_max_conf"] >= tau for a in e.aux) for e in normal]))
        rows.append(
            {
                "threshold": tau,
                "window_recall": float((positive >= tau).mean()) if len(positive) else None,
                "segment_recall": hit_segments / len(segments) if segments else None,
                "false_alarm_share": false,
            }
        )
    allowed = [r for r in rows if r["false_alarm_share"] <= max_false_alarm + 1e-9]
    best = allowed[0] if allowed else min(rows, key=lambda r: r["false_alarm_share"])
    return best | {"grid": rows, "normal_clips": len(normal), "max_false_alarm": max_false_alarm}


def motion_baseline(val: list[CachedEpisode]) -> dict:
    """Threshold on the largest window motion energy of an episode, for the best F1 on val."""
    scores = np.array([max(s.summary.motion_energy or 0.0 for s in e.scenes) for e in val])
    labels = np.array([e.label for e in val])
    threshold, f1 = f1_threshold(scores, labels)
    predicted = scores >= threshold
    window_scores = np.array([s.summary.motion_energy or 0.0 for e in val for s in e.scenes])
    window_labels = np.array([e.label for e in val for _ in e.scenes])
    return {
        "threshold": threshold,
        "f1": f1,
        "episode_auc": _auc(labels, scores),
        "window_auc": _auc(window_labels, window_scores),
        "confusion": {
            "tp": int((predicted & (labels == 1)).sum()),
            "fp": int((predicted & (labels == 0)).sum()),
            "fn": int((~predicted & (labels == 1)).sum()),
            "tn": int((~predicted & (labels == 0)).sum()),
        },
    }


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--out", type=Path, default=Path(f"results/baselines_{date.today().isoformat()}.md"))
    args = parser.parse_args(argv)
    sevilla = load_episodes(args.episodes, "us_gun", ["select"])
    normal = [e for e in load_episodes(args.episodes, "rwf2000", ["train"]) if e.label == 0]
    weapon = weapon_baseline(sevilla, normal)
    motion = motion_baseline(load_episodes(args.episodes, "rwf2000", ["val"]))
    lines = [
        f"# Базові методи на кеші — {date.today().isoformat()}",
        "",
        "## baseline_weapon (детекція зброї з порогом, як у бакалаврській)",
        "",
        f"- Поріг τ = {_fmt(weapon['threshold'])}: найменший, за якого NonFight нашого train RWF дають ≤ 1 % "
        f"кліпів із тривогою (усього {weapon['normal_clips']} кліпів).",
        f"- Севілья Cam1 + Cam5: повнота вікон {_fmt(weapon['window_recall'])}, сегментів "
        f"{_fmt(weapon['segment_recall'])}; хибні тривоги на NonFight train: "
        f"{weapon['false_alarm_share']:.1%}.",
        "",
        "| τ | Повнота вікон Севільї | Повнота сегментів | NonFight train з тривогою |",
        "|---|---|---|---|",
        *[
            f"| {_fmt(r['threshold'])} | {_fmt(r['window_recall'])} | {_fmt(r['segment_recall'])} | "
            f"{r['false_alarm_share']:.1%} |"
            for r in weapon["grid"]
        ],
        "",
        "## baseline_motion (енергія руху в кадрі з порогом)",
        "",
        f"- Поріг τ = {_fmt(motion['threshold'])} рівнів сірого на найбільшій енергії вікна епізоду "
        "(F1 на val).",
        f"- Val: AUC епізодів {_fmt(motion['episode_auc'])}, AUC вікон {_fmt(motion['window_auc'])}, "
        f"F1 {_fmt(motion['f1'])}; TP {motion['confusion']['tp']}, FP {motion['confusion']['fp']}, "
        f"FN {motion['confusion']['fn']}, TN {motion['confusion']['tn']}.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"weapon": weapon, "motion": motion}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
