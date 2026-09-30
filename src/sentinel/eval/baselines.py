"""Baselines on the episode cache (session 03, step 6; SPEC §9 — five experiment configurations).

    uv run sentinel eval baselines

- `baseline_weapon` — as in the bachelor's work: an alarm when a window has a weapon detection scoring at
  least τ (the best raw score of the window, `aux.weapon_max_conf`); no pose, no confirmation. τ is the
  detector's operating point chosen on dev (Sevilla Cam1 + Cam5 frames, session 02: `weapons.conf`); the
  strict τ (at most 1 % of our RWF train NonFight clips alarming, the trade-off of step 5) is shown next
  to it.
- `baseline_motion` — a classic simple violence heuristic: an alarm when a window's `motion_energy` is at
  least τ; the episode score is its largest window value; τ for the best episode F1 on our RWF val.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, _auc, check_complete, load_episodes
from sentinel.eval.train_a import f1_threshold

WEAPON_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))


def _weapon_row(sevilla: list[CachedEpisode], normal: list[CachedEpisode], tau: float) -> dict:
    positive = np.array([a["weapon_max_conf"] for e in sevilla for a in e.aux if a.get("label")])
    segments = [e for e in sevilla if any(a.get("label") for a in e.aux)]
    hit = sum(any(a["weapon_max_conf"] >= tau and a.get("label") for a in e.aux) for e in segments)
    return {
        "threshold": tau,
        "window_recall": float((positive >= tau).mean()) if len(positive) else None,
        "segment_recall": hit / len(segments) if segments else None,
        "false_alarm_share": float(
            np.mean([any(a["weapon_max_conf"] >= tau for a in e.aux) for e in normal])
        ),
    }


def weapon_baseline(
    sevilla: list[CachedEpisode],
    normal: list[CachedEpisode],
    operating_point: float,
    max_false_alarm: float = 0.01,
    grid: tuple[float, ...] = WEAPON_GRID,
) -> dict:
    """The detector at its own operating point, plus the strict threshold for comparison.

    The operating point was chosen on Sevilla Cam1 + Cam5 frames (session 02: at most 2 % false frames).
    The strict threshold — the lowest keeping at most `max_false_alarm` of normal clips quiet, the trade-off
    of the weapon rule — is reported next to it: on RWF it leaves the plain detector with no recall at all.
    """
    rows = [_weapon_row(sevilla, normal, tau) for tau in sorted(set(grid) | {operating_point})]
    allowed = [r for r in rows if r["false_alarm_share"] <= max_false_alarm + 1e-9]
    strict = allowed[0] if allowed else min(rows, key=lambda r: r["false_alarm_share"])
    chosen = next(r for r in rows if r["threshold"] == operating_point)
    return chosen | {
        "strict": strict,
        "grid": rows,
        "normal_clips": len(normal),
        "max_false_alarm": max_false_alarm,
    }


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
    parser.add_argument("--allow-partial", action="store_true", help="run on a cache still being extracted")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    sevilla = load_episodes(args.episodes, "us_gun", ["select"])
    train = load_episodes(args.episodes, "rwf2000", ["train"])
    val = load_episodes(args.episodes, "rwf2000", ["val"])
    for episodes, dataset, part in (
        (sevilla, "us_gun", "select"),
        (train, "rwf2000", "train"),
        (val, "rwf2000", "val"),
    ):
        check_complete(episodes, dataset, [part], allow_partial=args.allow_partial)
    normal = [e for e in train if e.label == 0]
    weapon = weapon_baseline(sevilla, normal, operating_point=config.weapons.conf)
    motion = motion_baseline(val)
    lines = [
        f"# Базові методи на кеші — {date.today().isoformat()}",
        "",
        "## baseline_weapon (детекція зброї з порогом, як у бакалаврській)",
        "",
        f"- Поріг τ = {_fmt(weapon['threshold'])} — робоча точка детектора, обрана на кадрах Севільї "
        "Cam1 + Cam5 (сесія 02: ≤ 2 % хибних кадрів).",
        f"- За неї: Севілья Cam1 + Cam5 — повнота вікон {_fmt(weapon['window_recall'])}, сегментів "
        f"{_fmt(weapon['segment_recall'])}; NonFight нашого train RWF з тривогою — "
        f"{weapon['false_alarm_share']:.1%} з {weapon['normal_clips']}.",
        f"- Строгий τ (≤ 1 % NonFight з тривогою, як для правила зброї) — "
        f"{_fmt(weapon['strict']['threshold'])}: повнота вікон {_fmt(weapon['strict']['window_recall'])}, "
        f"сегментів {_fmt(weapon['strict']['segment_recall'])}.",
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
