"""One-feature AUCs of A's features on cached episodes (session 03, step 2: the check before freezing).

    uv run sentinel eval feature-auc --dataset rwf2000 --parts train

Window label = clip label (RWF has clip labels only). A's features are judged on windows with people (A
returns zero risk for an empty scene without looking); pixel motion also over all windows. The episode AUC
ranks clips by the feature's largest value over their windows with people.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.aggregators.rules import FEATURE_NAMES, FEATURE_NAMES_UK, scene_to_features
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import OUT_DEFAULT, _auc, feature_aucs, load_episodes


def episode_aucs(episodes, config) -> dict[str, float | None]:
    neutral = config.aggregator_a.neutral
    values, labels = [], []
    for e in episodes:
        rows = [scene_to_features(s, config) for s in e.scenes if s.summary.n_persons > 0]
        values.append(np.max(rows, axis=0) if rows else np.array([neutral[n] for n in FEATURE_NAMES]))
        labels.append(e.label)
    x, y = np.array(values), np.array(labels)
    return {name: _auc(y, x[:, j]) for j, name in enumerate(FEATURE_NAMES)}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}".replace(".", ",")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--dataset", default="rwf2000")
    parser.add_argument("--parts", nargs="+", default=["train"])
    parser.add_argument(
        "--out", type=Path, default=Path(f"results/features_auc_{date.today().isoformat()}.md")
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    episodes = load_episodes(args.episodes, args.dataset, args.parts)
    table = feature_aucs(episodes, config)
    by_episode = episode_aucs(episodes, config)
    empty = {
        label: np.mean([s.summary.n_persons == 0 for e in episodes if e.label == y for s in e.scenes])
        for label, y in (("Fight", 1), ("NonFight", 0))
    }
    n_fight = sum(e.label for e in episodes)
    lines = [
        f"# Одновимірні AUC ознак A — {args.dataset} {' + '.join(args.parts)}, {date.today().isoformat()}",
        "",
        f"Епізодів {len(episodes)} ({n_fight} Fight / {len(episodes) - n_fight} NonFight), perception "
        f"{episodes[0].meta['perception_version']} / {episodes[0].meta['perception_hash']}. Вікон без людей: "
        f"Fight {empty['Fight']:.0%}, NonFight {empty['NonFight']:.0%}.",
        "",
        "| Ознака | Назва в коді | AUC вікон (з людьми) | Сила: abs(AUC − 0,5) | AUC епізодів (макс.) "
        "| AUC усіх вікон |",
        "|---|---|---|---|---|---|",
    ]
    for row in table:
        name = row["feature"]
        strength = None if row["auc"] is None else abs(row["auc"] - 0.5)
        lines.append(
            f"| {FEATURE_NAMES_UK[name]} | `{name}` | {_fmt(row['auc'])} | {_fmt(strength)} | "
            f"{_fmt(by_episode[name])} | {_fmt(row.get('auc_all'))} |"
        )
    lines += [
        "",
        f"Вікон з людьми: {table[0]['windows']}; усіх вікон: {sum(len(e.scenes) for e in episodes)}. "
        "AUC < 0,5 — ознака більша в NonFight.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"windows": table, "episodes": by_episode, "empty_share": empty}, indent=1),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
