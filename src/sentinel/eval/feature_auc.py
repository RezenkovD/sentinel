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


# figure 2.2 of the thesis: the 17 features of A v1.1 in the order of table 2.3, with the thesis labels
FIGURE_ORDER = (
    ("n_persons", "Кількість людей"),
    ("no_persons", "Людей не знайдено"),
    ("crowd_motion", "Середня швидкість людей"),
    ("max_speed", "Найбільша швидкість людини"),
    ("pair_max_speed", "Швидкість людини, що має пару"),
    ("max_limb_speed", "Швидкість рук відносно тулуба"),
    ("max_limb_accel", "Прискорення рук відносно тулуба"),
    ("limb_accel_missing", "Немає даних про руки"),
    ("max_pose_motion", "Рух тіла відносно тулуба"),
    ("max_person_motion", "Рух пікселів у рамці людини"),
    ("motion_energy", "Енергія руху в кадрі"),
    ("motion_area", "Частка кадру з рухом"),
    ("min_distance_h", "Найменша відстань між людьми"),
    ("max_approach_speed", "Швидкість зближення"),
    ("max_contact_s", "Тривалість близького контакту"),
    ("n_close_pairs", "Кількість пар упритул"),
    ("zone_prior", "Апріорний ризик зони"),
)
# features of the earlier versions that A v1.1 does not use (decisions 30.09, report 03), as they were defined
DROPPED = (
    ("weapon_visible", "Зброя в кадрі"),
    ("weapon_held", "Зброя в руці"),
    ("facing_close", "Обличчям одне до одного впритул"),
)


def dropped_features(scene, config) -> dict[str, float]:
    contact = config.features.contact_distance_h
    close = [i for i in scene.interactions if i.distance_h < contact]
    return {
        "weapon_visible": float(scene.summary.weapon_visible),
        "weapon_held": float(any(p.holds is not None for p in scene.persons)),
        "facing_close": float(any(i.facing is True for i in close)),
    }


def figure_aucs(episodes, config, raw: dict[str, list[float | None]] | None = None) -> dict:
    """Window AUCs on every window of the episodes (label = clip label): A v1.1's features exactly as A sees
    them (`scene_to_features`, nulls at their neutral values), the dropped ones, and — when given — the raw
    wrist acceleration in the frame (v2) on the same windows, with the same null rule as `max_limb_accel`."""
    rows, dropped, labels, raw_values, smooth_known = [], [], [], [], []
    neutral = config.aggregator_a.neutral["max_limb_accel"]
    for e in episodes:
        if raw is not None and len(raw[e.episode_id]) != len(e.scenes):
            raise ValueError(f"{e.episode_id}: {len(raw[e.episode_id])} raw windows, {len(e.scenes)} cached")
        for i, scene in enumerate(e.scenes):
            rows.append(scene_to_features(scene, config))
            dropped.append(dropped_features(scene, config))
            labels.append(e.label)
            smooth_known.append(any(p.limb_accel is not None for p in scene.persons))
            if raw is not None:
                raw_values.append(raw[e.episode_id][i])
    x, y = np.array(rows), np.array(labels)
    index = {name: j for j, name in enumerate(FEATURE_NAMES)}
    result = {
        "windows": {"fight": int(y.sum()), "calm": int(len(y) - y.sum()), "all": len(y)},
        "clips": {"fight": sum(e.label for e in episodes), "calm": sum(1 - e.label for e in episodes)},
        "perception": sorted(
            {f"{e.meta['perception_version']}:{e.meta['perception_hash']}" for e in episodes}
        ),
        "features": [
            {"feature": name, "label_uk": label, "auc": _auc(y, x[:, index[name]])}
            for name, label in FIGURE_ORDER
        ],
        "dropped": [
            {"feature": name, "label_uk": label, "auc": _auc(y, np.array([d[name] for d in dropped]))}
            for name, label in DROPPED
        ],
    }
    if raw is not None:
        r = np.array([neutral if v is None else v for v in raw_values], dtype=float)
        known = np.array([v is not None for v in raw_values])
        both = known & np.array(smooth_known)
        smooth = x[:, index["max_limb_accel"]]
        result["raw_wrist_accel"] = {
            "definition": "v2 limb_accel: finite differences of raw wrist points in the frame / median box "
            "height, heights/s², largest over people; null -> neutral of max_limb_accel",
            "auc": _auc(y, r),
            "windows_known": {"fight": int((known & (y == 1)).sum()), "calm": int((known & (y == 0)).sum())},
            "on_windows_where_both_known": {
                "windows": int(both.sum()),
                "auc_raw": _auc(y[both], r[both]),
                "auc_smoothed_relative": _auc(y[both], smooth[both]),
            },
        }
    return result


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
    parser.add_argument(
        "--figure", type=Path, help="figure 2.2 data (A v1.1 features, all windows) to this JSON instead"
    )
    parser.add_argument("--raw-wrist", type=Path, help="with --figure: `sentinel eval raw-wrist` output")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    episodes = load_episodes(args.episodes, args.dataset, args.parts)
    if args.figure is not None:
        raw = None
        if args.raw_wrist is not None:
            lines = [json.loads(line) for line in args.raw_wrist.read_text(encoding="utf-8").splitlines()]
            raw = {r["episode_id"]: r["raw_wrist_accel"] for r in lines}
        data = figure_aucs(episodes, config, raw) | {
            "dataset": f"{args.dataset} {' + '.join(args.parts)}",
            "created": date.today().isoformat(),
        }
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        args.figure.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(data, ensure_ascii=False, indent=1))
        return
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
