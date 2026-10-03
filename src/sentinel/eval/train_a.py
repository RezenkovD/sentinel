"""Training aggregator A: logistic regression on cached scene descriptions (session 03, step 4).

    uv run sentinel eval train-a            # RWF train -> fit, RWF val -> C, episode score, thresholds

- Windows of our RWF train; the window label is the clip label (weak labels: a Fight clip also has calm
  moments, a NonFight clip may have fast motion). `--windows people` (v1): windows with people only, and the
  model keeps every empty frame at zero. `--windows all` (decision 30.09, SPEC §5): also the empty frames
  where something moves, with the `no_persons` indicator — only an empty frame that does not move is zero.
- sklearn Pipeline: standardisation, clipping to ±`aggregator_a.clip`, `LogisticRegression(class_weight=
  "balanced")`; C by the window AUC on our val, fixed seed. The fitted numbers are exported to the same
  RulesAggregator that `sentinel run` uses, so the running A is exactly this model.
- Episode score from window risks (max / mean of the two largest / mean) by the episode AUC on val; the
  episode threshold for F1 on val; θ_high / θ_low for the incident automaton from the risks of calm (NonFight)
  val windows. Nothing here looks at the test part.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from sentinel.aggregators.rules import (
    FEATURE_NAMES,
    FEATURE_NAMES_UK,
    RulesAggregator,
    quiet_empty,
    scene_to_features,
)
from sentinel.config import DEFAULT_CONFIG, Config, load_config
from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, _auc, check_complete, load_episodes

C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
EPISODE_RULES = ("max", "top2", "mean")
WINDOWS = {"people": "zero", "all": "score"}  # training windows -> what the model does with empty frames


def uses_window(scene, windows: str, config: Config) -> bool:
    if windows == "people":
        return scene.summary.n_persons > 0
    if windows == "all":
        return not quiet_empty(scene, config.aggregator_a.quiet_motion_area)
    raise ValueError(f"windows must be one of {sorted(WINDOWS)}, got {windows!r}")


def train_windows(
    episodes: list[CachedEpisode], config: Config, windows: str = "people"
) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels (= episode label) of the windows A learns from: with people, or all but the empty
    frames that do not move."""
    rows, labels = [], []
    for episode in episodes:
        for scene in episode.scenes:
            if uses_window(scene, windows, config):
                rows.append(scene_to_features(scene, config))
                labels.append(episode.label)
    return np.array(rows).reshape(-1, len(FEATURE_NAMES)), np.array(labels)


def make_pipeline(c: float, clip: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("clip", FunctionTransformer(np.clip, kw_args={"a_min": -clip, "a_max": clip})),
            ("lr", LogisticRegression(C=c, class_weight="balanced", max_iter=5000, random_state=seed)),
        ]
    )


@dataclass
class FitResult:
    pipeline: object
    aggregator: RulesAggregator
    c: float
    window_auc_val: float
    by_c: list[tuple[float, float]]  # (C, window AUC on val)


def fit_a(
    train: list[CachedEpisode],
    val: list[CachedEpisode],
    config: Config,
    c_grid: tuple[float, ...] = C_GRID,
    seed: int = 2026,
    exclude: tuple[str, ...] = (),
    windows: str = "people",
) -> FitResult:
    """Fit on train for every C; keep the one with the best window AUC on val (ties: the smaller C).

    `exclude` leaves features out of the regression: they keep weight 0 in the exported model (e.g. weapon
    features, which RWF can only teach from false detections; weapons are handled by the rule).
    """
    unknown = set(exclude) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown features to exclude: {sorted(unknown)}")
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in exclude]
    x, y = train_windows(train, config, windows)
    xv, yv = train_windows(val, config, windows)
    x, xv = x[:, keep], xv[:, keep]
    clip = config.aggregator_a.clip
    best, by_c = None, []
    for c in c_grid:
        pipeline = make_pipeline(c, clip, seed).fit(x, y)
        auc = _auc(yv, pipeline.predict_proba(xv)[:, 1])
        by_c.append((c, auc))
        if best is None or auc > best[1] + 1e-12:
            best = (c, auc, pipeline)
    c, auc, pipeline = best
    scaler, lr = pipeline.named_steps["scale"], pipeline.named_steps["lr"]
    n = len(FEATURE_NAMES)
    weights, mean, scale = np.zeros(n), np.zeros(n), np.ones(n)  # excluded features: weight 0
    weights[keep], mean[keep], scale[keep] = lr.coef_[0], scaler.mean_, scaler.scale_
    aggregator = RulesAggregator(
        weights=weights,
        bias=float(lr.intercept_[0]),
        mean=mean,
        scale=scale,
        clip=clip,
        config=config,
        empty_windows=WINDOWS[windows],
    )
    return FitResult(pipeline, aggregator, c, auc, by_c)


def episode_score(risks: list[float], rule: str) -> float:
    """One number per episode from its window risks: max, mean of the two largest, or mean."""
    ordered = sorted(risks, reverse=True)
    if rule == "max":
        return ordered[0]
    if rule == "top2":
        return float(np.mean(ordered[:2]))
    if rule == "mean":
        return float(np.mean(ordered))
    raise ValueError(f"unknown episode score {rule!r}")


def f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold (score >= it means dangerous) with the best F1; ties go to the higher threshold."""
    best = (float(scores.max()), -1.0)
    for threshold in sorted(set(scores.tolist())):
        predicted = scores >= threshold
        tp = int((predicted & (labels == 1)).sum())
        fp = int((predicted & (labels == 0)).sum())
        fn = int((~predicted & (labels == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 >= best[1]:
            best = (float(threshold), f1)
    return best


def quantile_thresholds(calm: np.ndarray, high_q: float, low_q: float) -> tuple[float, float]:
    """θ_high / θ_low as quantiles of the risks of calm windows."""
    return float(np.quantile(calm, high_q)), float(np.quantile(calm, low_q))


def _fmt(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-partial", action="store_true", help="run on a cache still being extracted")
    parser.add_argument(
        "--high-quantile", type=float, default=0.95, help="θ_high: quantile of calm val windows"
    )
    parser.add_argument(
        "--low-quantile", type=float, default=0.80, help="θ_low: quantile of calm val windows"
    )
    parser.add_argument("--model", type=Path, default=Path("weights/aggregator_a/a_v1.json"))
    parser.add_argument(
        "--exclude", nargs="*", default=[], choices=FEATURE_NAMES, help="features left out of the regression"
    )
    parser.add_argument(
        "--windows",
        choices=sorted(WINDOWS),
        default="people",
        help="people: windows with people, empty frames stay zero; all: also empty frames that move",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(f"results/aggregator_a_{date.today().isoformat()}.md")
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    train = load_episodes(args.episodes, "rwf2000", ["train"])
    val = load_episodes(args.episodes, "rwf2000", ["val"])
    check_complete(train, "rwf2000", ["train"], allow_partial=args.allow_partial)
    check_complete(val, "rwf2000", ["val"], allow_partial=args.allow_partial)
    result = fit_a(train, val, config, seed=args.seed, exclude=tuple(args.exclude), windows=args.windows)
    a = result.aggregator

    val_risks = [[a.assess(s).risk for s in e.scenes] for e in val]  # empty windows: as A really answers
    labels = np.array([e.label for e in val])
    rule_aucs = {r: _auc(labels, np.array([episode_score(rs, r) for rs in val_risks])) for r in EPISODE_RULES}
    rule = max(EPISODE_RULES, key=lambda r: (rule_aucs[r], -EPISODE_RULES.index(r)))
    scores = np.array([episode_score(rs, rule) for rs in val_risks])
    threshold, f1 = f1_threshold(scores, labels)
    predicted = scores >= threshold
    confusion = {
        "tp": int((predicted & (labels == 1)).sum()),
        "fp": int((predicted & (labels == 0)).sum()),
        "fn": int((~predicted & (labels == 1)).sum()),
        "tn": int((~predicted & (labels == 0)).sum()),
    }
    window_risk = np.array([r for rs in val_risks for r in rs])
    window_label = np.array([e.label for e, rs in zip(val, val_risks, strict=True) for _ in rs])
    theta_high, theta_low = quantile_thresholds(
        window_risk[window_label == 0], args.high_quantile, args.low_quantile
    )
    fight_high = float((window_risk[window_label == 1] >= theta_high).mean())
    with_people = np.array([s.summary.n_persons > 0 for e in val for s in e.scenes])
    metrics = {
        "window_auc_val_fit_windows": result.window_auc_val,  # the windows of the training rule (choice of C)
        "window_auc_val_with_people": _auc(window_label[with_people], window_risk[with_people]),
        "window_auc_val_all": _auc(window_label, window_risk),
        "episode_auc_val": rule_aucs[rule],
        "episode_rule_aucs": rule_aucs,
        "episode_f1_val": f1,
        "confusion_val": confusion,
        "fight_windows_above_theta_high": fight_high,
    }
    meta = train[0].meta
    extra = {
        "perception_version": meta["perception_version"],
        "perception_hash": meta["perception_hash"],
        "trained_on": {"train_episodes": len(train), "val_episodes": len(val)},
        "c": result.c,
        "c_grid": [{"c": c, "window_auc_val": auc} for c, auc in result.by_c],
        "excluded_features": list(args.exclude),
        "training_windows": args.windows,
        "episode_rule": rule,
        "episode_threshold": threshold,
        "theta_high": theta_high,
        "theta_low": theta_low,
        "theta_quantiles": [args.high_quantile, args.low_quantile],
        "metrics_val": metrics,
        "seed": args.seed,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    a.save(args.model, extra=extra)
    from sentinel.models import sha256

    digest = sha256(args.model)
    order = np.argsort(-np.abs(a.weights))
    lines = [
        f"# Агрегатор A — логістична регресія (`{args.model.name}`), {date.today().isoformat()}",
        "",
        f"Навчання: наш train RWF ({len(train)} кліпів), "
        + (
            "вікна з людьми (порожні кадри — нуль)"
            if args.windows == "people"
            else "усі вікна, крім порожніх без руху (τ_quiet = "
            f"{_fmt(config.aggregator_a.quiet_motion_area)} % кадру), з індикатором `no_persons`"
        )
        + f", мітка вікна = мітка кліпу. Вибір C, "
        f"оцінки епізоду й порогів — наш val ({len(val)} кліпів). Тест не використовувався. Perception "
        f"{meta['perception_version']} / {meta['perception_hash']}. Модель: `{args.model}` (SHA-256 "
        f"`{digest[:16]}…`).",
        "",
        "## Метрики на val",
        "",
        f"- C = {result.c:g} (AUC вікон на val для сітки: "
        + ", ".join(f"{c:g} → {_fmt(auc)}" for c, auc in result.by_c)
        + ")",
        f"- AUC вікон: {_fmt(metrics['window_auc_val_with_people'])} (з людьми), "
        f"{_fmt(metrics['window_auc_val_all'])} (усі вікна, як A справді відповідає)",
        "- AUC епізодів за оцінкою: "
        + ", ".join(f"{r} {_fmt(v)}" for r, v in rule_aucs.items())
        + f" → `{rule}`",
        f"- Поріг епізоду (F1 на val): {_fmt(threshold)}; F1 = {_fmt(f1)}",
        f"- Матриця помилок (епізоди val): TP {confusion['tp']}, FP {confusion['fp']}, FN {confusion['fn']}, "
        f"TN {confusion['tn']}",
        f"- θ_high = {_fmt(theta_high)} ({args.high_quantile:.0%} спокійних вікон val нижче), θ_low = "
        f"{_fmt(theta_low)} ({args.low_quantile:.0%}); вікон бійок ≥ θ_high: {fight_high:.0%}",
        "",
        "## Коефіцієнти (на стандартизованих ознаках, за спаданням модуля)",
        "",
        "| Ознака | Назва в коді | Коефіцієнт | Середнє | Масштаб (σ) |",
        "|---|---|---|---|---|",
    ]
    for j in order:
        name = FEATURE_NAMES[j]
        lines.append(
            f"| {FEATURE_NAMES_UK[name]} | `{name}` | {_fmt(a.weights[j])} | {_fmt(a.mean[j])} "
            f"| {_fmt(a.scale[j])} |"
        )
    lines.append(f"| зсув (bias) | — | {_fmt(a.bias)} | — | — |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(extra, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
