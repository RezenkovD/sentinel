"""Pilot of aggregator B on stratified windows of our RWF val (session 05, step 3; free Groq tier).

    uv run sentinel eval b-pilot --prompt b_draft2 --per-stratum 2            # a prompt iteration
    uv run sentinel eval b-pilot --prompt b_v1 --per-stratum 15 --repeat-windows 40 --repeats 2

Windows of our val only, stratified by clip label (calm / fight) x people (with / empty) x risk of A (low /
high, 0.5); per stratum the first `per-stratum` of a seeded order, so a small iteration sample is part of the
full one. Answers go through B's cache (resumable; limits are waited out). Repeats for consistency skip the
cache and are kept in their own file. Reported: share of valid JSON, latency p50 / p95, tokens per call (input
/ output / reasoning), window AUC of B and of A on the same windows, consistency, examples of reason_uk.
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, _auc, load_episodes
from sentinel.schema import SceneDescription

HIGH = 0.5  # A risk splitting the strata


def stratum(label: int, scene: SceneDescription, a_risk: float) -> tuple[str, str, str]:
    return (
        "fight" if label == 1 else "calm",
        "people" if scene.summary.n_persons > 0 else "empty",
        "high" if a_risk >= HIGH else "low",
    )


def pilot_sample(episodes: list[CachedEpisode], aggregator_a, per_stratum: int, seed: int) -> list[dict]:
    """`per_stratum` windows of every stratum that has them, the first ones of a seeded order."""
    windows: dict[tuple, list[dict]] = {}
    for episode in episodes:
        if episode.meta["part"] != "val":
            raise ValueError(f"{episode.episode_id}: the pilot uses our val part only")
        for i, scene in enumerate(episode.scenes):
            risk = aggregator_a.assess(scene).risk
            key = stratum(episode.label, scene, risk)
            windows.setdefault(key, []).append(
                {
                    "episode": episode.episode_id,
                    "window": i,
                    "label": episode.label,
                    "a_risk": risk,
                    "stratum": key,
                }
            )
    out = []
    for key in sorted(windows):
        pool = windows[key]
        random.Random(f"{seed}-{key}").shuffle(pool)
        out += pool[:per_stratum]
    return out


def consistency(repeats: dict[str, list[tuple[float, str]]]) -> dict:
    """Spread of the risk over repeated answers of each window and the share of windows with one level."""
    usable = {k: v for k, v in repeats.items() if len(v) >= 2}
    if not usable:
        return {"windows": 0, "risk_std_mean": None, "risk_range_max": None, "level_agreement": None}
    stds = [float(np.std([r for r, _ in v])) for v in usable.values()]
    ranges = [max(r for r, _ in v) - min(r for r, _ in v) for v in usable.values()]
    agree = [len({lvl for _, lvl in v}) == 1 for v in usable.values()]
    return {
        "windows": len(usable),
        "risk_std_mean": float(np.mean(stds)),
        "risk_range_max": float(max(ranges)),
        "level_agreement": float(np.mean(agree)),
    }


def _q(values: list[float], q: float) -> float | None:
    return float(np.quantile(values, q)) if values else None


def main(argv: list[str] | None = None) -> None:
    from sentinel.aggregators.llm import make_b
    from sentinel.aggregators.rules import make_aggregator
    from sentinel.config import DEFAULT_CONFIG, load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--prompt", required=True, help="prompt version, prompts/<version>.md")
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--repeat-windows", type=int, default=0, help="windows asked again without the cache")
    parser.add_argument("--repeats", type=int, default=2, help="extra answers for each of those windows")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    val = load_episodes(args.episodes, "rwf2000", ["val"])
    sample = pilot_sample(val, make_aggregator(config), args.per_stratum, args.seed)
    scenes = {e.episode_id: e.scenes for e in val}
    b = make_b(config, wait_on_limits=True, prompt_version=args.prompt)
    out = args.out or Path(f"results/b_pilot_{args.prompt}_{date.today().isoformat()}")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n, w in enumerate(sample, 1):
        answer = b.answer(scenes[w["episode"]][w["window"]])
        rows.append(w | {"key": b.key(scenes[w["episode"]][w["window"]]), "answer": answer})
        print(f"{n}/{len(sample)} {w['stratum']} {answer.status} cached={answer.cached}", flush=True)
    # consistency: the first windows of the sample, asked again without the cache (own resumable file)
    repeats_file = out.with_name(out.name + "_repeats.jsonl")
    done: dict[str, list[dict]] = {}
    if repeats_file.exists():
        for line in repeats_file.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.setdefault(r["key"], []).append(r)
    for row in rows[: args.repeat_windows]:
        scene = scenes[row["episode"]][row["window"]]
        while len(done.get(row["key"], [])) < args.repeats:
            again = b.answer(scene, use_cache=False)
            record = {
                "key": row["key"],
                "status": again.status,
                "result": again.result.model_dump(mode="json") if again.result else None,
                "usage": again.usage,
                "latency_s": again.latency_s,
            }
            with repeats_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            done.setdefault(row["key"], []).append(record)
            print(f"repeat {row['key'][:8]} {again.status}", flush=True)
    write_report(out, args, rows, done)


def write_report(out: Path, args, rows: list[dict], repeats: dict[str, list[dict]]) -> dict:
    answers = [r["answer"] for r in rows]
    ok = [r for r in rows if r["answer"].status == "ok"]
    calls = [a for a in answers if a.usage.get("input")]
    labels = np.array([r["label"] for r in ok])
    b_auc = _auc(labels, np.array([r["answer"].result.risk for r in ok])) if ok else None
    a_auc = _auc(labels, np.array([r["a_risk"] for r in ok])) if ok else None
    series = {
        r["key"]: [(r["answer"].result.risk, r["answer"].result.level)]
        + [(x["result"]["risk"], x["result"]["level"]) for x in repeats.get(r["key"], []) if x["result"]]
        for r in ok
        if r["key"] in repeats
    }
    metrics = {
        "prompt": args.prompt,
        "windows": len(rows),
        "valid_share": len(ok) / len(rows) if rows else None,
        "statuses": {s: sum(a.status == s for a in answers) for s in ("ok", "invalid", "unavailable")},
        "latency_p50_s": _q([a.latency_s for a in calls], 0.5),
        "latency_p95_s": _q([a.latency_s for a in calls], 0.95),
        "tokens_mean": {
            k: float(np.mean([a.usage[k] for a in calls])) if calls else None
            for k in ("input", "output", "reasoning")
        },
        "window_auc_b": b_auc,
        "window_auc_a_same_windows": a_auc,
        "consistency": consistency(series),
        "repeat_calls": sum(len(v) for v in repeats.values()),
        "strata": {"-".join(k): n for k, n in _count([r["stratum"] for r in rows]).items()},
    }
    examples = [
        {"stratum": "-".join(r["stratum"]), "label": r["label"], "risk": r["answer"].result.risk,
         "level": r["answer"].result.level, "reason_uk": r["answer"].result.reason_uk}
        for r in ok
    ]  # fmt: skip
    out.with_suffix(".json").write_text(
        json.dumps({"metrics": metrics, "examples": examples}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=1))
    return metrics


def _count(items: list) -> dict:
    out: dict = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


if __name__ == "__main__":
    main()
