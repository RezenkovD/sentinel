"""Grey zone of the cascade on our val clips, from the caches of A and B (session 05, step 4).

    uv run sentinel eval cascade-grid [--allow-missing]      # -> results/cascade_grid_<date>.{md,json,png}

For every (g_low, g_high) of the grid each val clip runs through `cascade.simulate` (B's answer counts at
window t + ceil(latency)); reported: the share of windows that asked B, the episode AUC and F1 of the
resulting window risks (A's episode rule), and windows that asked B without an answer in the cache. The zone:
the best episode AUC with at most 25 % of the windows sent to B. A missing answer is never replaced by A: with
any missing the result is refused unless --allow-missing (then it is labelled partial).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.aggregators.cascade import simulate
from sentinel.config import IncidentsConfig
from sentinel.eval.cache import _auc
from sentinel.schema import RiskResult

G_LOWS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
G_HIGHS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


def grid(
    clips: list[dict], cfg: IncidentsConfig, g_lows=G_LOWS, g_highs=G_HIGHS, rule: str = "mean"
) -> list[dict]:
    from sentinel.eval.train_a import episode_score, f1_threshold

    labels = np.array([c["label"] for c in clips])
    windows = sum(len(c["scenes"]) for c in clips)
    rows = []
    for g_low in g_lows:
        for g_high in g_highs:
            if g_high < g_low:
                continue
            calls, missing, scores = 0, 0, []
            for clip in clips:
                run = simulate(clip["scenes"], clip["a"], clip["b"], cfg, g_low, g_high)
                calls, missing = calls + run.calls, missing + len(run.missing)
                scores.append(episode_score([r.risk for r in run.risks], rule))
            scores_arr = np.array(scores)
            threshold, f1 = f1_threshold(scores_arr, labels)
            rows.append(
                {
                    "g_low": g_low,
                    "g_high": g_high,
                    "b_share": calls / windows if windows else 0.0,
                    "episode_auc": _auc(labels, scores_arr),
                    "episode_f1": f1,
                    "threshold": threshold,
                    "missing": missing,
                }
            )
    return rows


def choose_zone(rows: list[dict], max_share: float = 0.25) -> dict | None:
    fit = [r for r in rows if r["b_share"] <= max_share and r["episode_auc"] is not None]
    return max(fit, key=lambda r: (r["episode_auc"], -r["b_share"])) if fit else None


def main(argv: list[str] | None = None) -> None:
    from sentinel.aggregators.llm import make_b
    from sentinel.aggregators.rules import make_aggregator
    from sentinel.config import DEFAULT_CONFIG, load_config
    from sentinel.eval.cache import OUT_DEFAULT, load_episodes

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--max-share", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path(f"results/cascade_grid_{date.today().isoformat()}"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    a = make_aggregator(config)
    b = make_b(config, wait_on_limits=True)  # only its cache is read: no request is made here
    model = json.loads(Path(config.aggregator_a.model).read_text())
    clips = []
    for episode in load_episodes(args.episodes, "rwf2000", ["val"]):
        answers = {}
        for i, scene in enumerate(episode.scenes):
            record = b.cache.get(b.key(scene))
            if record and record["status"] == "ok":
                answers[i] = (RiskResult.model_validate(record["result"]), record["latency_s"])
        clips.append(
            {
                "label": episode.label,
                "scenes": episode.scenes,
                "a": [a.assess(s) for s in episode.scenes],
                "b": answers,
            }
        )
    rows = grid(clips, config.incidents, rule=model["episode_rule"])
    missing = max(r["missing"] for r in rows)
    if missing and not args.allow_missing:
        raise SystemExit(
            f"B has no cached answer for up to {missing} windows that ask it: run B on val first"
        )
    best = choose_zone(rows, args.max_share)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "best": best, "partial": bool(missing)}, indent=1)
    )
    _plot(rows, best, args.out.with_suffix(".png"))
    print(json.dumps({"best": best, "partial": bool(missing)}, indent=1))


def _plot(rows: list[dict], best: dict | None, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([100 * r["b_share"] for r in rows], [r["episode_auc"] for r in rows], s=14, color="#4C72B0")
    if best:
        ax.scatter([100 * best["b_share"]], [best["episode_auc"]], s=60, color="#C44E52", label="обрано")
        ax.legend(frameon=False)
    ax.set_xlabel("вікон, що пішли в B, %")
    ax.set_ylabel("AUC епізодів (val)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
