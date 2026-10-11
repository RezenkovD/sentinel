"""Grey zone of the cascade on our val clips, from the caches of A and B (session 05, step 4).

    uv run sentinel eval cascade-grid [--allow-missing]      # -> results/cascade_grid_<date>.{md,json,png}

For every (g_low, g_high) of the grid each val clip runs through `cascade.simulate` (B's answer counts at
window t + ceil(latency)); reported: the share of windows that asked B, the episode AUC and F1 of the
resulting window risks (A's episode rule), their window AUC, and windows that asked B without an answer in the
cache. The zone: the best episode AUC with at most --max-share of the windows sent to B. No cap by default:
the 25 % cap of session 05 cannot be met (transitions alone send 27 % of the windows to B), so the share is
read off the curve on val (decision 11.10). A missing answer is never replaced by A: with any missing the
result is refused unless --allow-missing (then it is labelled partial).
The clips are the ones B answered in full (`sentinel eval b-run --clips 120`, the same seed); A alone and B
alone on them are given as the two ends of the cascade curve.
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
            calls, missing, scores, window_risks, window_labels = 0, 0, [], [], []
            by_label = {0: [0, 0], 1: [0, 0]}  # calls, windows on calm / fight clips
            for clip in clips:
                run = simulate(clip["scenes"], clip["a"], clip["b"], cfg, g_low, g_high)
                calls, missing = calls + run.calls, missing + len(run.missing)
                by_label[clip["label"]][0] += run.calls
                by_label[clip["label"]][1] += len(clip["scenes"])
                scores.append(episode_score([r.risk for r in run.risks], rule))
                window_risks += [r.risk for r in run.risks]
                window_labels += [clip["label"]] * len(run.risks)
            scores_arr = np.array(scores)
            threshold, f1 = f1_threshold(scores_arr, labels)
            rows.append(
                {
                    "g_low": g_low,
                    "g_high": g_high,
                    "b_share": calls / windows if windows else 0.0,
                    "b_share_calm": by_label[0][0] / by_label[0][1] if by_label[0][1] else None,
                    "b_share_fight": by_label[1][0] / by_label[1][1] if by_label[1][1] else None,
                    "episode_auc": _auc(labels, scores_arr),
                    "window_auc": _auc(np.array(window_labels), np.array(window_risks)),
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
    parser.add_argument("--max-share", type=float, default=1.0)
    parser.add_argument("--clips", type=int, default=120, help="the val clips of b-run (0 = every val clip)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(f"results/cascade_grid_{date.today().isoformat()}"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    a = make_aggregator(config)
    b = make_b(config, wait_on_limits=True)  # only its cache is read: no request is made here
    model = json.loads(Path(config.aggregator_a.model).read_text())
    clips = []
    from sentinel.eval.b_run import pick_clips

    val = load_episodes(args.episodes, "rwf2000", ["val"])
    for episode in val if args.clips == 0 else pick_clips(val, args.clips, args.seed):
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
    ends = endpoints(clips, model["episode_rule"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "clips": len(clips),
                "windows": sum(len(c["scenes"]) for c in clips),
                "max_share": args.max_share,
                "rows": rows,
                "best": best,
                "a_only": ends["a"],
                "b_only": ends["b"],
                "partial": bool(missing),
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {"best": best, "a_only": ends["a"], "b_only": ends["b"], "partial": bool(missing)}, indent=1
        )
    )


def endpoints(clips: list[dict], rule: str) -> dict:
    """A alone and B alone on the same clips (B where it answered; a clip without any answer is skipped)."""
    from sentinel.eval.train_a import episode_score

    out = {}
    for name in ("a", "b"):
        labels, scores, wl, wr = [], [], [], []
        for c in clips:
            risks = [r.risk for r in c["a"]] if name == "a" else [c["b"][i][0].risk for i in sorted(c["b"])]
            if risks:
                labels.append(c["label"])
                scores.append(episode_score(risks, rule))
                wl += [c["label"]] * len(risks)
                wr += risks
        out[name] = {
            "b_share": 0.0 if name == "a" else 1.0,
            "episode_auc": _auc(np.array(labels), np.array(scores)),
            "window_auc": _auc(np.array(wl), np.array(wr)),
        }
    return out


if __name__ == "__main__":
    main()
