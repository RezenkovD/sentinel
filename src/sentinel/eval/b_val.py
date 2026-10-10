"""B against A on the 120 val clips of `b-run` and on the 117 pilot windows — from the caches, no calls.

    uv run sentinel eval b-val          # -> results/b_val_<date>.json, docs/bench/b_val_<date>.md

Window AUC of B (frozen prompt `b_v1`) and of A v1.1 on exactly the same windows, with 95 % intervals from a
bootstrap over clips (1000 resamples, seed 2026) and the paired difference B − A; also on windows with and
without people, and the episode AUC (mean of the window risks, A's episode rule). Share of valid answers,
latency p50 / p95 and tokens of the original calls, and what they would cost at the provider's list price
(the free tier was used; cached input is not recorded, so the cost is an upper bound).
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import OUT_DEFAULT, _auc, load_episodes
from sentinel.eval.experiment import bootstrap


def compare(groups: list[dict]) -> dict:
    """groups: one per clip — labels, B risks, A risks (windows with a valid B answer only)."""
    groups = [g for g in groups if len(g["y"])]

    def auc_of(key: str):
        def stat(idx: np.ndarray) -> float | None:
            y = np.concatenate([groups[i]["y"] for i in idx])
            return _auc(y, np.concatenate([groups[i][key] for i in idx]))

        return stat

    def diff(idx: np.ndarray) -> float | None:
        b, a = auc_of("b")(idx), auc_of("a")(idx)
        return None if b is None or a is None else b - a

    every = np.arange(len(groups))
    return {
        "clips": len(groups),
        "windows": int(sum(len(g["y"]) for g in groups)),
        "fight_windows": int(sum(g["y"].sum() for g in groups)),
        "auc_b": auc_of("b")(every),
        "auc_b_ci": bootstrap(auc_of("b"), len(groups)),
        "auc_a": auc_of("a")(every),
        "auc_a_ci": bootstrap(auc_of("a"), len(groups)),
        "b_minus_a": diff(every),
        "b_minus_a_ci": bootstrap(diff, len(groups)),
    }


def usage_stats(records: list[dict], price_in: float, price_out: float) -> dict:
    calls = [r for r in records if r and r.get("usage", {}).get("input")]
    tokens = {k: int(sum(r["usage"][k] for r in calls)) for k in ("input", "output", "reasoning")}
    latency = [r["latency_s"] for r in calls]
    cost = tokens["input"] * price_in / 1e6 + tokens["output"] * price_out / 1e6
    return {
        "calls": len(calls),
        "latency_p50_s": float(np.quantile(latency, 0.5)) if latency else None,
        "latency_p95_s": float(np.quantile(latency, 0.95)) if latency else None,
        "tokens_total": tokens,
        "tokens_mean": {k: v / len(calls) for k, v in tokens.items()} if calls else None,
        "cost_usd_list_price": round(cost, 4),
        "cost_per_call_usd": cost / len(calls) if calls else None,
    }


def evaluate(windows: list[dict], price_in: float, price_out: float) -> dict:
    """windows: episode, label, people, a_risk, record (B's cached answer or None)."""
    statuses = {"ok": 0, "invalid": 0, "unavailable": 0, "missing": 0}
    for w in windows:
        statuses[w["record"]["status"] if w["record"] else "missing"] += 1
    ok = [w for w in windows if w["record"] and w["record"]["status"] == "ok"]

    def groups(keep) -> list[dict]:
        by: dict[str, list[dict]] = {}
        for w in ok:
            if keep(w):
                by.setdefault(w["episode"], []).append(w)
        return [
            {
                "y": np.array([w["label"] for w in ws]),
                "b": np.array([w["record"]["result"]["risk"] for w in ws]),
                "a": np.array([w["a_risk"] for w in ws]),
            }
            for ws in by.values()
        ]

    episodes: dict[str, list[dict]] = {}
    for w in ok:
        episodes.setdefault(w["episode"], []).append(w)
    labels = np.array([ws[0]["label"] for ws in episodes.values()])
    levels: dict[str, int] = {}
    for w in ok:
        levels[w["record"]["result"]["level"]] = levels.get(w["record"]["result"]["level"], 0) + 1
    return {
        "windows": len(windows),
        "statuses": statuses,
        "valid_share": len(ok) / len(windows) if windows else None,
        "all": compare(groups(lambda w: True)),
        "with_people": compare(groups(lambda w: w["people"])),
        "empty": compare(groups(lambda w: not w["people"])),
        "episode_auc": {
            "clips": len(labels),
            "b": _auc(
                labels,
                np.array([np.mean([w["record"]["result"]["risk"] for w in ws]) for ws in episodes.values()]),
            ),
            "a": _auc(labels, np.array([np.mean([w["a_risk"] for w in ws]) for ws in episodes.values()])),
        },
        "levels_b": levels,
        "usage": usage_stats([w["record"] for w in windows], price_in, price_out),
    }


def _n(v, d: int = 3) -> str:
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def _ci(v, ci) -> str:
    return _n(v) + (f" [{_n(ci[0])}; {_n(ci[1])}]" if ci else "")


def render(result: dict) -> str:
    lines = [f"# Агрегатор B проти A на val — {result['created']}", ""]
    for key, title in (
        ("val120", "120 кліпів val (усі вікна, `sentinel eval b-run`)"),
        ("pilot", "117 вікон пілоту"),
    ):
        r = result[key]
        u = r["usage"]
        lines += [
            f"## {title}",
            "",
            f"Вікон {r['windows']}; статуси {r['statuses']}; валідних {_n(100 * r['valid_share'], 1)} %.",
            "",
            "| Вікна | Кліпів | Вікон (бійка) | AUC B [95 % ДІ] | AUC A [95 % ДІ] | B − A [95 % ДІ] |",
            "|---|---|---|---|---|---|",
        ]
        for sub, name in (("all", "усі"), ("with_people", "з людьми"), ("empty", "без людей")):
            c = r[sub]
            lines.append(
                f"| {name} | {c['clips']} | {c['windows']} ({c['fight_windows']}) | "
                f"{_ci(c['auc_b'], c['auc_b_ci'])} | "
                f"{_ci(c['auc_a'], c['auc_a_ci'])} | {_ci(c['b_minus_a'], c['b_minus_a_ci'])} |"
            )
        e = r["episode_auc"]
        lines += [
            "",
            f"AUC епізодів (середній ризик вікон, {e['clips']} кліпів): B {_n(e['b'])}, A {_n(e['a'])}. "
            f"Рівні B: {r['levels_b']}.",
            f"Виклики: {u['calls']}; затримка p50 / p95 {_n(u['latency_p50_s'], 2)} / "
            f"{_n(u['latency_p95_s'], 2)} с; "
            f"токенів на виклик (вхід / вихід / з них міркування) {u['tokens_mean']['input']:.0f} / "
            f"{u['tokens_mean']['output']:.0f} / {u['tokens_mean']['reasoning']:.0f}; "
            f"за прейскурантом Groq — ${_n(u['cost_usd_list_price'], 2)} "
            "(верхня оцінка; використано безкоштовний рівень).",
            "",
        ]
    lines += [
        "Інтервали — бутстреп за кліпами, 1000 повторів, seed 2026; B − A — парний (ті самі кліпи). "
        "Мітка вікна — мітка кліпу.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    from sentinel.aggregators.llm import PROMPT_VERSION, BAggregator, LlmCache
    from sentinel.aggregators.rules import make_aggregator
    from sentinel.eval.b_pilot import pilot_sample
    from sentinel.eval.b_run import pick_clips

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--clips", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path(f"results/b_val_{date.today().isoformat()}.json"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    provider = config.llm.providers[config.llm.provider]
    cache = LlmCache(config.llm.cache_dir, provider.model, PROMPT_VERSION)
    b = BAggregator(config, None, cache, config.llm.provider)  # keys only: no client, no request
    a = make_aggregator(config)
    val = load_episodes(args.episodes, "rwf2000", ["val"])

    def window(e, i: int, risk: float | None = None) -> dict:
        scene = e.scenes[i]
        return {
            "episode": e.episode_id,
            "label": e.label,
            "people": scene.summary.n_persons > 0,
            "a_risk": a.assess(scene).risk if risk is None else risk,
            "record": cache.get(b.key(scene)),
        }

    clips = pick_clips(val, args.clips, args.seed)
    run = [window(e, i) for e in clips for i in range(len(e.scenes))]
    by_id = {e.episode_id: e for e in val}
    pilot = [
        window(by_id[w["episode"]], w["window"], w["a_risk"]) for w in pilot_sample(val, a, 15, args.seed)
    ]
    pin, pout = provider.price_in_per_m, provider.price_out_per_m
    result = {
        "created": date.today().isoformat(),
        "prompt": PROMPT_VERSION,
        "model": provider.model,
        "price_per_m_usd": {"input": pin, "output": pout},
        "val120": evaluate(run, pin, pout),
        "pilot": evaluate(pilot, pin, pout),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    md = Path(f"docs/bench/b_val_{result['created']}.md")
    md.write_text(render(result), encoding="utf-8")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
