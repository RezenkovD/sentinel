"""B on every window of whole val clips, for the cascade grid and fsm-tune of B (session 05, step 4).

    uv run sentinel eval b-run --clips 120        # resumable: answers go through B's cache, limits waited out

A stratified, seeded subsample of our val clips (half Fight, half NonFight; a smaller run is part of a larger
one), or every val clip with --clips 0. Each window is answered once with the frozen prompt; a missing answer
stays missing (counted at the end), it is never replaced by A.
"""

import argparse
import random
from pathlib import Path

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, load_episodes


def pick_clips(episodes: list[CachedEpisode], n: int, seed: int) -> list[CachedEpisode]:
    if any(e.meta["part"] != "val" for e in episodes):
        raise ValueError("B for the grey zone runs on our val clips only")
    out = []
    for label in (1, 0):
        pool = sorted((e for e in episodes if e.label == label), key=lambda e: e.episode_id)
        random.Random(f"{seed}-{label}").shuffle(pool)
        out += pool[: n // 2]
    return sorted(out, key=lambda e: e.episode_id)


def main(argv: list[str] | None = None) -> None:
    from sentinel.aggregators.llm import PROMPT_VERSION, make_b
    from sentinel.config import DEFAULT_CONFIG, load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--clips", type=int, default=120, help="0 = every val clip")
    parser.add_argument("--prompt", default=PROMPT_VERSION)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    val = load_episodes(args.episodes, "rwf2000", ["val"])
    clips = val if args.clips == 0 else pick_clips(val, args.clips, args.seed)
    b = make_b(config, wait_on_limits=True, prompt_version=args.prompt)
    windows = [s for e in clips for s in e.scenes]
    counts = {"ok": 0, "invalid": 0, "unavailable": 0}
    for n, scene in enumerate(windows, 1):
        answer = b.answer(scene)
        counts[answer.status] = counts.get(answer.status, 0) + 1
        if n % 20 == 0 or n == len(windows):
            print(f"{n}/{len(windows)} windows of {len(clips)} clips: {counts}", flush=True)


if __name__ == "__main__":
    main()
