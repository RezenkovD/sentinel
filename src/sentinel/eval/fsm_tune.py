"""θ_low and M of the incident automaton on compilations of our val clips (decision 30.09, report 03, item 7).

    uv run sentinel eval fsm-tune            # -> results/fsm_tuning_<date>.md

A long camera stream is simulated from the episode cache of our RWF val, seeded: a few calm (NonFight) clips,
then fights (2–4 Fight clips in a row) with calm gaps between them. A — the model in the config — scores every
window once; the incident automaton runs over each stream for every θ_low × M of the grid, θ_high fixed.

A fight is **handled** when an incident is open during it, closes within M + 5 s after the fight's end and
does not flicker: no close before the fight is over and no second opening (new or merged) until M + 5 s after
its end. The best pair has the largest share of handled fights among the detected ones, then the fewest
incidents opened in calm video, then the shorter M. Only our val part is used.
"""

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, IncidentsConfig, load_config
from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, check_complete, load_episodes
from sentinel.incidents import IncidentEvent, IncidentManager
from sentinel.schema import RiskResult, SceneDescription

THETA_LOW_GRID = (0.43, 0.45, 0.5, 0.55, 0.6, 0.65)
QUIET_GRID = (3.0, 5.0, 7.0, 10.0)
GRACE_S = 5.0  # "closes within M + 5 s after the end of the fight"

Scored = tuple[RiskResult, SceneDescription]  # a window of a cached clip with A's answer


@dataclass(frozen=True)
class Fight:
    start: float
    end: float
    next_start: float  # start of the next fight, or the end of the stream


@dataclass
class Stream:
    windows: list[tuple[float, RiskResult, SceneDescription]]  # (window end, risk, scene) in stream time
    fights: list[Fight]
    calm_s: float  # seconds of calm video in the stream


@dataclass(frozen=True)
class FightOutcome:
    detected: bool  # an incident was open at some moment of the fight
    closed_in_time: bool  # it closed within M + GRACE_S after the fight's end
    flicker: bool  # closed before the fight was over, or opened again soon after
    close_delay_s: float | None  # first close after the fight's end, relative to that end


def _shift(scene: SceneDescription, offset: float, id_offset: int) -> SceneDescription:
    """The scene placed at `offset` seconds of the stream, with track ids that no other clip uses."""
    window = scene.window.model_copy(
        update={"t_start": scene.window.t_start + offset, "t_end": scene.window.t_end + offset}
    )
    persons = tuple(p.model_copy(update={"id": p.id + id_offset}) for p in scene.persons)
    pairs = tuple(
        i.model_copy(update={"pair": (i.pair[0] + id_offset, i.pair[1] + id_offset)})
        for i in scene.interactions
    )
    return scene.model_copy(update={"window": window, "persons": persons, "interactions": pairs})


def build_streams(
    fights: list[list[Scored]],
    calms: list[list[Scored]],
    n: int,
    seed: int,
    fights_per_stream: int = 3,
    lead_clips: int = 3,
    fight_clips: tuple[int, int] = (2, 4),
    gap_clips: tuple[int, int] = (3, 9),
) -> list[Stream]:
    """`n` streams: `lead_clips` calm clips, then fights of `fight_clips` Fight clips, each followed by a calm
    gap of `gap_clips` clips (ranges inclusive). No clip repeats within a stream."""
    rng = random.Random(seed)
    streams = []
    for _ in range(n):
        fight_pool, calm_pool = rng.sample(fights, len(fights)), rng.sample(calms, len(calms))
        plan: list[tuple[str, list[Scored]]] = [("calm", calm_pool.pop()) for _ in range(lead_clips)]
        for _ in range(fights_per_stream):
            plan += [("fight", fight_pool.pop()) for _ in range(rng.randint(*fight_clips))]
            plan += [("calm", calm_pool.pop()) for _ in range(rng.randint(*gap_clips))]
        windows, spans, offset, calm_s = [], [], 0.0, 0.0
        for index, (kind, clip) in enumerate(plan):
            for risk, scene in clip:
                shifted = _shift(scene, offset, 1000 * (index + 1))
                windows.append((shifted.window.t_end, risk, shifted))
            length = clip[-1][1].window.t_end
            if kind == "fight":
                if spans and spans[-1][1] == offset:
                    spans[-1] = (spans[-1][0], offset + length)  # fight clips in a row are one fight
                else:
                    spans.append((offset, offset + length))
            else:
                calm_s += length
            offset += length
        starts = [s for s, _ in spans[1:]] + [offset]
        streams.append(
            Stream(windows, [Fight(s, e, nxt) for (s, e), nxt in zip(spans, starts, strict=True)], calm_s)
        )
    return streams


def run_stream(stream: Stream, cfg: IncidentsConfig) -> list[IncidentEvent]:
    manager = IncidentManager(cfg, "cam-1", lambda *_: None, "fsm-tune")
    events = []
    for t, risk, scene in stream.windows:
        events += manager.on_window(t, risk, scene)
    return events + manager.finish(stream.windows[-1][0])


def _intervals(events: list[IncidentEvent], end: float) -> list[tuple[float, float, bool]]:
    """(open, close, closed by calm) of every opening — a new incident, or the previous one re-opened (a
    merged update); one still open at the end of the stream closes there."""
    spans, opened = [], None
    for event in events:
        if event.kind == "open" or (event.kind == "update" and event.merged):
            opened = event.t
        elif event.kind == "close" and opened is not None:
            spans.append((opened, event.t, not event.stream_ended))
            opened = None
    if opened is not None:
        spans.append((opened, end, False))
    return spans


def fight_outcomes(events: list[IncidentEvent], stream: Stream, quiet_s: float) -> list[FightOutcome]:
    spans = _intervals(events, stream.windows[-1][0])
    out = []
    for fight in stream.fights:
        horizon = fight.end + quiet_s + GRACE_S
        during = [s for s in spans if s[0] <= fight.end and s[1] >= fight.start]
        near = [s for s in spans if s[0] <= horizon and s[1] >= fight.start]
        closes = [c for o, c, calm in near if calm and c >= fight.end]
        delay = min(closes) - fight.end if closes else None
        flicker = any(fight.start <= c < fight.end for _, c, _ in near) or len(near) > 1
        out.append(
            FightOutcome(
                detected=bool(during),
                closed_in_time=bool(during) and delay is not None and delay <= quiet_s + GRACE_S,
                flicker=flicker,
                close_delay_s=delay,
            )
        )
    return out


def calm_incidents(events: list[IncidentEvent], stream: Stream, quiet_s: float) -> int:
    """Incidents opened in calm video: not during a fight nor within M + 5 s after one."""
    spans = _intervals(events, stream.windows[-1][0])
    return sum(not any(f.start <= o <= f.end + quiet_s + GRACE_S for f in stream.fights) for o, _, _ in spans)


def evaluate(streams: list[Stream], cfg: IncidentsConfig) -> dict:
    outcomes, calm_opened = [], 0
    for stream in streams:
        events = run_stream(stream, cfg)
        outcomes += fight_outcomes(events, stream, cfg.quiet_s)
        calm_opened += calm_incidents(events, stream, cfg.quiet_s)
    detected = [o for o in outcomes if o.detected]
    delays = [o.close_delay_s for o in detected if o.close_delay_s is not None]
    handled = [o for o in detected if o.closed_in_time and not o.flicker]
    calm_h = sum(s.calm_s for s in streams) / 3600
    return {
        "theta_low": cfg.theta_low,
        "quiet_s": cfg.quiet_s,
        "fights": len(outcomes),
        "detected": len(detected) / len(outcomes),
        "handled": len(handled) / len(detected) if detected else 0.0,
        "closed_in_time": sum(o.closed_in_time for o in detected) / len(detected) if detected else 0.0,
        "flicker": sum(o.flicker for o in detected) / len(detected) if detected else 0.0,
        "median_close_delay_s": float(np.median(delays)) if delays else None,
        "calm_incidents_per_h": calm_opened / calm_h if calm_h else 0.0,
    }


def choose(rows: list[dict]) -> dict:
    """Most handled fights; then fewest incidents in calm video; then the shorter M; then the lower θ_low."""
    return min(
        rows, key=lambda r: (-round(r["handled"], 9), r["calm_incidents_per_h"], r["quiet_s"], r["theta_low"])
    )


def scored_clips(episodes: list[CachedEpisode], aggregator) -> tuple[list[list[Scored]], list[list[Scored]]]:
    fights, calms = [], []
    for episode in episodes:
        clip = [(aggregator.assess(scene), scene) for scene in episode.scenes]
        if clip:
            (fights if episode.label == 1 else calms).append(clip)
    return fights, calms


def replace_cfg(cfg: IncidentsConfig, theta_low: float, quiet_s: float) -> IncidentsConfig:
    return cfg.model_copy(update={"theta_low": theta_low, "quiet_s": quiet_s})


def _theta(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _pct(value: float) -> str:
    return f"{100 * value:.1f}".replace(".", ",") + " %"


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}".replace(".", ",")


def main(argv: list[str] | None = None) -> None:
    from sentinel.aggregators.rules import make_aggregator

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--streams", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--theta-low", type=float, nargs="+", default=list(THETA_LOW_GRID))
    parser.add_argument("--quiet-s", type=float, nargs="+", default=list(QUIET_GRID))
    parser.add_argument("--out", type=Path, default=Path(f"results/fsm_tuning_{date.today().isoformat()}.md"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    val = load_episodes(args.episodes, "rwf2000", ["val"])
    check_complete(val, "rwf2000", ["val"], allow_partial=args.allow_partial)
    aggregator = make_aggregator(config)
    fights, calms = scored_clips(val, aggregator)
    streams = build_streams(fights, calms, args.streams, args.seed)
    base = config.incidents
    rows = [
        evaluate(streams, replace_cfg(base, theta_low, quiet_s))
        for theta_low in args.theta_low
        for quiet_s in args.quiet_s
        if theta_low < base.theta_high
    ]
    best = choose(rows)
    current = next(
        (
            r
            for r in rows
            if np.isclose(r["theta_low"], base.theta_low) and np.isclose(r["quiet_s"], base.quiet_s)
        ),
        None,
    )
    video_h = sum(s.windows[-1][0] for s in streams) / 3600
    lines = [
        f"# Автомат інцидентів: θ_low і M на склейках val — {date.today().isoformat()}",
        "",
        f"Склейок {len(streams)} (seed {args.seed}), {video_h:.1f} год відео з кліпів нашого val RWF "
        f"({len(fights)} Fight, {len(calms)} NonFight): 3 спокійні кліпи, далі тричі бійка з 2–4 кліпів "
        "Fight і спокійна пауза з 3–9 кліпів NonFight. Ризик вікон — A з конфігу "
        f"(`{config.aggregator_a.model}`); θ_high = {_theta(base.theta_high)} не змінюється; k = {base.k} з "
        f"n = {base.n}.",
        "",
        "Бійка **опрацьована**, якщо під час неї відкрито інцидент, він закрився не пізніше M + 5 с після "
        "кінця бійки й не мерехтів (не закрився до кінця бійки й не відкрився вдруге до M + 5 с після неї). "
        "Частки опрацьованих, вчасно закритих і мерехтливих — серед виявлених бійок.",
        "",
        "| θ_low | M, с | виявлено бійок | опрацьовано | закрито вчасно | мерехтіння "
        "| медіана закриття після бійки, с | інцидентів у спокої / год |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        mark = " **(обрано)**" if r is best else (" (зараз)" if r is current else "")
        lines.append(
            f"| {_theta(r['theta_low'])}{mark} | {_num(r['quiet_s'])} | {_pct(r['detected'])} "
            f"| {_pct(r['handled'])} | {_pct(r['closed_in_time'])} | {_pct(r['flicker'])} "
            f"| {_num(r['median_close_delay_s'])} | {_num(r['calm_incidents_per_h'])} |"
        )
    lines += [
        "",
        f"Обрано: θ_low = {_theta(best['theta_low'])}, M = {_num(best['quiet_s'])} с — найбільша "
        "частка опрацьованих бійок, далі найменше інцидентів у спокої, далі коротше M.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({"rows": rows, "best": best}, indent=1),
        encoding="utf-8",
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
