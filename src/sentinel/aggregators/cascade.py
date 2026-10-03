"""Cascade: A on every window, B only where A is unsure or the incident automaton is about to change state.

SPEC §3 and the fixed decision "B only in the zone of uncertainty and on incident transitions". A window asks
B when A's risk lies in the grey zone [g_low, g_high], or when A's risk would move the automaton: NORMAL with
a risk >= θ_high (an alarm may start), ALARM with a risk < θ_low (it may calm down), and every window of the
uncertain states ATTENTION and COOLDOWN. The window's risk is B's answer when there is one, else A's.

Live: B runs in the background (`BWorker`); its answer is applied when it arrives, the automaton never waits.
Experiment (from the cache): the answer for window t reaches the automaton at window t + ceil(latency) — its
real latency from the call record — so the time to an alarm stays honest. A window that asked B but has no
answer in the cache is listed in `missing`: counted and shown, never silently replaced by A.
"""

import math
from dataclasses import dataclass, field

from sentinel.config import IncidentsConfig
from sentinel.incidents import IncidentEvent, IncidentManager, State
from sentinel.schema import RiskResult, SceneDescription


def needs_b(a_risk: float, state: State, cfg: IncidentsConfig, g_low: float, g_high: float) -> bool:
    if g_low <= a_risk <= g_high or state in (State.ATTENTION, State.COOLDOWN):
        return True
    if state is State.NORMAL:
        return a_risk >= cfg.theta_high
    return a_risk < cfg.theta_low  # ALARM


@dataclass
class CascadeRun:
    risks: list[RiskResult]  # what the automaton got for each window
    calls: int  # windows that asked B
    missing: list[int]  # windows that asked B without an answer in the cache
    events: list[IncidentEvent] = field(default_factory=list)
    from_b: list[bool] = field(default_factory=list)  # the window's risk came from B


def simulate(
    scenes: list[SceneDescription],
    a: list[RiskResult],
    b: dict[int, tuple[RiskResult, float]],
    cfg: IncidentsConfig,
    g_low: float,
    g_high: float,
) -> CascadeRun:
    """One stream of windows (1 s apart) through the cascade and the automaton; `b` maps a window index to B's
    cached answer and its latency in seconds."""
    manager = IncidentManager(cfg, "cam-1", lambda *_: None, "cascade")
    arriving: dict[int, list[tuple[int, RiskResult]]] = {}
    run = CascadeRun([], 0, [])
    for i, (scene, a_result) in enumerate(zip(scenes, a, strict=True)):
        if needs_b(a_result.risk, manager.state, cfg, g_low, g_high):
            run.calls += 1
            if i in b:
                answer, latency = b[i]
                arriving.setdefault(i + max(1, math.ceil(latency)), []).append((i, answer))
            else:
                run.missing.append(i)
        landed = arriving.pop(i, [])
        risk = max(landed, key=lambda x: x[0])[1] if landed else a_result
        run.risks.append(risk)
        run.from_b.append(bool(landed))
        run.events += manager.on_window(scene.window.t_end, risk, scene)
    run.events += manager.finish(scenes[-1].window.t_end) if scenes else []
    return run
