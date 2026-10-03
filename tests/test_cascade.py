"""Cascade: A always, B in A's grey zone or when the automaton is about to change state (session 05)."""

from sentinel.aggregators.cascade import needs_b, simulate
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.incidents import State
from sentinel.schema import RiskResult, SceneDescription, Summary, Window, Zone

CFG = load_config(DEFAULT_CONFIG).incidents  # θ_high 0.71, θ_low 0.65


def _r(risk: float) -> RiskResult:
    return RiskResult(risk=risk, level="normal", reason_uk="—")


def _scenes(n: int) -> list[SceneDescription]:
    return [
        SceneDescription(
            window=Window(t_start=float(i), t_end=float(i + 1), source="cam-1"),
            zone=Zone(type="street", hours="day", expected_density="medium"),
            summary=Summary(n_persons=0, crowd_motion=0.0, weapon_visible=False),
        )
        for i in range(n)
    ]


def test_b_is_asked_in_the_grey_zone():
    assert needs_b(0.5, State.NORMAL, CFG, g_low=0.4, g_high=0.6)
    assert not needs_b(0.2, State.NORMAL, CFG, g_low=0.4, g_high=0.6)


def test_b_is_asked_when_a_would_change_the_automatons_state():
    assert needs_b(0.9, State.NORMAL, CFG, g_low=0.4, g_high=0.6)  # would open ATTENTION
    assert not needs_b(0.9, State.ALARM, CFG, g_low=0.4, g_high=0.6)  # keeps the alarm
    assert needs_b(0.1, State.ALARM, CFG, g_low=0.4, g_high=0.6)  # would start calming down
    assert needs_b(0.1, State.ATTENTION, CFG, g_low=0.4, g_high=0.6)  # ATTENTION and COOLDOWN always
    assert needs_b(0.9, State.COOLDOWN, CFG, g_low=0.4, g_high=0.6)


def test_a_b_answer_counts_only_once_it_could_have_arrived():
    """The answer for window t is there for the automaton at window t + ceil(latency): decided at its end."""
    a = [_r(0.5), _r(0.5), _r(0.5), _r(0.5)]
    b = {0: (_r(0.95), 1.4), 2: (_r(0.1), 0.3)}
    run = simulate(_scenes(4), a, b, CFG, g_low=0.4, g_high=0.6)
    assert [round(r.risk, 2) for r in run.risks] == [0.5, 0.5, 0.95, 0.1]
    assert run.calls == 4 and run.missing == [1, 3]  # asked, but no answer in the cache: counted, never A


def test_of_two_answers_arriving_together_the_newer_one_counts():
    a = [_r(0.5)] * 3
    b = {0: (_r(0.9), 1.6), 1: (_r(0.2), 0.4)}  # both arrive at window 2
    assert round(simulate(_scenes(3), a, b, CFG, 0.4, 0.6).risks[2].risk, 2) == 0.2


def test_windows_outside_the_grey_zone_never_ask_b():
    a = [_r(0.05), _r(0.05)]
    run = simulate(_scenes(2), a, {}, CFG, g_low=0.4, g_high=0.6)
    assert run.calls == 0 and run.missing == [] and [r.risk for r in run.risks] == [0.05, 0.05]
