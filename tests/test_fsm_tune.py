"""θ_low and M of the incident automaton on compilations of val clips (decision 30.09, report 03, item 7)."""

import numpy as np

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.fsm_tune import Fight, Stream, build_streams, choose, fight_outcomes, run_stream
from sentinel.schema import Person, RiskResult, SceneDescription, Summary, Window, Zone

CONFIG = load_config(DEFAULT_CONFIG)
ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(t: float, pid: int = 1) -> SceneDescription:
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(n_persons=1, crowd_motion=0.1, weapon_visible=False),
        persons=(Person(id=pid, speed=0.1, limb_accel=None, holds=None, conf=0.8),),
    )


def _clip(risks: list[float]) -> list[tuple[RiskResult, SceneDescription]]:
    """One cached clip: a scene per 1 s window with its risk from A."""
    return [
        (RiskResult(risk=r, level="normal", reason_uk="—"), _scene(float(i))) for i, r in enumerate(risks)
    ]


def _stream(risks: list[float], fights: list[tuple[float, float]]) -> Stream:
    windows = [(float(i + 1), *pair) for i, pair in enumerate(_clip(risks))]
    ends = [f[0] for f in fights[1:]] + [float(len(risks))]
    return Stream(windows, [Fight(s, e, n) for (s, e), n in zip(fights, ends, strict=True)], calm_s=0.0)


def _cfg(theta_low: float, quiet_s: float):
    return CONFIG.incidents.model_copy(update={"theta_low": theta_low, "quiet_s": quiet_s})


def test_streams_put_clips_back_to_back_with_their_own_track_ids():
    calm, fight = _clip([0.1] * 5), _clip([0.9] * 5)
    stream = build_streams(
        [fight, fight],
        [calm] * 6,
        n=1,
        seed=1,
        fights_per_stream=1,
        lead_clips=2,
        fight_clips=(2, 2),
        gap_clips=(3, 3),
    )[0]
    ends = [t for t, _, _ in stream.windows]
    assert ends == [float(i) for i in range(1, len(ends) + 1)]  # 1 s windows without holes or overlaps
    assert stream.fights == [Fight(10.0, 20.0, 35.0)] and stream.calm_s == 25.0
    ids = [s.persons[0].id for _, _, s in stream.windows]
    assert len(set(ids)) == 7  # one id per clip: a person of one clip is never "the same" in the next
    assert [s.window.t_start for _, _, s in stream.windows][:6] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_a_fight_that_closes_soon_after_its_end_is_handled():
    risks = [0.1] * 10 + [0.9] * 10 + [0.1] * 25
    stream = _stream(risks, [(10.0, 20.0)])
    (outcome,) = fight_outcomes(run_stream(stream, _cfg(0.5, 5.0)), stream, quiet_s=5.0)
    assert outcome.detected and outcome.closed_in_time and not outcome.flicker
    assert 5.0 <= outcome.close_delay_s <= 10.0


def test_a_close_in_the_middle_of_the_fight_is_flicker():
    risks = [0.1] * 10 + [0.9] * 5 + [0.2] * 6 + [0.9] * 5 + [0.1] * 25
    stream = _stream(risks, [(10.0, 26.0)])
    short, long = (
        fight_outcomes(run_stream(stream, _cfg(0.5, m)), stream, quiet_s=m)[0] for m in (3.0, 10.0)
    )
    assert short.flicker and not long.flicker  # a longer M rides through the pause


def test_reopening_soon_after_a_timely_close_is_flicker():
    """The previous incident re-opened (a merged update, same Telegram message) right after it closed."""
    risks = [0.1] * 10 + [0.9] * 10 + [0.1] * 5 + [0.9] * 3 + [0.1] * 30
    stream = _stream(risks, [(10.0, 20.0)])
    (outcome,) = fight_outcomes(run_stream(stream, _cfg(0.5, 3.0)), stream, quiet_s=3.0)
    assert outcome.closed_in_time and outcome.flicker


def test_lingering_medium_risk_after_the_fight_is_a_late_close():
    risks = [0.1] * 10 + [0.9] * 10 + [0.6] * 12 + [0.1] * 25
    stream = _stream(risks, [(10.0, 20.0)])
    low, high = (
        fight_outcomes(run_stream(stream, _cfg(t, 5.0)), stream, quiet_s=5.0)[0] for t in (0.5, 0.65)
    )
    assert not low.closed_in_time and high.closed_in_time  # 0.6 keeps a θ_low of 0.5 from calming down


def test_choice_prefers_handled_fights_then_fewer_calm_incidents_then_the_shorter_wait():
    rows = [
        {"theta_low": 0.5, "quiet_s": 10.0, "handled": 0.8, "calm_incidents_per_h": 1.0},
        {"theta_low": 0.6, "quiet_s": 5.0, "handled": 0.9, "calm_incidents_per_h": 3.0},
        {"theta_low": 0.55, "quiet_s": 7.0, "handled": 0.9, "calm_incidents_per_h": 2.0},
        {"theta_low": 0.45, "quiet_s": 5.0, "handled": 0.9, "calm_incidents_per_h": 2.0},
    ]
    assert choose(rows) == rows[3]
    assert np.isclose(choose(rows[:3])["theta_low"], 0.55)
