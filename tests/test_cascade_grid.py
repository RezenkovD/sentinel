"""Grey-zone grid of the cascade on val clips from the caches of A and B (session 05, step 4)."""

import pytest

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cascade_grid import choose_zone, grid
from sentinel.schema import RiskResult, SceneDescription, Summary, Window, Zone

CFG = load_config(DEFAULT_CONFIG).incidents


def _r(risk: float) -> RiskResult:
    return RiskResult(risk=risk, level="normal", reason_uk="—")


def _clip(label: int, a: list[float], b: list[float]) -> dict:
    scenes = [
        SceneDescription(
            window=Window(t_start=float(i), t_end=float(i + 1), source="cam-1"),
            zone=Zone(type="street", hours="day", expected_density="medium"),
            summary=Summary(n_persons=1, crowd_motion=0.1, weapon_visible=False),
        )
        for i in range(len(a))
    ]
    return {
        "label": label,
        "scenes": scenes,
        "a": [_r(x) for x in a],
        "b": {i: (_r(x), 0.5) for i, x in enumerate(b)},
    }


def test_the_grid_reports_the_share_sent_to_b_and_the_episode_quality():
    clips = [_clip(1, [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]), _clip(0, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1])]
    rows = grid(clips, CFG, g_lows=(0.4,), g_highs=(0.6,), rule="mean")
    (row,) = rows
    assert row["b_share"] == pytest.approx(1.0) and row["episode_auc"] == pytest.approx(1.0)
    none = grid(clips, CFG, g_lows=(0.55,), g_highs=(0.56,), rule="mean")[0]
    assert none["b_share"] == 0.0 and none["episode_auc"] == pytest.approx(0.5)  # A alone ties


def test_missing_b_answers_are_counted_not_filled_in():
    clip = _clip(1, [0.5, 0.5], [])
    (row,) = grid([clip, _clip(0, [0.1, 0.1], [])], CFG, g_lows=(0.4,), g_highs=(0.6,), rule="mean")
    assert row["missing"] == 2


def test_the_zone_is_the_best_auc_within_the_share_of_calls():
    rows = [
        {"g_low": 0.3, "g_high": 0.7, "b_share": 0.4, "episode_auc": 0.95, "missing": 0},
        {"g_low": 0.4, "g_high": 0.6, "b_share": 0.2, "episode_auc": 0.90, "missing": 0},
        {"g_low": 0.45, "g_high": 0.55, "b_share": 0.1, "episode_auc": 0.88, "missing": 0},
    ]
    assert choose_zone(rows, max_share=0.25)["g_low"] == 0.4
