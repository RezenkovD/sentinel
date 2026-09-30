"""Baselines on the episode cache (session 03, step 6)."""

import pytest

from sentinel.eval.baselines import motion_baseline, weapon_baseline
from sentinel.eval.cache import CachedEpisode
from sentinel.schema import SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(t: float, energy: float | None) -> SceneDescription:
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(n_persons=0, crowd_motion=0.0, motion_energy=energy, weapon_visible=False),
    )


def _episode(
    label: int, confs: list[float], labels=None, energies=None, part: str = "select"
) -> CachedEpisode:
    energies = energies or [1.0] * len(confs)
    scenes = [_scene(float(t), e) for t, e in enumerate(energies)]
    aux = [
        {"weapon_max_conf": c, "label": None if labels is None else labels[t]} for t, c in enumerate(confs)
    ]
    return CachedEpisode({"episode_id": f"{part}/{label}", "label": label, "part": part}, scenes, aux)


def test_weapon_baseline_runs_at_the_detector_operating_point_and_reports_the_strict_one():
    sevilla = [_episode(1, [0.2, 0.6, 0.35], labels=[1, 1, 1]), _episode(1, [0.1, 0.45], labels=[0, 1])]
    normal = [_episode(0, [0.1, 0.25], part="train") for _ in range(99)] + [_episode(0, [0.4], part="train")]
    result = weapon_baseline(
        sevilla, normal, operating_point=0.3, max_false_alarm=0.01, grid=(0.2, 0.3, 0.4, 0.5)
    )
    # chosen: the detector's own operating point (dev-chosen on Sevilla frames); 3 of 4 weapon windows
    assert result["threshold"] == pytest.approx(0.3)
    assert result["window_recall"] == pytest.approx(3 / 4) and result["segment_recall"] == pytest.approx(1.0)
    # the strict threshold (lowest with <= 1 % of normal clips alarming) is reported next to it
    assert result["strict"]["threshold"] == pytest.approx(0.3)
    assert result["strict"]["false_alarm_share"] == pytest.approx(0.01)


def test_strict_threshold_may_leave_no_recall():
    sevilla = [_episode(1, [0.3, 0.4], labels=[1, 1])]
    normal = [_episode(0, [0.5], part="train") for _ in range(10)]
    result = weapon_baseline(sevilla, normal, operating_point=0.3, max_false_alarm=0.01, grid=(0.3, 0.6))
    assert result["strict"]["threshold"] == pytest.approx(0.6) and result["strict"]["window_recall"] == 0.0
    assert result["threshold"] == pytest.approx(0.3) and result["false_alarm_share"] == pytest.approx(1.0)


def test_motion_threshold_maximises_episode_f1_on_val():
    val = [
        _episode(1, [0.0, 0.0], energies=[3.0, 9.0], part="val"),
        _episode(1, [0.0, 0.0], energies=[8.0, 2.0], part="val"),
        _episode(0, [0.0, 0.0], energies=[2.0, 4.0], part="val"),
        _episode(0, [0.0, 0.0], energies=[None, 1.0], part="val"),
    ]
    result = motion_baseline(val)
    assert result["threshold"] == pytest.approx(8.0) and result["f1"] == pytest.approx(1.0)
    assert result["episode_auc"] == pytest.approx(1.0)
