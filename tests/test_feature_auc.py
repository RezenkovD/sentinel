"""Figure 2.2 data: A v1.1's features as A sees them, the dropped ones, the raw wrist acceleration."""

import pytest

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import CachedEpisode
from sentinel.eval.feature_auc import FIGURE_ORDER, figure_aucs
from sentinel.schema import Person, SceneDescription, Summary, Window, Zone

CONFIG = load_config(DEFAULT_CONFIG)


def _scene(t: float, speed: float | None, accel: float | None) -> SceneDescription:
    persons = (
        (Person(id=1, speed=speed, limb_accel=accel, holds=None, conf=0.9),) if speed is not None else ()
    )
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=Zone(type="street", hours="day", expected_density="medium"),
        summary=Summary(n_persons=len(persons), crowd_motion=speed, weapon_visible=False),
        persons=persons,
    )


def _episode(eid: str, label: int, speeds: list[float | None]) -> CachedEpisode:
    scenes = [_scene(float(t), s, None if s is None else 10 * s) for t, s in enumerate(speeds)]
    meta = {
        "episode_id": eid,
        "label": label,
        "part": "train",
        "perception_version": "3.0",
        "perception_hash": "h",
    }
    return CachedEpisode(meta, scenes, [{} for _ in scenes])


def test_every_window_counts_and_features_follow_table_2_3():
    eps = [_episode("f", 1, [1.0, 2.0, None]), _episode("c", 0, [0.1, 0.2, None])]
    data = figure_aucs(eps, CONFIG, {"f": [5.0, None, None], "c": [1.0, 2.0, None]})
    assert data["windows"] == {"fight": 3, "calm": 3, "all": 6}  # empty windows included, as A sees them
    assert [r["feature"] for r in data["features"]] == [name for name, _ in FIGURE_ORDER]
    assert len(data["features"]) == 17 and {r["feature"] for r in data["dropped"]} == {
        "weapon_visible", "weapon_held", "facing_close",
    }  # fmt: skip
    crowd = next(r for r in data["features"] if r["feature"] == "crowd_motion")
    # nulls at the neutral 0: (1, 2, 0) against (0.1, 0.2, 0) — 6 wins and one tie of 9 pairs
    assert crowd["auc"] == pytest.approx(6.5 / 9)
    raw = data["raw_wrist_accel"]
    assert raw["windows_known"] == {"fight": 1, "calm": 2}
    assert raw["on_windows_where_both_known"]["windows"] == 3


def test_a_raw_record_with_other_windows_is_refused():
    with pytest.raises(ValueError, match="raw windows"):
        figure_aucs([_episode("f", 1, [1.0, 2.0])], CONFIG, {"f": [1.0]})
