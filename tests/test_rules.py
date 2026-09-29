from pathlib import Path

import numpy as np
import pytest

from sentinel.aggregators.rules import FEATURE_NAMES, RulesAggregator, scene_to_features
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.schema import Interaction, Person, SceneDescription, Summary, Window, Zone

REPO = Path(__file__).parents[1]
CONFIG = load_config(REPO / DEFAULT_CONFIG)
ZONE = Zone(type="public_space", hours="day", expected_density="medium")


def _aggregator() -> RulesAggregator:
    return RulesAggregator.from_config(CONFIG)


def _scene(
    persons: list[Person] | None = None,
    interactions: list[Interaction] | None = None,
    weapon_visible: bool = False,
    zone: Zone = ZONE,
) -> SceneDescription:
    persons = persons if persons is not None else [_person(1), _person(2)]
    speeds = [p.speed for p in persons if p.speed is not None]
    summary = Summary(
        n_persons=len(persons),
        crowd_motion=float(np.mean(speeds)) if speeds else (0.0 if not persons else None),
        weapon_visible=weapon_visible,
    )
    return SceneDescription(
        window=Window(t_start=0.0, t_end=1.0, source="cam-1"),
        zone=zone,
        summary=summary,
        persons=tuple(persons),
        interactions=tuple(interactions or ()),
    )


def _person(
    pid: int, speed: float | None = 0.3, accel: float | None = 2.0, holds: str | None = None
) -> Person:
    return Person(id=pid, speed=speed, limb_accel=accel, holds=holds, conf=0.8)


def _pair(d: float = 1.5, approach: float | None = 0.0, facing: bool | None = False, contact: float = 0.0):
    return Interaction(pair=(1, 2), distance_h=d, approach_speed=approach, facing=facing, duration_s=contact)


def test_feature_order_is_fixed():
    # saved weights of the logistic regression are tied to this order
    assert FEATURE_NAMES == (
        "n_persons",
        "crowd_motion",
        "max_speed",
        "max_limb_accel",
        "min_distance_h",
        "max_approach_speed",
        "facing_close",
        "max_contact_s",
        "n_close_pairs",
        "weapon_visible",
        "weapon_held",
        "zone_prior",
        "limb_accel_missing",
    )


def test_features_are_read_from_the_scene_by_name():
    scene = _scene(
        persons=[_person(1, speed=0.4, accel=6.0), _person(2, speed=1.2, accel=1.0, holds="knife")],
        interactions=[_pair(d=0.5, approach=0.8, facing=True, contact=3.0)],
        weapon_visible=True,
    )
    x = dict(zip(FEATURE_NAMES, scene_to_features(scene, CONFIG), strict=True))
    assert (
        x["n_persons"] == 2
        and x["max_speed"] == pytest.approx(1.2)
        and x["max_limb_accel"] == pytest.approx(6.0)
    )
    assert x["min_distance_h"] == pytest.approx(0.5) and x["max_approach_speed"] == pytest.approx(0.8)
    assert (x["facing_close"], x["max_contact_s"], x["n_close_pairs"]) == (1.0, pytest.approx(3.0), 1.0)
    assert (x["weapon_visible"], x["weapon_held"], x["limb_accel_missing"]) == (1.0, 1.0, 0.0)


def test_missing_feature_becomes_neutral_value_with_an_indicator():
    scene = _scene(persons=[_person(1, speed=None, accel=None)])
    x = dict(zip(FEATURE_NAMES, scene_to_features(scene, CONFIG), strict=True))
    neutral = CONFIG.aggregator_a.neutral
    assert x["max_limb_accel"] == neutral["max_limb_accel"] and x["limb_accel_missing"] == 1.0
    assert x["max_speed"] == neutral["max_speed"]
    assert x["min_distance_h"] == neutral["min_distance_h"]  # nobody close: the "far apart" value


@pytest.mark.parametrize(
    ("calm", "tense"),
    [
        (
            _scene(persons=[_person(1, accel=2.0), _person(2)]),
            _scene(persons=[_person(1, accel=25.0), _person(2)]),
        ),
        (_scene(interactions=[_pair(d=1.8)]), _scene(interactions=[_pair(d=0.4)])),
        (
            _scene(interactions=[_pair(d=0.8, approach=0.0)]),
            _scene(interactions=[_pair(d=0.8, approach=1.5)]),
        ),
    ],
    # contact duration is not a sign of aggression on RWF windows (weight 0 since 29.09, docs/DECISIONS.md)
    ids=["wrist acceleration", "distance", "approach speed"],
)
def test_risk_grows_with_signs_of_aggression(calm: SceneDescription, tense: SceneDescription):
    a = _aggregator()
    assert a.assess(tense).risk > a.assess(calm).risk


def test_weapon_in_hand_overrides_to_alarm():
    a = _aggregator()
    result = a.assess(_scene(persons=[_person(1, holds="gun"), _person(2)]))
    assert result.risk >= CONFIG.incidents.theta_high + 0.1 - 1e-9
    assert result.level == "alarm"


def test_weapon_only_in_view_is_just_a_feature():
    a = _aggregator()
    seen = a.assess(_scene(weapon_visible=True))
    assert a.assess(_scene()).risk < seen.risk < CONFIG.incidents.theta_high + 0.1


def test_empty_scene_has_zero_risk():
    result = _aggregator().assess(_scene(persons=[]))
    assert (result.risk, result.level) == (0.0, "normal")


def test_reason_names_the_largest_contributions_in_ukrainian():
    scene = _scene(
        persons=[_person(1, accel=30.0), _person(2)],
        interactions=[_pair(d=0.4, approach=0.2, facing=False, contact=0.0)],
    )
    reason = _aggregator().assess(scene).reason_uk
    assert "0,4 зросту" in reason and "різкі рухи рук" in reason.lower()
    assert reason[0].isupper()


def test_calm_scene_reason_says_so():
    scene = _scene(persons=[_person(1, accel=0.5, speed=0.1)])
    result = _aggregator().assess(scene)
    assert result.level == "normal" and result.reason_uk


def test_fitted_model_survives_save_and_load(tmp_path: Path):
    rng = np.random.default_rng(0)
    x_calm = rng.normal(0.0, 1.0, size=(50, len(FEATURE_NAMES)))
    x_fight = x_calm + 3.0
    a = _aggregator()
    a.fit(np.vstack([x_calm, x_fight]), np.array([0] * 50 + [1] * 50))
    path = tmp_path / "a.json"
    a.save(path)
    b = RulesAggregator.load(path, CONFIG)
    scene = _scene(persons=[_person(1, accel=25.0), _person(2)], interactions=[_pair(d=0.4, contact=3.0)])
    assert b.assess(scene).risk == pytest.approx(a.assess(scene).risk)


def test_loading_weights_saved_for_other_features_fails(tmp_path: Path):
    path = tmp_path / "a.json"
    _aggregator().save(path)
    path.write_text(path.read_text().replace('"weapon_held"', '"weapon_in_hand"'))
    with pytest.raises(ValueError):
        RulesAggregator.load(path, CONFIG)


def test_reason_mentions_fast_motion_once():
    scene = _scene(persons=[_person(1, speed=3.0, accel=1.0), _person(2, speed=2.0, accel=1.0)])
    reason = _aggregator().assess(scene).reason_uk.lower()
    assert reason.count("швидк") == 1


def test_wrist_acceleration_at_pose_noise_level_is_not_given_as_a_reason():
    # ~10 heights/s² is what pose jitter alone produces at 10 fps (sanity check, 29.09)
    scene = _scene(persons=[_person(1, speed=0.1, accel=10.0), _person(2, speed=0.1, accel=10.0)])
    assert "різкі рухи рук" not in _aggregator().assess(scene).reason_uk.lower()
