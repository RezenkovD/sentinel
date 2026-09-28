import json

import pytest
from pydantic import ValidationError

from sentinel.schema import (
    SCHEMA_VERSION,
    Interaction,
    Person,
    RiskResult,
    SceneDescription,
    Summary,
    Window,
    Zone,
)

# The example from SPEC, section 5 (plus the schema version the contract requires).
SPEC_EXAMPLE = {
    "schema_version": 1,
    "window": {"t_start": 132.0, "t_end": 133.0, "source": "cam-1"},
    "zone": {"type": "transport_stop", "hours": "night", "expected_density": "low"},
    "summary": {"n_persons": 3, "crowd_motion": 0.42, "weapon_visible": False},
    "persons": [
        {"id": 7, "speed": 0.8, "limb_accel": 3.1, "holds": None, "conf": 0.91},
        {"id": 9, "speed": 1.4, "limb_accel": 0.6, "holds": "knife", "conf": 0.64},
    ],
    "interactions": [
        {"pair": [7, 9], "distance_h": 0.45, "approach_speed": 1.2, "facing": True, "duration_s": 4.0}
    ],
}


def _scene(**overrides) -> SceneDescription:
    data = json.loads(json.dumps(SPEC_EXAMPLE)) | overrides
    return SceneDescription.model_validate(data)


def test_spec_example_round_trips_through_json_unchanged():
    scene = SceneDescription.model_validate(SPEC_EXAMPLE)
    assert json.loads(scene.model_dump_json()) == SPEC_EXAMPLE


def test_serialized_scene_carries_schema_version_1():
    scene = SceneDescription(
        window=Window(t_start=0.0, t_end=1.0, source="cam-1"),
        zone=Zone(type="street", hours="day", expected_density="high"),
        summary=Summary(n_persons=0, crowd_motion=0.0, weapon_visible=False),
    )
    assert SCHEMA_VERSION == 1
    assert json.loads(scene.model_dump_json())["schema_version"] == 1


def test_scene_with_other_schema_version_is_rejected():
    with pytest.raises(ValidationError):
        _scene(schema_version=2)


def test_floats_are_rounded_to_two_decimals():
    person = Person(id=1, speed=0.8149, limb_accel=3.105001, holds=None, conf=0.9051)
    assert (person.speed, person.limb_accel, person.conf) == (0.81, 3.11, 0.91)
    pair = Interaction(pair=(1, 2), distance_h=0.4449, approach_speed=1.23456, facing=True, duration_s=3.999)
    assert (pair.distance_h, pair.approach_speed, pair.duration_s) == (0.44, 1.23, 4.0)
    window = Window(t_start=132.00049, t_end=133.0061, source="cam-1")
    assert (window.t_start, window.t_end) == (132.0, 133.01)
    assert Summary(n_persons=2, crowd_motion=0.4249, weapon_visible=False).crowd_motion == 0.42


def test_rounded_json_text_is_short():
    person = Person(id=1, speed=0.123456789, limb_accel=2.0 / 3.0, holds=None, conf=0.87654321)
    assert person.model_dump_json() == '{"id":1,"speed":0.12,"limb_accel":0.67,"holds":null,"conf":0.88}'


def test_tiny_negative_value_serializes_as_plain_zero():
    # -0.0 and 0.0 must give the same JSON text, otherwise identical scenes miss the LLM cache.
    pair = Interaction(pair=(1, 2), distance_h=1.0, approach_speed=-0.001, facing=False, duration_s=0.0)
    assert '"approach_speed":0.0' in pair.model_dump_json()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_feature_is_rejected(bad):
    # approach_speed has no range bounds: only the finiteness check keeps NaN/inf out of the LLM prompt.
    with pytest.raises(ValidationError):
        Interaction(pair=(1, 2), distance_h=1.0, approach_speed=bad, facing=False, duration_s=0.0)


def test_scene_cannot_be_modified_after_creation():
    # A and B read the same object; neither may change what the other sees.
    scene = SceneDescription.model_validate(SPEC_EXAMPLE)
    with pytest.raises(ValidationError):
        scene.persons[0].speed = 5.0
    with pytest.raises(AttributeError):
        scene.persons.append(scene.persons[0])
    with pytest.raises(AttributeError):
        scene.interactions.clear()
    assert json.loads(scene.model_dump_json()) == SPEC_EXAMPLE


def test_holds_accepts_only_known_weapon_types():
    assert Person(id=1, speed=0, limb_accel=0, holds="pistol", conf=0.5).holds == "pistol"
    with pytest.raises(ValidationError):
        Person(id=1, speed=0, limb_accel=0, holds="umbrella", conf=0.5)


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ValidationError):
        Person(id=1, speed=0, limb_accel=0, holds=None, conf=1.2)


def test_empty_scene_is_valid():
    scene = _scene(
        summary={"n_persons": 0, "crowd_motion": 0.0, "weapon_visible": False}, persons=[], interactions=[]
    )
    assert scene.persons == () and scene.interactions == ()


def test_at_most_five_interactions():
    pair = {"pair": [1, 2], "distance_h": 1.0, "approach_speed": 0.0, "facing": False, "duration_s": 0.0}
    assert len(_scene(interactions=[pair] * 5).interactions) == 5
    with pytest.raises(ValidationError):
        _scene(interactions=[pair] * 6)


def test_unknown_field_is_rejected():
    # A misspelled feature must fail loudly instead of silently disappearing from the aggregators' input.
    with pytest.raises(ValidationError):
        Person.model_validate(
            {"id": 1, "speed": 0, "limb_acc": 3.0, "limb_accel": 0, "holds": None, "conf": 0.5}
        )


def test_risk_result_parses_strict_llm_json():
    result = RiskResult.model_validate_json(
        '{"risk": 0.8349, "level": "alarm", "reason_uk": "Бійка двох людей"}'
    )
    assert (result.risk, result.level, result.reason_uk) == (0.83, "alarm", "Бійка двох людей")


@pytest.mark.parametrize("bad", [{"risk": 1.3, "level": "alarm"}, {"risk": 0.5, "level": "panic"}])
def test_risk_result_rejects_out_of_range_risk_or_unknown_level(bad):
    with pytest.raises(ValidationError):
        RiskResult.model_validate(bad | {"reason_uk": "—"})
