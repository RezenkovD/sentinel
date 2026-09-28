"""Scene description contract shared by all agents (SPEC, section 5): pydantic models serialized to JSON.

Both aggregators receive exactly the same SceneDescription: A turns it into a feature vector, B gets its JSON
as text. Distances and speeds are normalized by the person's height in the frame; every float is rounded to
2 decimals so the JSON stays short and identical scenes hit the LLM cache. Changing any field means bumping
SCHEMA_VERSION so that old experiment results stay reproducible.
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

WeaponType = Literal["pistol", "knife"]
AlertLevel = Literal["normal", "attention", "alarm"]


def _round2(value: float) -> float:
    # Adding 0.0 turns -0.0 into 0.0, so that tiny negative values serialize like zero.
    return round(value, 2) + 0.0


Rounded = Annotated[float, AfterValidator(_round2)]
NonNegative = Annotated[float, Field(ge=0), AfterValidator(_round2)]
Probability = Annotated[float, Field(ge=0, le=1), AfterValidator(_round2)]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Window(_Contract):
    t_start: NonNegative
    t_end: NonNegative
    source: str


class Zone(_Contract):
    type: str
    hours: str
    expected_density: str


class Summary(_Contract):
    n_persons: int = Field(ge=0)
    crowd_motion: NonNegative
    weapon_visible: bool


class Person(_Contract):
    id: int
    speed: NonNegative
    limb_accel: NonNegative
    holds: WeaponType | None
    conf: Probability


class Interaction(_Contract):
    pair: tuple[int, int]
    distance_h: NonNegative
    approach_speed: Rounded  # negative when the two people move apart
    facing: bool
    duration_s: NonNegative


class SceneDescription(_Contract):
    schema_version: Literal[1] = SCHEMA_VERSION
    window: Window
    zone: Zone
    summary: Summary
    # Tuples, not lists: frozen=True alone would still allow in-place append/sort on shared input.
    persons: tuple[Person, ...] = ()
    interactions: tuple[Interaction, ...] = Field(default=(), max_length=5)


class RiskResult(_Contract):
    """Output of an aggregator; for B it is exactly the strict JSON `{risk, level, reason_uk}`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    risk: Probability
    level: AlertLevel
    reason_uk: str
