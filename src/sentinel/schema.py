"""Scene description contract shared by all agents (SPEC, section 5): pydantic models serialized to JSON.

Both aggregators receive exactly the same SceneDescription: A turns it into a feature vector, B gets its JSON
as text. Distances and speeds are normalized by the person's height in the frame; every float is rounded to
2 decimals so the JSON stays short and identical scenes hit the LLM cache. Changing any field means bumping
SCHEMA_VERSION so that old experiment results stay reproducible.

v2 (29.09): a feature that cannot be computed (no skeleton, a track seen in one frame only, no nose or
shoulder points for a pair) is null instead of an invented value — speed, limb_accel, approach_speed,
facing, crowd_motion; `holds` is "gun" / "knife" (was "pistol" / "knife").
"""

from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

SCHEMA_VERSION = 2

WeaponType = Literal["gun", "knife"]  # v2: pistols and rifles are both "gun"
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
    crowd_motion: NonNegative | None  # mean person speed; null if nobody's speed is known
    weapon_visible: bool


class Person(_Contract):
    id: int
    speed: NonNegative | None  # heights/s; null if the track was seen in a single frame of the window
    limb_accel: NonNegative | None  # heights/s²; null without three confident wrist points
    holds: WeaponType | None
    conf: Probability


class Interaction(_Contract):
    pair: tuple[int, int]
    distance_h: NonNegative
    approach_speed: Rounded | None  # heights/s, positive when approaching; null with one common frame
    facing: bool | None  # null without nose and shoulder points
    duration_s: NonNegative


class SceneDescription(_Contract):
    schema_version: Literal[2] = SCHEMA_VERSION
    window: Window
    zone: Zone
    summary: Summary
    # Tuples, not lists: frozen=True alone would still allow in-place append/sort on shared input.
    persons: tuple[Person, ...] = ()
    interactions: tuple[Interaction, ...] = Field(default=(), max_length=5)


class RiskResult(_Contract):
    """Output of an aggregator; for B it is exactly the strict JSON `{risk, level, reason_uk}`."""

    schema_version: Literal[2] = SCHEMA_VERSION
    risk: Probability
    level: AlertLevel
    reason_uk: str
