"""Configuration: loads configs/default.yaml into typed settings; secrets come only from .env."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sentinel.agents.features import FeatureConfig

DEFAULT_CONFIG = Path("configs/default.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(_Section):
    mode: Literal["live", "paced", "offline"]
    camera_id: str


class AnalysisConfig(_Section):
    fps: float = Field(gt=0)
    window_s: float = Field(gt=0)


class PersonModelConfig(_Section):
    weights: Path
    conf: float = Field(ge=0, le=1)
    tracker: str


class PoseModelConfig(_Section):
    weights: Path
    conf: float = Field(ge=0, le=1)
    min_iou: float = Field(ge=0, le=1)
    min_inside: float = Field(ge=0, le=1)
    keypoint_conf: float = Field(ge=0, le=1)
    box_margin: float = Field(ge=0)


class ModelsConfig(_Section):
    device: str
    imgsz: int = Field(gt=0)
    threads: int | None = Field(default=None, gt=0)
    person: PersonModelConfig
    pose: PoseModelConfig


class FeaturesConfig(_Section):
    min_keypoint_conf: float = Field(ge=0, le=1)
    hold_margin_h: float = Field(ge=0)
    hold_min_frames: int = Field(ge=1)
    pair_max_distance_h: float = Field(gt=0)
    max_pairs: int = Field(ge=0)
    contact_distance_h: float = Field(gt=0)
    facing_min_offset_h: float = Field(ge=0)

    def to_feature_config(self) -> FeatureConfig:
        return FeatureConfig(**self.model_dump())


class ContextConfig(_Section):
    zones: Path


class ZonePriorConfig(_Section):
    hours: dict[str, float]
    density: dict[str, float]
    type: dict[str, float]


class AggregatorAConfig(_Section):
    bias: float
    clip: float = Field(gt=0)
    weights: dict[str, float]
    scale: dict[str, float]
    neutral: dict[str, float]
    zone_prior: ZonePriorConfig


class IncidentsConfig(_Section):
    theta_high: float = Field(gt=0, le=1)
    theta_low: float = Field(ge=0, lt=1)
    k: int = Field(ge=1)
    n: int = Field(ge=1)
    quiet_s: float = Field(ge=0)
    merge_after_close_s: float = Field(ge=0)
    max_new_per_minute: int = Field(ge=1)
    weapon_override_margin: float = Field(ge=0)
    blur_faces: bool
    output_dir: Path


class BenchConfig(_Section):
    imgsz: list[int]
    fps: list[float]
    warmup_frames: int = Field(ge=0)


class Config(_Section):
    source: SourceConfig
    analysis: AnalysisConfig
    models: ModelsConfig
    features: FeaturesConfig
    context: ContextConfig
    aggregator_a: AggregatorAConfig
    incidents: IncidentsConfig
    bench: BenchConfig


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
