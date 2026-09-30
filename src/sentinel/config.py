"""Configuration: loads configs/default.yaml into typed settings; secrets come only from .env."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class PeopleModelConfig(_Section):
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
    backend: Literal["torch", "openvino"]
    device: Literal["cpu", "gpu"]  # gpu: the integrated Intel GPU, OpenVINO only
    imgsz: int = Field(gt=0)
    threads: int | None = Field(default=None, gt=0)
    people: PeopleModelConfig
    pose: PoseModelConfig

    @model_validator(mode="after")
    def _device(self) -> "ModelsConfig":
        if self.backend == "torch" and self.device != "cpu":
            raise ValueError("the torch backend runs on the CPU only (no CUDA on this machine)")
        return self

    @property
    def runtime_device(self) -> str:
        """Device string for Ultralytics: cpu (PyTorch) or intel:cpu / intel:gpu (OpenVINO)."""
        return "cpu" if self.backend == "torch" else f"intel:{self.device}"


class WeaponsConfig(_Section):
    weights: Path
    mode: Literal["frame", "crops"]
    imgsz: int = Field(gt=0)
    conf: float = Field(ge=0, le=1)
    every_n_frames: int = Field(ge=1, le=2)
    crop_pad: float = Field(ge=0)
    class_map: dict[str, Literal["gun", "knife"]]


class NotifyConfig(_Section):
    incidents_jsonl: Path
    edit_interval_s: float = Field(ge=0)


class FeaturesConfig(_Section):
    min_keypoint_conf: float = Field(ge=0, le=1)
    hold_margin_h: float = Field(ge=0)
    hold_min_detections: int = Field(ge=1)
    pair_max_distance_h: float = Field(gt=0)
    max_pairs: int = Field(ge=0, le=5)  # SPEC §5: at most 5 pairs per scene description
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
    reason_min_contribution: float = Field(ge=0)
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
    merge_participants_max_s: float | None = Field(default=None, ge=0)  # None = no time limit (SPEC §7)
    max_new_per_minute: int = Field(ge=1)
    update_interval_s: float = Field(ge=0)
    weapon_override_margin: float = Field(ge=0)
    blur_faces: bool
    output_dir: Path

    @model_validator(mode="after")
    def _hysteresis(self) -> "IncidentsConfig":
        if not self.theta_low < self.theta_high:
            raise ValueError(f"theta_low ({self.theta_low}) must be below theta_high ({self.theta_high})")
        if self.k > self.n:
            raise ValueError(f"k ({self.k}) cannot exceed n ({self.n})")
        return self


class BenchConfig(_Section):
    imgsz: list[int]
    fps: list[float]
    warmup_frames: int = Field(ge=0)


class Config(_Section):
    source: SourceConfig
    analysis: AnalysisConfig
    models: ModelsConfig
    weapons: WeaponsConfig
    notify: NotifyConfig
    features: FeaturesConfig
    context: ContextConfig
    aggregator_a: AggregatorAConfig
    incidents: IncidentsConfig
    bench: BenchConfig


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
