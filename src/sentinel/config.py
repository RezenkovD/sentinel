"""Configuration: loads configs/default.yaml into typed settings; secrets come only from .env."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

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


class BenchConfig(_Section):
    imgsz: list[int]
    fps: list[float]
    warmup_frames: int = Field(ge=0)


class Config(_Section):
    source: SourceConfig
    analysis: AnalysisConfig
    models: ModelsConfig
    bench: BenchConfig


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, encoding="utf-8") as f:
        return Config.model_validate(yaml.safe_load(f))
