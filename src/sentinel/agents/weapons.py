"""Weapon detection agent: `gun` / `knife` boxes from a YOLO model trained on weapons.

Class names of the chosen checkpoint (e.g. Pistol / Rifle / gun / Knife) are mapped to our two kinds through
a configurable table; other classes are ignored. The model may run only on every n-th frame: in between, the
last boxes are kept until the next run.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

WeaponKind = Literal["gun", "knife"]


@dataclass(frozen=True)
class WeaponBox:
    kind: WeaponKind
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    conf: float


def map_class(name: str, class_map: dict[str, str]) -> WeaponKind | None:
    """Our weapon kind for a checkpoint class name (case-insensitive), or None for classes we ignore."""
    kind = {k.lower(): v for k, v in class_map.items()}.get(name.lower())
    if kind not in (None, "gun", "knife"):
        raise ValueError(f"class map points {name!r} to unknown kind {kind!r}")
    return kind


class WeaponDetector:
    """Runs `run(frame)` every `every_n_frames` frames and returns the last boxes in between."""

    def __init__(self, run: Callable[[np.ndarray], list[WeaponBox]], every_n_frames: int = 1) -> None:
        if every_n_frames < 1:
            raise ValueError(f"every_n_frames must be at least 1, got {every_n_frames}")
        self._run = run
        self.every_n_frames = every_n_frames
        self.reset()

    def reset(self) -> None:
        self._frames_since_run: int | None = None
        self._boxes: list[WeaponBox] = []

    def detect(self, frame: np.ndarray) -> list[WeaponBox]:
        if self._frames_since_run is None or self._frames_since_run >= self.every_n_frames - 1:
            self._boxes = self._run(frame)
            self._frames_since_run = 0
        else:
            self._frames_since_run += 1
        return self._boxes

    @classmethod
    def from_weights(
        cls,
        weights: str | Path,
        imgsz: int,
        conf: float,
        every_n_frames: int,
        class_map: dict[str, str],
        device: str = "cpu",
    ) -> "WeaponDetector":
        if not Path(weights).exists():  # Ultralytics would silently download official weights instead
            raise FileNotFoundError(f"weights not found: {weights} (see data/SOURCES.md)")
        from ultralytics import YOLO

        model = YOLO(str(weights), task="detect")

        def run(frame: np.ndarray) -> list[WeaponBox]:
            result = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            boxes = result.boxes
            found = []
            for xyxy, score, cls_id in zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), strict=True
            ):
                kind = map_class(result.names[int(cls_id)], class_map)
                if kind is not None:
                    found.append(WeaponBox(kind, tuple(float(v) for v in xyxy), float(score)))
            return found

        return cls(run, every_n_frames)
