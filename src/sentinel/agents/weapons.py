"""Weapon detection agent: `gun` / `knife` boxes from a YOLO model trained on weapons.

Class names of the chosen checkpoint (e.g. Pistol / Rifle / gun / Knife) are mapped to our two kinds through
a configurable table; other classes are ignored. The model may run only on every n-th frame: in between, the
last boxes are kept until the next run.

Two modes: `frame` runs the model on the whole frame; `crops` runs it on a padded square around every tracked
person, where a small weapon is 2–3 times larger for the model (weapons in CCTV frames are ~60 px of 1080).
"""

from collections.abc import Callable, Sequence
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


def person_crop(
    box: tuple[float, float, float, float], frame_shape: tuple[int, ...], pad: float
) -> tuple[int, int, int, int]:
    """Square around a person box, side = longer box side x (1 + pad), shifted (not cut) to stay inside."""
    h_img, w_img = frame_shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(int(round(max(box[2] - box[0], box[3] - box[1]) * (1 + pad))), w_img, h_img)
    x1 = min(max(int(round(cx - side / 2)), 0), w_img - side)
    y1 = min(max(int(round(cy - side / 2)), 0), h_img - side)
    return x1, y1, x1 + side, y1 + side


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: list["WeaponBox"], iou_threshold: float = 0.5) -> list["WeaponBox"]:
    """Greedy NMS per kind: a weapon seen from two overlapping crops is kept once (highest score)."""
    kept: list[WeaponBox] = []
    for b in sorted(boxes, key=lambda b: -b.conf):
        if all(k.kind != b.kind or _iou(k.box, b.box) < iou_threshold for k in kept):
            kept.append(b)
    return kept


def crops_run(
    predict: Callable[[np.ndarray], list["WeaponBox"]], pad: float
) -> Callable[[np.ndarray, Sequence[tuple[float, float, float, float]]], list["WeaponBox"]]:
    """A `run(frame, people)` that calls `predict` on a padded square around each person."""

    def run(frame: np.ndarray, people: Sequence[tuple[float, float, float, float]]) -> list[WeaponBox]:
        found = []
        for person in people:
            x1, y1, x2, y2 = person_crop(person, frame.shape, pad)
            for b in predict(frame[y1:y2, x1:x2]):
                box = (b.box[0] + x1, b.box[1] + y1, b.box[2] + x1, b.box[3] + y1)
                found.append(WeaponBox(b.kind, box, b.conf))
        return merge_boxes(found)

    return run


def map_class(name: str, class_map: dict[str, str]) -> WeaponKind | None:
    """Our weapon kind for a checkpoint class name (case-insensitive), or None for classes we ignore."""
    kind = {k.lower(): v for k, v in class_map.items()}.get(name.lower())
    if kind not in (None, "gun", "knife"):
        raise ValueError(f"class map points {name!r} to unknown kind {kind!r}")
    return kind


Run = Callable[[np.ndarray, Sequence[tuple[float, float, float, float]]], list[WeaponBox]]


class WeaponDetector:
    """Runs `run(frame, people)` every `every_n_frames` frames and returns the last boxes in between.

    Boxes scoring below `conf` are dropped; `last_max_conf` keeps the best raw score of the last run (the
    model may run at a lower score to record it, e.g. for the baseline detector on cached episodes).
    """

    def __init__(self, run: Run, every_n_frames: int = 1, conf: float = 0.0) -> None:
        if every_n_frames < 1:
            raise ValueError(f"every_n_frames must be at least 1, got {every_n_frames}")
        self._run = run
        self.every_n_frames = every_n_frames
        self.conf = conf
        self.reset()

    def reset(self) -> None:
        self._frames_since_run: int | None = None
        self._boxes: list[WeaponBox] = []
        self.ran = False  # the model ran on the last frame given to `detect` (else its boxes were kept)
        self.last_max_conf = 0.0  # best raw score of the last run, before the `conf` threshold

    def detect(
        self, frame: np.ndarray, people: Sequence[tuple[float, float, float, float]] = ()
    ) -> list[WeaponBox]:
        """Weapon boxes; `people` (person boxes) are needed in the crops mode only."""
        self.ran = self._frames_since_run is None or self._frames_since_run >= self.every_n_frames - 1
        if self.ran:
            raw = self._run(frame, people)
            self.last_max_conf = max((b.conf for b in raw), default=0.0)
            self._boxes = [b for b in raw if b.conf >= self.conf]
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
        mode: str = "frame",
        crop_pad: float = 0.25,
        device: str = "cpu",
        record_conf: float | None = None,
    ) -> "WeaponDetector":
        """`record_conf` (lower than `conf`) makes the model score boxes down to it for `last_max_conf`."""
        if not Path(weights).exists():  # Ultralytics would silently download official weights instead
            raise FileNotFoundError(f"weights not found: {weights} (see data/SOURCES.md)")
        from ultralytics import YOLO

        model = YOLO(str(weights), task="detect")
        predict_conf = conf if record_conf is None else min(conf, record_conf)

        def predict(image: np.ndarray) -> list[WeaponBox]:
            result = model.predict(image, imgsz=imgsz, conf=predict_conf, device=device, verbose=False)[0]
            boxes = result.boxes
            found = []
            for xyxy, score, cls_id in zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy(), strict=True
            ):
                kind = map_class(result.names[int(cls_id)], class_map)
                if kind is not None:
                    found.append(WeaponBox(kind, tuple(float(v) for v in xyxy), float(score)))
            return found

        if mode == "frame":
            return cls(lambda frame, people: predict(frame), every_n_frames, conf)
        if mode == "crops":
            return cls(crops_run(predict, crop_pad), every_n_frames, conf)
        raise ValueError(f"unknown weapon detector mode {mode!r} (frame | crops)")
