"""Detection agent: people (YOLO nano, COCO) tracked by ByteTrack; weapon detection is added later."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PERSON_CLASS = 0  # COCO


@dataclass(frozen=True)
class Track:
    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in frame pixels
    conf: float


def tracks_from_arrays(xyxy: np.ndarray, ids: np.ndarray | None, conf: np.ndarray) -> list[Track]:
    """Tracker output as Track objects; `ids` is None while the tracker has no confirmed tracks."""
    if ids is None:
        return []
    return [
        Track(track_id=int(i), box=tuple(float(v) for v in box), conf=float(c))
        for box, i, c in zip(xyxy, ids, conf, strict=True)
    ]


class PersonTracker:
    """YOLO person detector + ByteTrack with state kept between frames of one stream."""

    def __init__(
        self, weights: str | Path, imgsz: int, conf: float, tracker: str, device: str = "cpu"
    ) -> None:
        if not Path(weights).exists():  # Ultralytics would silently download official weights instead
            raise FileNotFoundError(f"weights not found: {weights} (see data/SOURCES.md)")
        self.model = YOLO(str(weights), task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = tracker
        self.device = device

    def update(self, frame: np.ndarray) -> list[Track]:
        result = self.model.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            imgsz=self.imgsz,
            conf=self.conf,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        ids = None if boxes.id is None else boxes.id.cpu().numpy()
        return tracks_from_arrays(boxes.xyxy.cpu().numpy(), ids, boxes.conf.cpu().numpy())

    def reset(self) -> None:
        """Forget all tracks and restart IDs (call between unrelated clips)."""
        for tracker in getattr(self.model.predictor, "trackers", []):
            tracker.reset()
