"""Detection agent: people tracked by ByteTrack — from a YOLO detector (`detect`) or from the pose model
itself (`pose`: people and skeletons from one model, so one model less per frame; decision 30.09.4).
"""

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


Box = tuple[float, float, float, float]


def pose_tracks(
    xyxy: np.ndarray, ids: np.ndarray | None, conf: np.ndarray, keypoints: np.ndarray
) -> tuple[list[Track], dict[int, np.ndarray], list[tuple[Box, np.ndarray]]]:
    """Tracked pose detections -> tracks, {track_id: keypoints (17, 3)} and every person (box, keypoints)."""
    tracks = tracks_from_arrays(xyxy, ids, conf)
    people = [(tuple(float(v) for v in b), k) for b, k in zip(xyxy, keypoints, strict=True)]
    skeletons = {t.track_id: keypoints[i] for i, t in enumerate(tracks)}
    return tracks, skeletons, people


class PoseTracker:
    """People and skeletons from one YOLO-pose model tracked by ByteTrack (people.source: pose).

    Serves as both the people agent (`update`) and the pose agent (`estimate`) of the pipeline.
    """

    def __init__(
        self, weights: str | Path, imgsz: int, conf: float, tracker: str, device: str = "cpu"
    ) -> None:
        if not Path(weights).exists():  # Ultralytics would silently download official weights instead
            raise FileNotFoundError(f"weights not found: {weights} (see data/SOURCES.md)")
        self.model = YOLO(str(weights), task="pose")
        self.imgsz, self.conf, self.tracker, self.device = imgsz, conf, tracker, device
        self._skeletons: dict[int, np.ndarray] = {}
        self.last_people: list[tuple[Box, np.ndarray]] = []  # every tracked person, for blurring faces

    def update(self, frame: np.ndarray) -> list[Track]:
        result = self.model.track(
            frame,
            persist=True,
            imgsz=self.imgsz,
            conf=self.conf,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        ids = None if boxes.id is None else boxes.id.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        keypoints = (
            np.zeros((len(xyxy), 17, 3)) if result.keypoints is None else result.keypoints.data.cpu().numpy()
        )
        tracks, self._skeletons, self.last_people = pose_tracks(
            xyxy, ids, boxes.conf.cpu().numpy(), keypoints
        )
        return tracks

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        """Skeletons of the given tracks, found by `update` on the same frame (no second model run)."""
        return {t.track_id: self._skeletons[t.track_id] for t in tracks if t.track_id in self._skeletons}

    def reset(self) -> None:
        for tracker in getattr(self.model.predictor, "trackers", []):
            tracker.reset()
        self._skeletons, self.last_people = {}, []
