"""Pose and interaction agent: YOLO-pose skeletons attached to person tracks; person and pair features.

The pose model runs on the full frame; each detected skeleton is attached to at most one person track and
vice versa (greedy, highest box IoU first). A pair qualifies only if the boxes overlap enough AND most of the
skeleton's confident keypoints lie inside the track box (slightly expanded, because the tracker's box is a
smoothed estimate): box IoU alone would hand a neighbour's skeleton to a track whose own skeleton was missed.
Person and pair features are added in the next stage.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from sentinel.agents.detector import Track


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of boxes a (N, 4) and b (M, 4) in x1, y1, x2, y2 format."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def keypoints_inside_share(
    boxes: np.ndarray, keypoints: np.ndarray, keypoint_conf: float, margin: float
) -> np.ndarray:
    """(N boxes, M skeletons): share of each skeleton's confident keypoints inside each box.

    Boxes are expanded by `margin` × their width/height on every side; a skeleton without confident
    keypoints gets 0 for every box.
    """
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    x1, x2 = boxes[:, 0] - margin * w, boxes[:, 2] + margin * w
    y1, y2 = boxes[:, 1] - margin * h, boxes[:, 3] + margin * h
    x, y = keypoints[None, :, :, 0], keypoints[None, :, :, 1]  # (1, M, 17)
    inside = (
        (x >= x1[:, None, None])
        & (x <= x2[:, None, None])
        & (y >= y1[:, None, None])
        & (y <= y2[:, None, None])
    )
    confident = keypoints[None, :, :, 2] >= keypoint_conf
    n_confident = confident.sum(axis=2)
    return np.divide(
        (inside & confident).sum(axis=2), n_confident, out=np.zeros(inside.shape[:2]), where=n_confident > 0
    )


@dataclass(frozen=True)
class MatchRules:
    min_iou: float  # minimal box IoU between the track and the skeleton's box
    min_inside: float  # minimal share of confident keypoints inside the (expanded) track box
    keypoint_conf: float  # a keypoint counts as confident from this score
    margin: float  # track box expansion, share of its width/height on every side


def match_poses(
    track_boxes: np.ndarray,
    pose_boxes: np.ndarray,
    keypoints: np.ndarray,
    min_iou: float,
    min_inside: float,
    keypoint_conf: float,
    margin: float,
) -> dict[int, int]:
    """Greedy one-to-one matching, highest IoU first, among qualifying pairs: {track index: pose index}."""
    if len(track_boxes) == 0 or len(pose_boxes) == 0:
        return {}
    iou = iou_matrix(track_boxes, pose_boxes)
    inside = keypoints_inside_share(track_boxes, keypoints, keypoint_conf, margin)
    score = np.where((iou >= min_iou) & (inside >= min_inside), iou, -1.0)
    matches: dict[int, int] = {}
    used_poses: set[int] = set()
    for flat in np.argsort(-score, axis=None):
        t, p = np.unravel_index(flat, score.shape)
        if score[t, p] < 0:
            break
        if t not in matches and p not in used_poses:
            matches[int(t)] = int(p)
            used_poses.add(int(p))
    return matches


class PoseEstimator:
    """YOLO-pose (17 COCO keypoints) on the full frame, skeletons attached to person tracks."""

    def __init__(
        self, weights: str | Path, imgsz: int, conf: float, rules: MatchRules, device: str = "cpu"
    ) -> None:
        if not Path(weights).exists():  # Ultralytics would silently download official weights instead
            raise FileNotFoundError(f"weights not found: {weights} (see data/SOURCES.md)")
        self.model = YOLO(str(weights), task="pose")
        self.imgsz = imgsz
        self.conf = conf
        self.rules = rules
        self.device = device

    def estimate(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        """{track_id: keypoints (17, 3) as x, y, confidence in frame pixels} for tracks with a skeleton."""
        result = self.model.predict(
            frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        if result.keypoints is None or not tracks:
            return {}
        pose_boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        track_boxes = np.array([t.box for t in tracks])
        r = self.rules
        matches = match_poses(
            track_boxes, pose_boxes, keypoints, r.min_iou, r.min_inside, r.keypoint_conf, r.margin
        )
        return {tracks[t].track_id: keypoints[p] for t, p in matches.items()}
