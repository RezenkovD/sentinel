"""Pose and interaction agent: YOLO-pose skeletons attached to person tracks; person and pair features.

The pose model runs on the full frame; each detected skeleton is attached to the person track whose box it
overlaps most (greedy, one-to-one, IoU ≥ min_iou). Person and pair features are added in the next stage.
"""

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


def match_by_iou(track_boxes: np.ndarray, pose_boxes: np.ndarray, min_iou: float) -> dict[int, int]:
    """Greedy one-to-one matching, highest IoU first: {track index: pose index}."""
    if len(track_boxes) == 0 or len(pose_boxes) == 0:
        return {}
    iou = iou_matrix(track_boxes, pose_boxes)
    matches: dict[int, int] = {}
    used_poses: set[int] = set()
    for flat in np.argsort(-iou, axis=None):
        t, p = np.unravel_index(flat, iou.shape)
        if iou[t, p] < min_iou:
            break
        if t not in matches and p not in used_poses:
            matches[int(t)] = int(p)
            used_poses.add(int(p))
    return matches


class PoseEstimator:
    """YOLO-pose (17 COCO keypoints) on the full frame, skeletons attached to person tracks."""

    def __init__(
        self, weights: str | Path, imgsz: int, conf: float, min_iou: float, device: str = "cpu"
    ) -> None:
        self.model = YOLO(str(weights), task="pose")
        self.imgsz = imgsz
        self.conf = conf
        self.min_iou = min_iou
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
        matches = match_by_iou(track_boxes, pose_boxes, self.min_iou)
        return {tracks[t].track_id: keypoints[p] for t, p in matches.items()}
