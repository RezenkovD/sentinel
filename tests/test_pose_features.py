import numpy as np
import pytest

from sentinel.agents.pose_features import iou_matrix, match_poses

RULES = {"min_iou": 0.3, "min_inside": 0.8, "keypoint_conf": 0.5, "margin": 0.1}


def _skeleton(x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> np.ndarray:
    """17 keypoints spread over the box with a 10 % inset, as (x, y, confidence)."""
    w, h = x2 - x1, y2 - y1
    xs = np.linspace(x1 + 0.1 * w, x2 - 0.1 * w, 17)
    ys = np.linspace(y1 + 0.1 * h, y2 - 0.1 * h, 17)
    return np.stack([xs, ys, np.full(17, conf)], axis=1)


def _poses(*boxes: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    return np.array(boxes, dtype=float), np.stack([_skeleton(*b) for b in boxes])


def test_iou_of_half_overlapping_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]])
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou_matrix(a, b) == pytest.approx(np.array([[1 / 3, 0.0, 1.0]]))


def test_each_pose_goes_to_the_track_it_covers():
    tracks = np.array([[0.0, 0.0, 10.0, 20.0], [100.0, 0.0, 110.0, 20.0]])
    boxes, keypoints = _poses((101.0, 1.0, 111.0, 21.0), (1.0, 0.0, 11.0, 20.0))  # opposite order
    assert match_poses(tracks, boxes, keypoints, **RULES) == {0: 1, 1: 0}


def test_pose_overlapping_too_little_is_not_attached():
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    boxes, keypoints = _poses((8.0, 0.0, 18.0, 10.0))  # IoU = 20 / 180 ≈ 0.11
    assert match_poses(tracks, boxes, keypoints, **RULES) == {}


def test_one_pose_is_attached_to_one_track_only():
    tracks = np.array([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]])
    boxes, keypoints = _poses((2.0, 0.0, 12.0, 10.0))  # IoU 1.0 with track 1, 0.67 with track 0
    assert match_poses(tracks, boxes, keypoints, **RULES) == {1: 0}


def test_neighbours_skeleton_is_not_given_to_a_track_that_lost_its_own():
    # Person A is tracked but the pose model missed them; person B, half-overlapping, is not tracked.
    # Box IoU is 1/3 (passes 0.3), but most of B's keypoints lie outside A's box.
    track_a = np.array([[0.0, 0.0, 100.0, 200.0]])
    boxes, keypoints = _poses((50.0, 0.0, 150.0, 200.0))
    assert match_poses(track_a, boxes, keypoints, **RULES) == {}


def test_own_skeleton_is_kept_when_the_track_box_lags_behind():
    track = np.array([[0.0, 0.0, 100.0, 200.0]])  # tracker estimate trails the person by 20 px
    boxes, keypoints = _poses((20.0, 0.0, 120.0, 200.0))
    assert match_poses(track, boxes, keypoints, **RULES) == {0: 0}


def test_skeleton_without_confident_keypoints_is_not_attached():
    track = np.array([[0.0, 0.0, 100.0, 200.0]])
    boxes = np.array([[0.0, 0.0, 100.0, 200.0]])
    keypoints = _skeleton(0.0, 0.0, 100.0, 200.0, conf=0.2)[None]
    assert match_poses(track, boxes, keypoints, **RULES) == {}


def test_nothing_to_match():
    empty_boxes, empty_keypoints = np.zeros((0, 4)), np.zeros((0, 17, 3))
    boxes, keypoints = _poses((0.0, 0.0, 1.0, 1.0))
    assert match_poses(np.zeros((0, 4)), boxes, keypoints, **RULES) == {}
    assert match_poses(np.array([[0.0, 0.0, 1.0, 1.0]]), empty_boxes, empty_keypoints, **RULES) == {}
