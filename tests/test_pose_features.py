import numpy as np
import pytest

from sentinel.agents.pose_features import iou_matrix, match_by_iou


def test_iou_of_half_overlapping_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]])
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou_matrix(a, b) == pytest.approx(np.array([[1 / 3, 0.0, 1.0]]))


def test_each_pose_goes_to_the_track_it_covers():
    tracks = np.array([[0.0, 0.0, 10.0, 20.0], [100.0, 0.0, 110.0, 20.0]])
    poses = np.array([[101.0, 1.0, 111.0, 21.0], [1.0, 0.0, 11.0, 20.0]])  # listed in the opposite order
    assert match_by_iou(tracks, poses, min_iou=0.3) == {0: 1, 1: 0}


def test_pose_overlapping_too_little_is_not_attached():
    tracks = np.array([[0.0, 0.0, 10.0, 10.0]])
    poses = np.array([[8.0, 0.0, 18.0, 10.0]])  # IoU = 20 / 180 ≈ 0.11
    assert match_by_iou(tracks, poses, min_iou=0.3) == {}


def test_one_pose_is_attached_to_one_track_only():
    tracks = np.array([[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0]])
    poses = np.array([[2.0, 0.0, 12.0, 10.0]])  # IoU 1.0 with track 1, 0.67 with track 0
    assert match_by_iou(tracks, poses, min_iou=0.3) == {1: 0}


def test_nothing_to_match():
    empty = np.zeros((0, 4))
    assert match_by_iou(empty, np.array([[0.0, 0.0, 1.0, 1.0]]), min_iou=0.3) == {}
    assert match_by_iou(np.array([[0.0, 0.0, 1.0, 1.0]]), empty, min_iou=0.3) == {}
