import os
import subprocess
import sys

import numpy as np
import pytest

from sentinel.agents.detector import PersonTracker, PoseTracker, Track, pose_tracks, tracks_from_arrays
from sentinel.agents.pose_features import MatchRules, PoseEstimator


def test_importing_sentinel_stops_ultralytics_from_installing_packages_at_runtime():
    # Ultralytics pip-installs missing extras (e.g. `lap` for ByteTrack) on the fly, silently changing the
    # locked environment. Checked in a fresh interpreter, because the flag is read once at import time.
    env = {k: v for k, v in os.environ.items() if k != "YOLO_AUTOINSTALL"}
    code = "import sentinel.agents.detector; from ultralytics.utils import AUTOINSTALL; print(AUTOINSTALL)"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_frame_without_confirmed_tracks_gives_no_tracks():
    xyxy = np.array([[10.0, 20.0, 50.0, 120.0]])
    assert tracks_from_arrays(xyxy, ids=None, conf=np.array([0.4])) == []


def test_tracks_keep_id_box_and_confidence():
    xyxy = np.array([[10.0, 20.0, 50.0, 120.0], [200.0, 30.0, 260.0, 180.0]])
    tracks = tracks_from_arrays(xyxy, ids=np.array([3.0, 8.0]), conf=np.array([0.91, 0.55]))
    assert tracks == [
        Track(track_id=3, box=(10.0, 20.0, 50.0, 120.0), conf=0.91),
        Track(track_id=8, box=(200.0, 30.0, 260.0, 180.0), conf=0.55),
    ]
    assert all(type(t.track_id) is int for t in tracks)


def test_missing_weights_fail_fast_instead_of_being_downloaded(tmp_path):
    # Ultralytics would silently fetch a missing official checkpoint, bypassing data/SOURCES.md.
    missing = tmp_path / "yolo11n.pt"
    with pytest.raises(FileNotFoundError):
        PersonTracker(missing, imgsz=640, conf=0.1, tracker="bytetrack.yaml")
    rules = MatchRules(min_iou=0.3, min_inside=0.8, keypoint_conf=0.5, margin=0.1)
    with pytest.raises(FileNotFoundError):
        PoseEstimator(tmp_path / "yolo11n-pose.pt", imgsz=640, conf=0.25, rules=rules)
    assert not missing.exists()


def _kps(x: float) -> np.ndarray:
    k = np.zeros((17, 3))
    k[:, 0], k[:, 2] = x, 0.9
    return k


def test_pose_source_gives_tracks_and_their_skeletons_from_one_model():
    xyxy = np.array([[10.0, 20.0, 50.0, 120.0], [200.0, 30.0, 260.0, 180.0]])
    kps = np.stack([_kps(30.0), _kps(230.0)])
    tracks, skeletons, people = pose_tracks(xyxy, np.array([4.0, 9.0]), np.array([0.8, 0.6]), kps)
    assert [t.track_id for t in tracks] == [4, 9]
    assert skeletons[4][0, 0] == 30.0 and skeletons[9][0, 0] == 230.0
    assert len(people) == 2 and people[1][0] == (200.0, 30.0, 260.0, 180.0)


def test_pose_source_without_confirmed_tracks_still_reports_people_for_blurring():
    xyxy = np.array([[10.0, 20.0, 50.0, 120.0]])
    tracks, skeletons, people = pose_tracks(xyxy, None, np.array([0.3]), np.stack([_kps(30.0)]))
    assert tracks == [] and skeletons == {} and len(people) == 1


def test_pose_tracker_needs_its_weights(tmp_path):
    with pytest.raises(FileNotFoundError):
        PoseTracker(tmp_path / "yolo26n-pose.pt", imgsz=480, conf=0.1, tracker="bytetrack.yaml")
