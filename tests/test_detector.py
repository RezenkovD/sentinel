import os
import subprocess
import sys

import numpy as np

from sentinel.agents.detector import Track, tracks_from_arrays


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
