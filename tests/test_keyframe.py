from pathlib import Path

import cv2
import numpy as np

from sentinel.agents.detector import Track
from sentinel.agents.weapons import WeaponBox
from sentinel.keyframe import KeyframeWriter, Snapshot, render_keyframe


def _checkerboard(h: int = 400, w: int = 600) -> np.ndarray:
    board = (np.indices((h, w)).sum(axis=0) % 2 * 255).astype(np.uint8)  # 1-px squares: maximal detail
    return np.dstack([board] * 3)


PERSON = Track(track_id=3, box=(200.0, 100.0, 300.0, 380.0), conf=0.9)


def _face_keypoints() -> np.ndarray:
    k = np.zeros((17, 3))
    k[0] = (250.0, 140.0, 0.9)  # nose
    k[1], k[2] = (240.0, 130.0, 0.9), (260.0, 130.0, 0.9)  # eyes
    return k


def _detail(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
    return float(image[y1:y2, x1:x2].std())


def test_face_is_blurred_around_the_face_keypoints():
    image = _checkerboard()
    out = render_keyframe(image, [PERSON], {3: _face_keypoints()}, [], blur_faces=True)
    assert _detail(out, 240, 130, 260, 150) < 0.2 * _detail(image, 240, 130, 260, 150)
    assert _detail(out, 220, 300, 280, 360) > 0.8 * _detail(image, 220, 300, 280, 360)  # body stays sharp


def test_without_face_keypoints_the_top_of_the_person_box_is_blurred():
    image = _checkerboard()
    out = render_keyframe(image, [PERSON], {}, [], blur_faces=True)
    assert _detail(out, 220, 110, 280, 140) < 0.2 * _detail(image, 220, 110, 280, 140)


def test_faces_are_kept_when_blurring_is_off():
    image = _checkerboard()
    out = render_keyframe(image, [PERSON], {3: _face_keypoints()}, [], blur_faces=False)
    assert np.array_equal(out[130:150, 240:260], image[130:150, 240:260])


def test_weapon_and_person_boxes_are_drawn_in_their_colours():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    gun = WeaponBox("gun", (400.0, 200.0, 440.0, 230.0), 0.8)
    out = render_keyframe(image, [PERSON], {}, [gun], blur_faces=False)
    assert tuple(out[215, 400]) == (0, 0, 255)  # weapon: red (BGR)
    assert tuple(out[250, 200]) == (0, 200, 0)  # person: green


def test_writer_saves_the_keyframe_under_the_incident(tmp_path: Path):
    writer = KeyframeWriter(tmp_path, blur_faces=True)
    snapshot = Snapshot(image=_checkerboard(), tracks=[PERSON], skeletons={}, weapons=[])
    path = writer("cam-1-001", "keyframe", snapshot)
    assert path == tmp_path / "cam-1-001" / "keyframe.jpg"
    assert cv2.imread(str(path)).shape == (400, 600, 3)
    assert writer("cam-1-001", "escalate_1", None) is None  # no frame to show
