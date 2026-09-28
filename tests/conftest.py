from pathlib import Path

import cv2
import numpy as np

GRAY_STEP = 8  # frame i is a flat gray image of value i * GRAY_STEP, so the index survives MJPG compression


def make_video(path: Path, n_frames: int, fps: int = 30) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * GRAY_STEP, dtype=np.uint8))
    writer.release()
    return path
