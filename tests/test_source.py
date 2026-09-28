import time
from pathlib import Path

import numpy as np
import pytest
from conftest import GRAY_STEP, make_video

from sentinel.io.source import LatestFrameReader

SRC_FPS = 30


def _make_video(path: Path, n_frames: int) -> Path:
    return make_video(path, n_frames, fps=SRC_FPS)


def _index_in_pixels(image: np.ndarray) -> int:
    return round(float(image.mean()) / GRAY_STEP)


@pytest.fixture
def video_30(tmp_path: Path) -> Path:
    return _make_video(tmp_path / "clip.avi", n_frames=30)


def test_offline_without_target_fps_delivers_every_frame(video_30: Path):
    with LatestFrameReader(video_30, mode="offline") as reader:
        frames = list(reader)
        stats = reader.stats
    assert [f.index for f in frames] == list(range(30))
    assert [_index_in_pixels(f.image) for f in frames] == list(range(30))
    assert (stats.received, stats.delivered, stats.skipped) == (30, 30, 0)


def test_offline_samples_frames_on_the_analysis_grid(video_30: Path):
    with LatestFrameReader(video_30, mode="offline", target_fps=10) as reader:
        frames = list(reader)
        stats = reader.stats
    assert [f.index for f in frames] == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert [_index_in_pixels(f.image) for f in frames] == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert [f.t for f in frames] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert (stats.received, stats.delivered, stats.skipped) == (30, 10, 20)


def test_offline_picks_first_frame_at_or_after_each_grid_time(video_30: Path):
    # Grid at 8 fps: 0, 0.125, 0.25, ... s; at 30 fps the first frames at or after those times are these.
    with LatestFrameReader(video_30, mode="offline", target_fps=8) as reader:
        assert [f.index for f in reader] == [0, 4, 8, 12, 15, 19, 23, 27]


def test_paced_file_is_not_read_faster_than_real_time(tmp_path: Path):
    video = _make_video(tmp_path / "half_second.avi", n_frames=15)
    started = time.monotonic()
    with LatestFrameReader(video, mode="paced") as reader:
        frames = list(reader)
        stats = reader.stats
    assert time.monotonic() - started >= 0.4
    assert stats.received == 15
    assert stats.delivered + stats.skipped == stats.received
    assert [f.index for f in frames] == sorted({f.index for f in frames})


def test_paced_slow_consumer_gets_freshest_frame_not_a_backlog(video_30: Path):
    with LatestFrameReader(video_30, mode="paced") as reader:
        frames = []
        while (frame := reader.read()) is not None:
            frames.append(frame)
            time.sleep(0.2)  # analysis much slower than the 30 fps source
        stats = reader.stats
    # A queue would hand out 0, 1, 2, ... one by one; the latest-frame reader jumps over stale frames.
    assert len(frames) <= 8
    assert stats.skipped >= 20
    assert stats.delivered + stats.skipped == stats.received == 30
    assert [_index_in_pixels(f.image) for f in frames] == [f.index for f in frames]


def test_paced_target_fps_limits_the_analysis_rate(video_30: Path):
    with LatestFrameReader(video_30, mode="paced", target_fps=5) as reader:
        frames = list(reader)
    assert 4 <= len(frames) <= 7  # 1 s of video analysed at 5 frames/s


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        LatestFrameReader(tmp_path / "nope.avi", mode="offline")
