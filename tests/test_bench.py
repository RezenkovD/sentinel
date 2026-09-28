from pathlib import Path

import numpy as np
import pytest
from conftest import make_video

from sentinel.agents.detector import Track
from sentinel.bench import (
    BenchRow,
    Measurement,
    Timing,
    append_row,
    render_table,
    run_measurement,
    timing,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeTracker:
    """Costs 3 s per call while cold, then 30 ms; always sees two people."""

    def __init__(self, clock: FakeClock, cold_calls: int) -> None:
        self.clock, self.cold_calls = clock, cold_calls
        self.calls = 0
        self.resets = 0

    def update(self, image: np.ndarray) -> list[Track]:
        self.calls += 1
        self.clock.now += 3.0 if self.calls <= self.cold_calls else 0.030
        return [Track(1, (0, 0, 10, 20), 0.9), Track(2, (20, 0, 30, 20), 0.8)]

    def reset(self) -> None:
        self.resets += 1


class FakePoser:
    """Costs 40 ms; finds a skeleton for the first track only."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock

    def estimate(self, image: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        self.clock.now += 0.040
        return {tracks[0].track_id: np.zeros((17, 3))}


def test_timing_mean_and_p95():
    assert timing([10.0, 20.0, 30.0, 40.0]) == Timing(mean_ms=25.0, p95_ms=38.5)


def test_rates_derived_from_mean_total_time():
    row = _row(total=Timing(50.0, 60.0))
    assert row.max_fps == pytest.approx(20.0)
    assert row.model_rtf(8) == pytest.approx(0.4)
    assert row.model_rtf(10) == pytest.approx(0.5)


def test_measurement_excludes_warmup_and_times_each_stage(tmp_path: Path):
    video = make_video(tmp_path / "clip.avi", n_frames=30)  # 1 s at 30 fps
    clock = FakeClock()
    tracker = FakeTracker(clock, cold_calls=5)
    m = run_measurement([video], tracker, FakePoser(clock), fps_list=[10, 8], warmup=5, clock=clock)

    assert tracker.calls == 5 + 10 + 8  # warm-up frames are processed, but not measured
    assert m.frames == 18
    assert (m.det.mean_ms, m.det.p95_ms) == pytest.approx((30.0, 30.0))
    assert (m.pose.mean_ms, m.pose.p95_ms) == pytest.approx((40.0, 40.0))
    assert (m.total.mean_ms, m.total.p95_ms) == pytest.approx((70.0, 70.0))
    assert m.pose_coverage == pytest.approx(0.5)
    # 1 s of video at 10 fps costs 10 × 70 ms, at 8 fps 8 × 70 ms
    assert m.rtf == pytest.approx({10: 0.7, 8: 0.56})
    assert tracker.resets == 2  # a fresh tracker state for every measured clip


def test_table_row_puts_each_value_in_its_column():
    row = _row(det=Timing(30.0, 35.0), pose=Timing(40.0, 45.0), total=Timing(70.0, 80.04))
    line = render_table([row]).splitlines()[-1]
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells == [
        "yolo11n",
        "torch",
        "cpu",
        "640",
        "8",
        "30.0 / 35.0",
        "40.0 / 45.0",
        "70.0 / 80.0",
        "14.3",
        "0.61",
        "0.77",
        "0.56 / 0.70",
        "85%",
        "120",
    ]


_ANY = Timing(1.0, 1.0)
_TOTAL = Timing(70.0, 80.0)


def _row(det: Timing = _ANY, pose: Timing = _ANY, total: Timing = _TOTAL) -> BenchRow:
    m = Measurement(det=det, pose=pose, total=total, frames=120, pose_coverage=0.85, rtf={8: 0.61, 10: 0.77})
    return BenchRow(model="yolo11n", backend="torch", device="cpu", imgsz=640, threads=8, measurement=m)


def test_rows_survive_the_jsonl_log_across_runs(tmp_path: Path):
    log = tmp_path / "bench.jsonl"
    first, second = _row(), _row(total=Timing(20.0, 25.0))
    append_row(log, first)
    assert append_row(log, second) == [first, second]
