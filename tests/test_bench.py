import time
from pathlib import Path

import numpy as np
import pytest
from conftest import make_video

from sentinel.agents.detector import Track
from sentinel.bench import (
    BenchRow,
    Measurement,
    PacedResult,
    PacedRow,
    Timing,
    append_paced_row,
    append_row,
    check_same_setup,
    clip_info,
    render_table,
    run_measurement,
    run_paced,
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


def test_max_analysis_rate_from_mean_total_time():
    assert _row(total=Timing(50.0, 60.0)).max_fps == pytest.approx(20.0)


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
        "85%",
        "120",
    ]


_ANY = Timing(1.0, 1.0)
_TOTAL = Timing(70.0, 80.0)


def _row(det: Timing = _ANY, pose: Timing = _ANY, total: Timing = _TOTAL) -> BenchRow:
    m = Measurement(det=det, pose=pose, total=total, frames=120, pose_coverage=0.85, rtf={8: 0.61, 10: 0.77})
    return BenchRow(
        model="yolo11n",
        backend="torch",
        device="cpu",
        imgsz=640,
        threads=8,
        measurement=m,
        clips=["fight_a.avi"],
    )


def test_rows_survive_the_jsonl_log_across_runs(tmp_path: Path):
    log = tmp_path / "bench.jsonl"
    first, second = _row(), _row(total=Timing(20.0, 25.0))
    append_row(log, first)
    assert append_row(log, second) == [first, second]


class SleepyAgents:
    """Real-time stand-in for both models: every analysed frame costs `cost_s` of wall time."""

    def __init__(self, cost_s: float) -> None:
        self.cost_s = cost_s

    def update(self, image: np.ndarray) -> list[Track]:
        time.sleep(self.cost_s)
        return [Track(1, (0, 0, 10, 20), 0.9)]

    def reset(self) -> None:
        pass

    def estimate(self, image: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        return {}


def test_paced_run_analyses_at_the_target_rate(tmp_path: Path):
    video = make_video(tmp_path / "clip.avi", n_frames=60)  # 2 s at 30 fps
    agents = SleepyAgents(cost_s=0.005)
    r = run_paced([video], agents, agents, fps=10, warmup=0)
    assert r.received == 60
    assert 17 <= r.delivered <= 21
    assert 8.5 <= r.effective_fps <= 10.5
    assert r.wall_s == pytest.approx(2.0, abs=0.3)


def test_paced_run_with_slow_analysis_drops_frames_but_lag_stays_bounded(tmp_path: Path):
    video = make_video(tmp_path / "clip.avi", n_frames=60)  # 2 s at 30 fps
    agents = SleepyAgents(cost_s=0.25)  # 4 frames/s at best, far below the 10 fps target
    r = run_paced([video], agents, agents, fps=10, warmup=0)
    # A frame queue would end ~13 s behind (60 frames × 0.25 s); the latest-frame reader lags one step.
    assert r.latency.p95_ms < 600
    assert r.delivered <= 10
    assert r.skipped_share >= 0.8


def test_paced_rows_survive_the_jsonl_log(tmp_path: Path):
    result = PacedResult(
        10.0, delivered=187, received=564, wall_s=18.9, duration_s=18.8, latency=Timing(52.0, 71.0)
    )
    row = PacedRow("yolo26n", "torch", "cpu", 480, 8, result, clips=["a.avi"])
    assert append_paced_row(tmp_path / "paced.jsonl", row) == [row]
    assert row.result.effective_fps == pytest.approx(187 / 18.9)
    assert row.result.skipped_share == pytest.approx(1 - 187 / 564)


def test_clip_info_reports_decodable_duration_of_a_truncated_file(tmp_path: Path):
    video = make_video(tmp_path / "clip.avi", n_frames=30)  # header: 30 frames = 1 s
    data = video.read_bytes()
    video.write_bytes(data[: len(data) // 2])  # a partially downloaded file keeps its full-length header
    info = clip_info(video)
    assert info.startswith("clip.avi: 64×48, 30 fps, ")
    assert not info.endswith("1.0 s")


def test_report_refuses_to_mix_runs_on_different_clips_or_warmup(tmp_path: Path):
    log = tmp_path / "bench.jsonl"
    append_row(log, _row())  # measured on fight_a.avi with a 10-frame warm-up
    check_same_setup([log, tmp_path / "no_such.paced.jsonl"], clips=["fight_a.avi"], warmup=10)
    with pytest.raises(ValueError):
        check_same_setup([log], clips=["nonfight_b.avi"], warmup=10)
    with pytest.raises(ValueError):
        check_same_setup([log], clips=["fight_a.avi"], warmup=0)
