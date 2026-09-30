import time
from pathlib import Path

import pytest

from sentinel.quiet import LoadWindow, QuietRunner, own_busy_threads, system_busy


def test_system_busy_counts_running_and_blocked_tasks(tmp_path: Path):
    stat = tmp_path / "stat"
    stat.write_text("cpu  1 2 3\nctxt 5\nprocs_running 7\nprocs_blocked 2\n")
    assert system_busy(stat) == 9


def _task(root: Path, pid: int, tid: int, state: str) -> None:
    d = root / str(pid) / "task" / str(tid)
    d.mkdir(parents=True)
    (d / "stat").write_text(f"{tid} (python3 (x) y) {state} 1 2 3")  # a name with spaces and parentheses


def test_own_busy_threads_are_counted_per_process(tmp_path: Path):
    _task(tmp_path, 10, 10, "R")
    _task(tmp_path, 10, 11, "S")
    _task(tmp_path, 10, 12, "D")
    _task(tmp_path, 20, 20, "R")
    assert own_busy_threads({10}, tmp_path) == 2
    assert own_busy_threads({10, 20, 99}, tmp_path) == 3  # a process that is gone counts nothing


def test_load_window_averages_the_last_minute():
    w = LoadWindow(span_s=60.0)
    for t in range(0, 120):
        w.add(float(t), 3.0 if t < 60 else 1.0)
    assert w.mean(119.0) == 1.0


def test_quiet_time_counts_from_when_the_average_dropped_below_the_limit():
    w = LoadWindow(span_s=10.0)
    for t in range(0, 30):
        w.add(float(t), 5.0 if t < 10 else 0.0)
    # the 10-s mean is below 1.5 from t = 17 (samples 8..17 hold two 5.0s -> 1.0)
    assert w.quiet_for(29.0, limit=1.5) == 12.0


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def _runner(loads, measured, events, clock, ac=True, profile="performance"):
    def sample() -> float:
        return loads(clock.t)

    def measure(combo, during) -> dict:
        # the measurement lasts 100 s; `during` is called while it runs, like the background sampler
        start = clock.t
        while clock.t < start + 100:
            clock.sleep(5)
            during()
        measured.append((combo, start))
        return {"combo": combo}

    return QuietRunner(
        sample=sample,
        measure=measure,
        power=lambda: (ac, profile),
        pause=lambda: events.append(("pause", clock.t)),
        resume=lambda: events.append(("resume", clock.t)),
        clock=clock,
        sleep=clock.sleep,
        start_limit=1.5,
        hold_s=120.0,
        noisy_limit=2.0,
        span_s=60.0,
        step_s=5.0,
    )


def test_held_pause_keeps_our_background_work_stopped_from_the_first_measurement_to_the_end():
    """Decision 30.09 (report 03): the quiet bench runs with the extraction paused for the whole bench."""
    clock, measured, events = FakeClock(), [], []
    runner = _runner(lambda t: 0.5, measured, events, clock)
    runner.hold_pause = True
    rows = runner.run(["a", "b", "c"], deadline_s=10_000)
    assert len(rows) == 3 and [e[0] for e in events] == ["pause", "resume"]
    assert events[0][1] == measured[0][1] and events[1][1] >= measured[-1][1] + 100


def test_held_pause_is_released_when_the_bench_fails():
    clock, events = FakeClock(), []

    def broken(combo, during) -> dict:
        raise RuntimeError("model file missing")

    runner = _runner(lambda t: 0.5, [], events, clock)
    runner.measure, runner.hold_pause = broken, True
    with pytest.raises(RuntimeError):
        runner.run(["a"], deadline_s=10_000)
    assert [e[0] for e in events] == ["pause", "resume"]


def test_measurement_waits_for_two_quiet_minutes_and_pauses_our_background_work():
    clock, measured, events = FakeClock(), [], []
    runner = _runner(lambda t: 4.0 if t < 600 else 0.5, measured, events, clock)
    rows = runner.run(["a"], deadline_s=10_000)
    assert [r["combo"] for r in rows] == ["a"] and rows[0]["noisy"] is False
    start = measured[0][1]
    assert start >= 600 + 120  # the 1-min mean falls below 1.5 some time after 600, then 2 min must pass
    assert events[0] == ("pause", start) and events[1][0] == "resume"


def test_noisy_run_is_marked_and_repeated_later():
    clock, measured, events = FakeClock(), [], []

    def loads(t: float) -> float:
        if 150 <= t < 200:  # someone else works for a while during the first measurement (120-220 s)
            return 6.0
        return 0.2

    rows = _runner(loads, measured, events, clock).run(["a"], deadline_s=10_000)
    assert [r["noisy"] for r in rows] == [True, False]
    assert len(measured) == 2 and measured[1][1] > measured[0][1]


def test_no_measurement_on_battery_or_without_the_performance_profile():
    for ac, profile in ((False, "performance"), (True, "balanced")):
        clock, measured, events = FakeClock(), [], []
        runner = _runner(lambda t: 0.0, measured, events, clock, ac=ac, profile=profile)
        rows = runner.run(["a"], deadline_s=1000)
        assert rows == [] and measured == []


def test_only_live_extraction_processes_are_paused(tmp_path: Path):
    import subprocess
    import sys

    from sentinel.quiet import extraction_pids, pause_extraction, register_extraction, resume_extraction

    extraction = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)", "sentinel", "eval", "extract"]
    )
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    for proc, marker in ((extraction, b"extract"), (other, b"sleep")):  # until fork has become exec
        for _ in range(500):
            if marker in Path(f"/proc/{proc.pid}/cmdline").read_bytes():
                break
            time.sleep(0.01)
    try:
        register_extraction(tmp_path, extraction.pid)
        register_extraction(tmp_path, other.pid)  # e.g. a PID reused by another program after a reboot
        (tmp_path / "999999999.pid").write_text("999999999")  # a process that is gone
        assert extraction_pids(tmp_path) == [extraction.pid]
        assert sorted(p.name for p in tmp_path.iterdir()) == [f"{extraction.pid}.pid"]  # stale files cleaned
        pause_extraction(tmp_path)
        state = Path(f"/proc/{extraction.pid}/stat").read_text().split(")")[1].split()[0]
        assert state == "T"
        resume_extraction(tmp_path)
        state = Path(f"/proc/{extraction.pid}/stat").read_text().split(")")[1].split()[0]
        assert state != "T"
        assert Path(f"/proc/{other.pid}/stat").read_text().split(")")[1].split()[0] != "T"
    finally:
        extraction.kill()
        other.kill()
