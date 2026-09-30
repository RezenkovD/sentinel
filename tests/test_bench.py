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
    PowerState,
    Timing,
    append_paced_row,
    append_row,
    check_same_setup,
    clip_info,
    read_power_state,
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
        "ac, performance",
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
        power=PowerState(source="ac", profile="performance"),
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


def test_report_refuses_to_mix_runs_on_different_clips_warmup_or_power(tmp_path: Path):
    log = tmp_path / "bench.jsonl"
    append_row(log, _row())  # fight_a.avi, 10-frame warm-up, on mains in the performance profile
    ac = PowerState(source="ac", profile="performance")
    check_same_setup([log, tmp_path / "no_such.paced.jsonl"], clips=["fight_a.avi"], warmup=10, power=ac)
    with pytest.raises(ValueError):
        check_same_setup([log], clips=["nonfight_b.avi"], warmup=10, power=ac)
    with pytest.raises(ValueError):
        check_same_setup([log], clips=["fight_a.avi"], warmup=0, power=ac)
    with pytest.raises(ValueError):  # the same run on battery is about twice as slow
        check_same_setup([log], clips=["fight_a.avi"], warmup=10, power=PowerState("battery", "balanced"))


def _supply(root: Path, name: str, kind: str, online: str | None = None) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "type").write_text(kind + "\n")
    if online is not None:
        (d / "online").write_text(online + "\n")


def test_power_state_reads_mains_adapter_and_profile(tmp_path: Path):
    _supply(tmp_path, "BAT0", "Battery")
    _supply(tmp_path, "ACAD", "Mains", online="1")
    state = read_power_state(tmp_path, profile_cmd=["echo", "performance"])
    assert state == PowerState(source="ac", profile="performance")


def test_power_state_on_battery(tmp_path: Path):
    _supply(tmp_path, "BAT0", "Battery")
    _supply(tmp_path, "ACAD", "Mains", online="0")
    assert read_power_state(tmp_path, profile_cmd=["echo", "balanced"]).source == "battery"


def test_power_state_unknown_without_mains_adapter_or_profile_tool(tmp_path: Path):
    state = read_power_state(tmp_path / "missing", profile_cmd=["no-such-profile-tool-xyz"])
    assert state == PowerState(source="unknown", profile="unknown")


def test_weapon_variants_are_parsed_from_the_command_line():
    from sentinel.bench import WeaponVariant

    assert WeaponVariant.parse("480:1") == WeaponVariant(mode="frame", imgsz=480, every_n_frames=1)
    assert WeaponVariant.parse("640:2") == WeaponVariant(mode="frame", imgsz=640, every_n_frames=2)
    assert WeaponVariant.parse("crops320:2") == WeaponVariant(mode="crops", imgsz=320, every_n_frames=2)
    assert WeaponVariant.parse("crops320:2").label == "crops320:2"
    with pytest.raises(ValueError):
        WeaponVariant.parse("640")


def _fake_export(pt: Path) -> None:
    out = pt.with_name(f"{pt.stem}_openvino_model")
    out.mkdir()
    (out / f"{pt.stem}.xml").write_text("<net/>")


def test_openvino_backend_loads_the_exports_of_all_three_models(tmp_path: Path):
    from sentinel.config import DEFAULT_CONFIG, load_config
    from sentinel.models import export_openvino
    from sentinel.pipeline import model_files

    config = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)
    people, pose, weapons = tmp_path / "people.pt", tmp_path / "pose.pt", tmp_path / "best.pt"
    for pt in (people, pose, weapons):
        pt.write_bytes(pt.name.encode())
        export_openvino(pt, _fake_export)
    models = config.models.model_copy(
        update={
            "backend": "openvino",
            "people": config.models.people.model_copy(update={"weights": people}),
            "pose": config.models.pose.model_copy(update={"weights": pose}),
        }
    )
    cfg = config.model_copy(
        update={"models": models, "weapons": config.weapons.model_copy(update={"weights": weapons})}
    )
    assert {role: p.name for role, p in model_files(cfg).items()} == {
        "people": "people_openvino_model",
        "pose": "pose_openvino_model",
        "weapons": "best_openvino_model",
    }


def test_people_variants_are_parsed_and_applied_to_the_config():
    from sentinel.bench import PeopleVariant, with_people
    from sentinel.config import DEFAULT_CONFIG, load_config

    config = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)
    detect = PeopleVariant.parse("detect:weights/yolo11n.pt")
    pose = PeopleVariant.parse("pose:weights/yolo26n-pose.pt")
    assert (detect.label, pose.label) == ("yolo11n detect", "yolo26n-pose track")
    cfg = with_people(config, detect)
    assert cfg.models.people.source == "detect" and cfg.models.people.weights == Path("weights/yolo11n.pt")
    assert cfg.models.pose.weights == config.models.pose.weights  # skeletons still from the pose model
    cfg = with_people(config, pose)
    assert cfg.models.people.source == "pose" and cfg.models.pose.weights == Path("weights/yolo26n-pose.pt")
    with pytest.raises(ValueError):
        PeopleVariant.parse("weights/yolo11n.pt")


def _quiet_row(backend: str, noisy: bool, rtf: float) -> dict:
    stage = {"mean_ms": 10.0, "p95_ms": 20.0}
    return {
        "backend": backend,
        "people": "yolo26n-pose track",
        "weapons": "480:2",
        "stages_ms": {k: stage for k in ("people", "pose", "weapons", "scene", "decision")},
        "total_ms": {"mean_ms": 50.0, "p95_ms": 90.0},
        "rtf": rtf,
        "paced": {
            "effective_fps": 9.9,
            "skipped_share": 0.67,
            "latency_ms": {"mean_ms": 60.0, "p95_ms": 80.0},
        },
        "frames": 1000,
        "power": {"source": "ac", "profile": "performance"},
        "load_start": 0.4,
        "load_end": 3.9,
        "time": "2026-09-30T23:10:00",
        "commit": "abc1234",
        "noisy": noisy,
        "foreign_load_mean": 2.5 if noisy else 0.3,
        "foreign_load_max": 3.1 if noisy else 0.6,
    }


def test_quiet_report_names_the_code_and_perception_it_measured():
    from sentinel.bench import render_quiet_report

    row = _quiet_row("openvino", noisy=False, rtf=0.5) | {"perception_version": "3.0"}
    text = render_quiet_report([row], sources=[])
    assert "abc1234" in text and "3.0" in text


def test_quiet_window_thresholds_come_from_the_config():
    from sentinel.config import DEFAULT_CONFIG, load_config

    quiet = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG).bench.quiet
    assert (quiet.start_limit, quiet.hold_s, quiet.noisy_limit, quiet.span_s) == (1.5, 120.0, 2.0, 60.0)


def test_quiet_report_shows_the_latest_clean_row_per_combination_and_counts_noisy_ones():
    from sentinel.bench import render_quiet_report

    rows = [
        _quiet_row("torch", noisy=True, rtf=1.3),
        _quiet_row("torch", noisy=False, rtf=0.61),
        _quiet_row("openvino", noisy=False, rtf=0.5),
    ]
    text = render_quiet_report(rows, sources=[])
    table = [line for line in text.splitlines() if line.startswith("| yolo26n-pose track")]
    assert len(table) == 2 and any("| torch |" in row and "| 0.61 |" in row for row in table)
    assert "1.30" not in text
    assert "Шумних прогонів: 1." in text


def test_quiet_report_says_whether_the_extraction_was_paused_for_the_whole_bench():
    from sentinel.bench import render_quiet_report

    held = render_quiet_report([_quiet_row("openvino", noisy=False, rtf=0.5) | {"extraction_held": True}], [])
    per_run = render_quiet_report([_quiet_row("openvino", noisy=False, rtf=0.5)], [])
    assert "призупинено на весь замір" in held and "на час кожного прогону" in per_run


def test_quiet_report_names_the_foreign_load_measure():
    from sentinel.bench import render_quiet_report

    cpu = render_quiet_report(
        [_quiet_row("openvino", noisy=False, rtf=0.5) | {"foreign_load": "cpu_time"}], []
    )
    tasks = render_quiet_report([_quiet_row("openvino", noisy=False, rtf=0.5)], [])
    assert "час процесора" in cpu and "задачі, що виконуються" in tasks
