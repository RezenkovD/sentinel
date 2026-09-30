"""Episode cache (session 03, step 3): datasets -> episodes -> scene descriptions per window."""

import json
import os
from pathlib import Path

import pytest

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.extract import (
    Episode,
    extract_episode,
    is_current,
    perception_hash,
    select_xd_subset,
    sevilla_frames,
    window_label,
    xd_labels,
)
from sentinel.schema import SceneDescription, Summary, Window, Zone

CONFIG = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)


def test_window_is_positive_when_at_least_half_of_it_is_inside_an_annotated_event():
    events = [(10.0, 12.5)]
    assert window_label(10.0, 11.0, events) == 1
    assert window_label(12.0, 13.0, events) == 1  # exactly half
    assert window_label(12.4, 13.4, events) == 0
    assert window_label(3.0, 4.0, []) == 0


def test_xd_labels_come_from_the_file_name():
    assert xd_labels("Brick.Mansions.2014__#00-41-25_00-42-36_label_B1-0-0") == ["B1"]
    assert xd_labels("v=abc__#1_label_B2-G-0") == ["B2", "G"]
    assert xd_labels("About.Time.2013__#00-23-50_00-24-31_label_A") == ["A"]


def test_xd_subset_is_balanced_by_duration_and_skips_out_of_scope_videos():
    videos = []
    for i in range(40):
        videos.append({"name": f"f{i}_label_B1-0-0", "duration_s": 60.0})
        videos.append({"name": f"s{i}_label_B2-0-0", "duration_s": 90.0})
        videos.append({"name": f"n{i}_label_A", "duration_s": 120.0})
        videos.append({"name": f"r{i}_label_B4-0-0", "duration_s": 60.0})  # riots: out of scope
        videos.append({"name": f"x{i}_label_B2-G-0", "duration_s": 60.0})  # shooting with explosions
    subset = select_xd_subset(videos, budget_hours=1.5, seed=2026)
    by_stratum = {}
    for v in subset["videos"]:
        by_stratum[v["stratum"]] = by_stratum.get(v["stratum"], 0.0) + v["duration_s"]
    assert set(by_stratum) == {"B1", "B2", "A"}
    assert all(abs(d - 1800.0) <= 120.0 for d in by_stratum.values())  # ~0.5 h each
    assert subset["excluded"]["out_of_scope"] == 80
    assert subset == select_xd_subset(videos, budget_hours=1.5, seed=2026)  # fixed seed


def test_a_short_shooting_share_is_filled_with_other_videos_that_have_shooting():
    """Decision 30.09 (report 03, item 9): labels within {B1, B2} may be mixed. When the B2-first videos run
    out before the B2 share, videos with B2 among other labels fill it; nothing chosen before is dropped."""
    videos = []
    for i in range(40):
        videos.append({"name": f"f{i}_label_B1-0-0", "duration_s": 60.0})
        videos.append({"name": f"n{i}_label_A", "duration_s": 120.0})
    videos += [{"name": f"s{i}_label_B2-0-0", "duration_s": 90.0} for i in range(5)]  # 450 s < 1800 s share
    videos += [{"name": f"m{i}_label_B1-B2-0", "duration_s": 60.0} for i in range(6)]
    videos += [{"name": "g0_label_B2-G-0", "duration_s": 60.0}]  # explosions: still out of scope
    before = select_xd_subset(videos, budget_hours=1.5, seed=2026, all_shooting=False)
    subset = select_xd_subset(videos, budget_hours=1.5, seed=2026)
    names = [v["name"] for v in subset["videos"]]
    assert [v["name"] for v in before["videos"]] == names[: len(before["videos"])]
    added = names[len(before["videos"]) :]
    assert added and all(n.startswith("m") for n in added)  # only mixed B1-B2 ones, never B2-G
    assert all(v["stratum"] == "B2" for v in subset["videos"][len(before["videos"]) :])
    assert len(added) + sum(n.startswith("m") for n in names[: len(before["videos"])]) == 6


def test_sevilla_frames_are_timed_at_two_per_second_and_labelled_from_xml(tmp_path: Path):
    for n, has_gun in ((100, False), (101, True), (103, True)):
        stem = f"Cam1-X_Segment_0_x264_frame_{n}"
        (tmp_path / f"{stem}.jpg").write_bytes(b"")
        objects = "<object><name>Handgun</name></object>" if has_gun else ""
        (tmp_path / f"{stem}.xml").write_text(f"<annotation>{objects}</annotation>")
    frames = sevilla_frames(tmp_path, "Cam1-X_Segment_0_x264")
    assert [(t, label) for t, _, label in frames] == [(0.0, 0), (0.5, 1), (1.5, 1)]  # frame 102 is missing


def test_perception_hash_follows_perception_settings_only():
    base = perception_hash(CONFIG)
    assert base == perception_hash(CONFIG)
    other_threshold = CONFIG.model_copy(
        update={"motion": CONFIG.motion.model_copy(update={"threshold": CONFIG.motion.threshold + 1})}
    )
    assert perception_hash(other_threshold) != base
    other_alarm = CONFIG.model_copy(update={"incidents": CONFIG.incidents.model_copy(update={"k": 2})})
    assert perception_hash(other_alarm) == base  # decisions downstream do not change the cache


class FakePerception:
    """Two windows of scenes with aux, like Perception.process / flush."""

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1
        self._n = 0

    def _scene(self, t0: float, t1: float) -> SceneDescription:
        return SceneDescription(
            window=Window(t_start=t0, t_end=t1, source="cam-1"),
            zone=Zone(type="street", hours="day", expected_density="medium"),
            summary=Summary(n_persons=0, crowd_motion=0.0, weapon_visible=False),
        )

    def process(self, frame):
        self._n += 1
        if frame.t == 1.0:  # the first frame of the second window closes the first one
            return self._scene(0.0, 1.0), {"frames": 2, "weapon_max_conf": 0.3}, None
        return None, None, None

    def flush(self):
        return self._scene(1.0, 1.5), {"frames": 2, "weapon_max_conf": 0.0}


class FakeFrame:
    def __init__(self, t: float) -> None:
        self.t, self.image = t, None


def _episode(tmp_path: Path) -> Episode:
    frames = [FakeFrame(t) for t in (0.0, 0.5, 1.0, 1.5)]
    return Episode(
        dataset="demo",
        part="dev",
        episode_id="dev/clip_1",
        label=1,
        duration_s=2.0,
        frames=lambda: iter(frames),
        events=[(0.0, 0.6)],
        frame_labels=None,
    )


def test_an_episode_is_written_as_meta_plus_one_line_per_window(tmp_path: Path):
    out = tmp_path / "dev" / "clip_1.jsonl"
    meta = {"perception_version": "3.0", "perception_hash": "abc"}
    assert extract_episode(_episode(tmp_path), FakePerception(), out, meta) is True
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert lines[0]["meta"]["episode_id"] == "dev/clip_1" and lines[0]["meta"]["label"] == 1
    assert lines[0]["meta"]["perception_hash"] == "abc" and lines[0]["meta"]["windows"] == 2
    assert [line["aux"]["label"] for line in lines[1:]] == [1, 0]  # 0.6 s of the first window is an event
    assert lines[1]["aux"]["weapon_max_conf"] == 0.3
    assert SceneDescription.model_validate(lines[2]["scene"]).window.t_end == 1.5


def test_a_finished_episode_is_skipped_and_a_different_perception_is_refused(tmp_path: Path):
    out = tmp_path / "clip.jsonl"
    meta = {"perception_version": "3.0", "perception_hash": "abc"}
    perception = FakePerception()
    extract_episode(_episode(tmp_path), perception, out, meta)
    assert extract_episode(_episode(tmp_path), perception, out, meta) is False and perception.resets == 1
    with pytest.raises(RuntimeError, match="perception"):
        extract_episode(_episode(tmp_path), perception, out, meta | {"perception_hash": "new"})


def test_a_file_of_the_compatible_perception_3_0_stays_current_for_3_1(tmp_path: Path):
    """Decision 30.09 (report 03): 3.1 only normalises pixel motion, caches of 3.0 are not extracted again."""
    out = tmp_path / "clip.jsonl"
    perception = FakePerception()
    v30 = {"perception_version": "3.0", "perception_hash": "abc"}
    extract_episode(_episode(tmp_path), perception, out, v30)
    assert is_current(out, {"perception_version": "3.1", "perception_hash": "abc"})
    assert not is_current(out, {"perception_version": "3.1", "perception_hash": "other"})
    old = tmp_path / "old.jsonl"
    extract_episode(_episode(tmp_path), perception, old, v30 | {"perception_version": "2.0"})
    assert not is_current(old, {"perception_version": "3.1", "perception_hash": "abc"})


def test_an_interrupted_episode_leaves_no_file_behind(tmp_path: Path):
    out = tmp_path / "clip.jsonl"

    class Broken(FakePerception):
        def process(self, frame):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        extract_episode(
            _episode(tmp_path), Broken(), out, {"perception_version": "3.0", "perception_hash": "x"}
        )
    assert not out.exists() and list(tmp_path.iterdir()) == []


def test_weapon_variant_for_a_separate_cache_is_applied_to_the_config():
    from sentinel.eval.extract import with_weapon_variant

    cfg = with_weapon_variant(CONFIG, "crops320:2", conf=0.40)
    assert (cfg.weapons.mode, cfg.weapons.imgsz, cfg.weapons.every_n_frames, cfg.weapons.conf) == (
        "crops",
        320,
        2,
        0.40,
    )
    assert perception_hash(cfg) != perception_hash(CONFIG)  # a separate cache, never mixed with the main one


def _xd_files(tmp_path: Path, annotation_lines: list[str], videos: list[dict]) -> tuple[Path, Path]:
    annotations = tmp_path / "annotations.txt"
    annotations.write_text("\n".join(annotation_lines))
    subset = tmp_path / "subset.json"
    subset.write_text(json.dumps({"videos": videos}))
    return subset, annotations


def test_xd_annotation_names_may_carry_the_file_extension(tmp_path: Path):
    from sentinel.eval.extract import xd_episodes

    subset, annotations = _xd_files(
        tmp_path,
        ["v=abc__#1_label_B1-0-0.mp4 24 48 96 120", "v=def__#1_label_B2-0-0 0 24"],
        [
            {"name": "v=abc__#1_label_B1-0-0", "stratum": "B1", "duration_s": 10.0},
            {"name": "v=def__#1_label_B2-0-0", "stratum": "B2", "duration_s": 10.0},
        ],
    )
    episodes = xd_episodes(subset, tmp_path, annotations, fps=10.0)  # no video files: 24 frames/s assumed
    assert episodes[0].events == [(1.0, 2.0), (4.0, 5.0)] and episodes[1].events == [(0.0, 1.0)]


def test_a_violent_xd_video_without_annotation_is_an_error(tmp_path: Path):
    from sentinel.eval.extract import xd_episodes

    subset, annotations = _xd_files(
        tmp_path, [], [{"name": "v=abc__#1_label_B1-0-0", "stratum": "B1", "duration_s": 10.0}]
    )
    with pytest.raises(ValueError, match="annotation"):
        xd_episodes(subset, tmp_path, annotations, fps=10.0)


def test_redo_replaces_only_files_of_another_perception(tmp_path: Path):
    current, stale = tmp_path / "current.jsonl", tmp_path / "stale.jsonl"
    meta = {"perception_version": "3.0", "perception_hash": "new"}
    perception = FakePerception()
    extract_episode(_episode(tmp_path), perception, current, meta)
    extract_episode(_episode(tmp_path), perception, stale, meta | {"perception_hash": "old"})
    resets = perception.resets
    assert extract_episode(_episode(tmp_path), perception, current, meta, redo=True) is False
    assert extract_episode(_episode(tmp_path), perception, stale, meta, redo=True) is True
    assert perception.resets == resets + 1
    assert json.loads(stale.read_text().splitlines()[0])["meta"]["perception_hash"] == "new"


def test_a_file_cut_short_by_a_power_loss_is_extracted_again(tmp_path: Path):
    out = tmp_path / "clip.jsonl"
    out.write_text("")  # an empty file left by a crash before the data reached the disk
    meta = {"perception_version": "3.0", "perception_hash": "abc"}
    assert extract_episode(_episode(tmp_path), FakePerception(), out, meta) is True
    assert json.loads(out.read_text().splitlines()[0])["meta"]["windows"] == 2


def test_one_failing_episode_does_not_stop_the_others(tmp_path: Path):
    from sentinel.eval.extract import extract_all

    class FailsOnSecond(FakePerception):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def reset(self) -> None:
            super().reset()
            self.calls += 1
            if self.calls == 2:
                raise OSError("cannot read the video")

    episodes = []
    for i in range(3):
        e = _episode(tmp_path)
        e.episode_id = f"dev/clip_{i}"
        episodes.append(e)
    log: list[str] = []
    counts = extract_all(
        episodes,
        FailsOnSecond(),
        tmp_path,
        {"perception_version": "3.0", "perception_hash": "x"},
        redo=False,
        log=log.append,
    )
    assert counts == {"done": 2, "skipped": 0, "failed": 1}
    assert (tmp_path / "demo" / "dev" / "clip_0.jsonl").exists() and (
        tmp_path / "demo" / "dev" / "clip_2.jsonl"
    ).exists()
    assert any("clip_1" in line and "cannot read" in line for line in log)


def test_meta_names_every_weight_file_hash_and_the_library_versions():
    from sentinel.eval.extract import stage_meta

    meta = stage_meta(CONFIG)
    assert set(meta["weights_sha256"]) == {"people", "pose", "weapons"}
    assert all(len(h) == 64 for h in meta["weights_sha256"].values())
    assert {"ultralytics", "openvino", "torch", "opencv", "numpy"} <= set(meta["versions"])


@pytest.mark.parametrize(
    "argv",
    [["--dataset", "rwf2000"], ["--dataset", "rwf2000", "--parts", "train", "--weapons", "crops320:2"]],
)
def test_risky_argument_combinations_are_refused_before_anything_starts(argv: list[str]):
    from sentinel.eval.extract import main

    before = os.nice(0)
    with pytest.raises(SystemExit):
        main(argv)  # no --parts would silently mean the test part; --weapons without --out mixes caches
    assert os.nice(0) == before  # refused before the process lowered its priority


def test_cores_are_the_last_ones_unless_given():
    from sentinel.eval.extract import pick_cores

    assert pick_cores(list(range(16)), threads=4, cpus=None) == [12, 13, 14, 15]
    assert pick_cores(list(range(16)), threads=4, cpus=[8, 9, 10, 11]) == [8, 9, 10, 11]
    with pytest.raises(ValueError):
        pick_cores(list(range(8)), threads=4, cpus=[8, 9])  # not available to this process
