import json
from pathlib import Path

import numpy as np

from sentinel.agents.detector import Track
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.io.source import Frame
from sentinel.pipeline import Agents, Pipeline
from sentinel.schema import RiskResult, SceneDescription

CONFIG = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)


class TwoPeople:
    """Fake person tracker: two people, apart before t = 2 s and fighting close together after."""

    def update(self, image: np.ndarray, t: float = 0.0) -> list[Track]:
        gap = 300.0 if self.t < 2.0 else 40.0
        return [
            Track(1, (100.0, 0.0, 140.0, 100.0), 0.9),
            Track(2, (100.0 + gap, 0.0, 140.0 + gap, 100.0), 0.9),
        ]

    def reset(self) -> None:
        pass


class NoSkeletons:
    def estimate(self, image: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        return {}


class NoWeapons:
    def detect(self, image: np.ndarray, people=()) -> list:
        return []

    def reset(self) -> None:
        pass


class CloseMeansDanger:
    """Fake aggregator: danger when the closest pair is within one height."""

    def assess(self, scene: SceneDescription) -> RiskResult:
        close = any(i.distance_h < 1.0 for i in scene.interactions)
        return RiskResult(risk=0.9 if close else 0.1, level="alarm" if close else "normal", reason_uk="бійка")


class Collect:
    def __init__(self) -> None:
        self.events = []

    def on_event(self, event) -> None:
        self.events.append(event)

    def close(self) -> None:
        pass


class RecordKeyframes:
    def __init__(self) -> None:
        self.saved = []

    def __call__(self, incident_id: str, name: str, snapshot) -> Path:
        self.saved.append((name, snapshot))
        return Path(f"/fake/{incident_id}/{name}.jpg")


def _run(tmp_path: Path, seconds: float = 8.0, decider=None):
    people = TwoPeople()
    notifier, keyframes = Collect(), RecordKeyframes()
    journal = tmp_path / "run.jsonl"
    pipeline = Pipeline(
        Agents(people=people, poser=NoSkeletons(), weapons=NoWeapons()),
        CloseMeansDanger(),
        notifier,
        CONFIG,
        source_id="cam-1",
        run_id="test",
        journal=journal,
        keyframes=keyframes,
        decider=decider,
    )
    for i in range(int(seconds * 10)):
        t = round(0.1 * i, 1)
        people.t = t
        image = np.full((100, 700, 3), i % 256, dtype=np.uint8)  # the frame index is visible in the pixels
        pipeline.process(Frame(image=image, t=t, index=i), age_s=0.05)
    pipeline.finish()
    return notifier.events, keyframes.saved, [json.loads(line) for line in journal.read_text().splitlines()]


def test_frames_become_windows_risk_and_one_incident_closed_at_the_end(tmp_path: Path):
    events, _, journal = _run(tmp_path)
    assert [e.kind for e in events] == ["open", "close"]  # the fight lasts until the video ends
    assert events[0].incident_id.startswith("test-cam-1-") and events[1].stream_ended
    assert len(journal) == 9  # one line per 1 s window (the last flushed) + the end of the stream
    assert journal[-1]["stream_end"] and journal[-1]["events"] == [
        {"kind": "close", "incident_id": events[0].incident_id}
    ]


def test_journal_line_holds_scene_risk_state_frame_age_and_timing(tmp_path: Path):
    _, _, journal = _run(tmp_path)
    assert [e["kind"] for e in journal[4]["events"]] == ["open"]  # windows [2,3), [3,4), [4,5) are 3 of 5
    line = journal[5]
    assert line["scene"]["schema_version"] == 3 and line["scene"]["summary"]["n_persons"] == 2
    assert line["risk"]["risk"] == 0.9 and line["state"] == "тривога"
    assert line["frame_age_s"] == 0.05
    assert set(line["timing_ms"]) >= {"people", "pose", "weapons", "scene", "decision", "frames"}
    assert line["events"] == []


def test_keyframe_comes_from_a_frame_of_the_window_that_opened_the_incident(tmp_path: Path):
    events, saved, _ = _run(tmp_path)
    name, snapshot = saved[0]
    assert name == "keyframe"
    frame_index = int(snapshot.image[0, 0, 0])
    assert 20 <= frame_index < 50  # close together from t = 2 s; opened after k of n windows
    assert len(snapshot.tracks) == 2


def test_last_frame_stage_times_are_exposed_for_the_bench(tmp_path: Path):
    people = TwoPeople()
    people.t = 0.0
    pipeline = Pipeline(
        Agents(people=people, poser=NoSkeletons(), weapons=NoWeapons()),
        CloseMeansDanger(),
        Collect(),
        CONFIG,
        source_id="cam-1",
        run_id="t",
        journal=None,
        keyframes=RecordKeyframes(),
    )
    pipeline.process(Frame(image=np.zeros((10, 10, 3), dtype=np.uint8), t=0.0, index=0))
    assert set(pipeline.last_timing_ms) == {"people", "pose", "weapons", "motion", "scene", "decision"}


class PoseWithBystander(NoSkeletons):
    """Fake pose agent that also saw a person the tracker has not confirmed yet."""

    last_people = [((500.0, 0.0, 540.0, 100.0), np.zeros((17, 3)))]


def test_keyframe_snapshot_carries_untracked_people_for_face_blurring(tmp_path: Path):
    people = TwoPeople()
    keyframes = RecordKeyframes()
    pipeline = Pipeline(
        Agents(people=people, poser=PoseWithBystander(), weapons=NoWeapons()),
        CloseMeansDanger(),
        Collect(),
        CONFIG,
        source_id="cam-1",
        run_id="t",
        journal=None,
        keyframes=keyframes,
    )
    for i in range(60):
        people.t = round(0.1 * i, 1)
        pipeline.process(Frame(image=np.zeros((100, 700, 3), dtype=np.uint8), t=people.t, index=i))
    _, snapshot = keyframes.saved[0]
    assert snapshot.others == PoseWithBystander.last_people


class ScriptedReader:
    """Yields frames, then either ends or fails like a broken camera."""

    def __init__(self, frames: int, fail: bool) -> None:
        self.frames, self.fail = frames, fail

    def __iter__(self):
        for i in range(self.frames):
            yield Frame(image=np.zeros((4, 4, 3), dtype=np.uint8), t=0.1 * i, index=i)
        if self.fail:
            raise RuntimeError("reading the video source failed")


class RecordingPipeline:
    def __init__(self) -> None:
        self.processed, self.finished = 0, False

    def process(self, frame, age_s=None):
        self.processed += 1
        return []

    def finish(self):
        self.finished = True
        return []


def test_camera_failure_is_reported_and_open_incidents_are_still_closed():
    from sentinel.pipeline import drive

    pipeline = RecordingPipeline()
    result = drive(ScriptedReader(3, fail=True), pipeline, mode="live")
    assert pipeline.processed == 3 and pipeline.finished
    assert result.error and result.source_lost


def test_a_live_stream_that_just_ends_counts_as_lost():
    from sentinel.pipeline import drive

    assert drive(ScriptedReader(3, fail=False), RecordingPipeline(), mode="live").source_lost


def test_a_file_that_ends_is_not_a_failure():
    from sentinel.pipeline import drive

    result = drive(ScriptedReader(3, fail=False), RecordingPipeline(), mode="offline")
    assert not result.source_lost and result.error is None and result.frames == 3


class LowScoreGun:
    """Fake weapon detector: a gun with score 0.2 on every run, below the scene threshold."""

    def __init__(self) -> None:
        self.ran, self.last_max_conf = False, 0.0

    def detect(self, image: np.ndarray, people=()) -> list:
        self.ran, self.last_max_conf = True, 0.2
        return []

    def reset(self) -> None:
        self.ran, self.last_max_conf = False, 0.0


class Clock:
    t = 0.0


def test_perception_gives_each_scene_with_its_window_aux():
    from sentinel.pipeline import Perception
    from sentinel.schema import Zone

    people = TwoPeople()
    people.t = 0.0
    agents = Agents(people=people, poser=NoSkeletons(), weapons=LowScoreGun())
    perception = Perception(agents, CONFIG, "cam-1", Zone(type="street", hours="day", expected_density="low"))
    results = []
    for i in range(25):
        frame = Frame(image=np.zeros((120, 480, 3), dtype=np.uint8), t=round(0.1 * i, 1), index=i)
        scene, aux = perception.process(frame)[:2]
        if scene is not None:
            results.append((scene, aux))
    scene, aux = perception.flush()
    results.append((scene, aux))
    assert [s.window.t_start for s, _ in results] == [0.0, 1.0, 2.0]
    assert [a["frames"] for _, a in results] == [10, 10, 5]
    assert all(a["weapon_max_conf"] == 0.2 for _, a in results)


def test_the_journal_says_where_each_window_risk_came_from(tmp_path: Path):
    _, _, journal = _run(tmp_path)
    assert journal[3]["decision"] == {"mode": "A", "source": "A"}

    class FromB:
        mode = "B"

        def decide(self, scene, state):
            return RiskResult(risk=0.2, level="normal", reason_uk="спокійно"), {"mode": "B", "source": "B"}

    events, _, journal = _run(tmp_path / "b", decider=FromB())
    assert events == [] and journal[3]["decision"]["source"] == "B" and journal[3]["risk"]["risk"] == 0.2
