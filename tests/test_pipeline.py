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


def _run(tmp_path: Path, seconds: float = 8.0):
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
    )
    for i in range(int(seconds * 10)):
        t = round(0.1 * i, 1)
        people.t = t
        image = np.full((100, 700, 3), i % 256, dtype=np.uint8)  # the frame index is visible in the pixels
        pipeline.process(Frame(image=image, t=t, index=i), age_s=0.05)
    pipeline.finish()
    return notifier.events, keyframes.saved, [json.loads(line) for line in journal.read_text().splitlines()]


def test_frames_become_windows_risk_and_one_incident(tmp_path: Path):
    events, _, journal = _run(tmp_path)
    assert [e.kind for e in events] == ["open"]
    assert events[0].incident_id.startswith("test-cam-1-")
    assert len(journal) == 8  # one line per 1 s window, the last one flushed at the end


def test_journal_line_holds_scene_risk_state_frame_age_and_timing(tmp_path: Path):
    _, _, journal = _run(tmp_path)
    assert journal[4]["events"] == ["open"]  # windows [2,3), [3,4), [4,5) are the 3 of 5
    line = journal[5]
    assert line["scene"]["schema_version"] == 2 and line["scene"]["summary"]["n_persons"] == 2
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
