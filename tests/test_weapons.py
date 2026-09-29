import numpy as np
import pytest

from sentinel.agents.weapons import WeaponBox, WeaponDetector, map_class

CLASS_MAP = {"pistol": "gun", "rifle": "gun", "gun": "gun", "knife": "knife"}


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Pistol", "gun"),
        ("Rifle", "gun"),
        ("gun", "gun"),
        ("Knife", "knife"),
        ("knife", "knife"),
        ("grenade", None),
    ],
)
def test_candidate_class_names_map_to_gun_or_knife(name: str, kind: str | None):
    assert map_class(name, CLASS_MAP) == kind


class CountingModel:
    """Stand-in for the YOLO call: returns a new box on every real run, so reuse is visible."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: np.ndarray) -> list[WeaponBox]:
        self.calls += 1
        return [WeaponBox("gun", (0.0, 0.0, 10.0, 10.0), conf=self.calls / 10)]


FRAME = np.zeros((8, 8, 3), dtype=np.uint8)


def test_every_frame_mode_runs_the_model_on_each_frame():
    model = CountingModel()
    detector = WeaponDetector(model, every_n_frames=1)
    confs = [detector.detect(FRAME)[0].conf for _ in range(3)]
    assert model.calls == 3 and confs == [0.1, 0.2, 0.3]


def test_every_second_frame_mode_keeps_the_last_boxes_in_between():
    model = CountingModel()
    detector = WeaponDetector(model, every_n_frames=2)
    confs = [detector.detect(FRAME)[0].conf for _ in range(5)]
    assert model.calls == 3  # frames 1, 3, 5
    assert confs == [0.1, 0.1, 0.2, 0.2, 0.3]


def test_reset_forgets_boxes_and_restarts_the_schedule():
    model = CountingModel()
    detector = WeaponDetector(model, every_n_frames=2)
    detector.detect(FRAME)
    detector.reset()
    assert detector.detect(FRAME)[0].conf == 0.2  # runs right away after reset
    assert model.calls == 2


def test_invalid_schedule_is_rejected():
    with pytest.raises(ValueError):
        WeaponDetector(CountingModel(), every_n_frames=0)
