import numpy as np
import pytest

from sentinel.agents.weapons import WeaponBox, WeaponDetector, crops_run, map_class

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

    def __call__(self, frame: np.ndarray, people=()) -> list[WeaponBox]:
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


def test_detector_reports_whether_the_model_ran_on_the_last_frame():
    detector = WeaponDetector(CountingModel(), every_n_frames=2)
    ran = []
    for _ in range(5):
        detector.detect(FRAME)
        ran.append(detector.ran)
    assert ran == [True, False, True, False, True]


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


def test_crop_mode_runs_the_model_around_each_person_and_maps_boxes_back():
    seen_shapes = []

    def predict(image: np.ndarray) -> list[WeaponBox]:
        seen_shapes.append(image.shape[:2])
        return [WeaponBox("knife", (10.0, 20.0, 30.0, 40.0), 0.7)]  # in crop coordinates

    run = crops_run(predict, pad=0.25)
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    boxes = run(frame, [(900.0, 400.0, 1000.0, 600.0)])
    assert seen_shapes == [(250, 250)]  # padded square around the 100 x 200 person
    assert boxes == [WeaponBox("knife", (835.0, 395.0, 855.0, 415.0), 0.7)]  # crop origin (825, 375)


def test_crop_mode_keeps_one_box_for_a_weapon_seen_from_two_overlapping_crops():
    def predict(image: np.ndarray) -> list[WeaponBox]:
        return [WeaponBox("gun", (0.0, 0.0, 40.0, 40.0), 0.6)]

    run = crops_run(predict, pad=0.0)
    frame = np.zeros((500, 500, 3), dtype=np.uint8)
    # two people whose squares both start at the frame corner, so both crops see the same gun
    boxes = run(frame, [(0.0, 0.0, 100.0, 100.0), (0.0, 0.0, 90.0, 100.0)])
    assert len(boxes) == 1


def test_crop_mode_without_people_finds_nothing():
    run = crops_run(lambda image: [WeaponBox("gun", (0.0, 0.0, 1.0, 1.0), 0.9)], pad=0.25)
    assert run(np.zeros((10, 10, 3), dtype=np.uint8), []) == []


def test_boxes_below_the_threshold_are_dropped_but_the_best_raw_score_is_kept():
    # extraction runs the model at a low score to record the best one for the baseline (aux), while the
    # scene only gets boxes at or above the configured threshold
    def model(frame, people=()):
        return [WeaponBox("gun", (0.0, 0.0, 5.0, 5.0), 0.12), WeaponBox("knife", (6.0, 0.0, 9.0, 5.0), 0.45)]

    detector = WeaponDetector(model, every_n_frames=1, conf=0.3)
    assert [b.kind for b in detector.detect(FRAME)] == ["knife"]
    assert detector.last_max_conf == pytest.approx(0.45)
    detector = WeaponDetector(
        lambda frame, people=(): [WeaponBox("gun", (0.0, 0.0, 5.0, 5.0), 0.12)], conf=0.3
    )
    assert detector.detect(FRAME) == [] and detector.last_max_conf == pytest.approx(0.12)
