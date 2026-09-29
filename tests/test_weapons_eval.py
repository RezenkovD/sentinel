from pathlib import Path

import pytest

from eval.weapons import Detection, average_precision, frame_rates, read_voc

VOC = """<annotation><size><width>1920</width><height>1080</height></size>
<object><name>Handgun</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object>
<object><name>Short_rifle</name><bndbox><xmin>50</xmin><ymin>50</ymin><xmax>90</xmax><ymax>70</ymax></bndbox></object>
<object><name>Knife</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>9</xmax><ymax>25</ymax></bndbox></object>
<object><name>Smartphone</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object>
</annotation>"""


def test_voc_boxes_are_mapped_to_gun_and_knife(tmp_path: Path):
    xml = tmp_path / "frame.xml"
    xml.write_text(VOC)
    assert read_voc(xml) == [
        ("gun", (10.0, 20.0, 30.0, 40.0)),
        ("gun", (50.0, 50.0, 90.0, 70.0)),
        ("knife", (5.0, 5.0, 9.0, 25.0)),
    ]


BOX = (0.0, 0.0, 10.0, 10.0)
GT = {"a": [("gun", BOX)], "b": [("gun", BOX)], "c": []}


def test_ap50_of_perfect_detections_is_one():
    dets = [Detection("a", "gun", 0.9, BOX), Detection("b", "gun", 0.8, BOX)]
    assert average_precision(dets, GT, "gun") == pytest.approx(1.0)


def test_ap50_with_a_false_positive_ranked_between_true_ones():
    # ranks: TP (P 1, R .5), FP (P .5, R .5), TP (P .67, R 1) -> envelope 1.0 up to R .5, then .67
    dets = [
        Detection("a", "gun", 0.9, BOX),
        Detection("c", "gun", 0.8, BOX),
        Detection("b", "gun", 0.7, BOX),
    ]
    assert average_precision(dets, GT, "gun") == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_second_detection_of_the_same_weapon_is_a_false_positive():
    dets = [Detection("a", "gun", 0.9, BOX), Detection("a", "gun", 0.8, BOX)]
    assert average_precision(dets, GT, "gun") == pytest.approx(0.5)  # recall stops at 1/2


def test_box_that_misses_the_weapon_does_not_count():
    dets = [Detection("a", "gun", 0.9, (20.0, 20.0, 30.0, 30.0))]
    assert average_precision(dets, GT, "gun") == pytest.approx(0.0)


def test_frame_recall_and_false_frame_share_per_threshold():
    gt = {"w1": [("gun", BOX)], "w2": [("knife", BOX)], "e1": [], "e2": [], "e3": [], "e4": []}
    dets = [
        Detection("w1", "gun", 0.6, BOX),
        Detection("w2", "gun", 0.3, BOX),  # any weapon class counts at frame level
        Detection("e1", "knife", 0.5, BOX),
    ]
    rates = frame_rates(dets, gt, thresholds=[0.25, 0.4, 0.55, 0.7])
    assert rates == [
        (0.25, 1.0, 0.25),
        (0.4, 0.5, 0.25),
        (0.55, 0.5, 0.0),
        (0.7, 0.0, 0.0),
    ]
