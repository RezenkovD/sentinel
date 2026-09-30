"""Metadata for fine-tuning the weapon detector on Kaggle (session 03, step 8); no images leave the PC."""

import pytest

from sentinel.eval.cache import CachedEpisode
from sentinel.eval.weapon_finetune import hard_negative_windows, split_segments
from sentinel.schema import SceneDescription, Summary, Window, Zone

ZONE = Zone(type="street", hours="day", expected_density="medium")


def _episode(episode_id: str, label: int, confs: list[float]) -> CachedEpisode:
    scenes = [
        SceneDescription(
            window=Window(t_start=float(t), t_end=float(t + 1), source="cam-1"),
            zone=ZONE,
            summary=Summary(n_persons=1, crowd_motion=0.1, weapon_visible=c >= 0.3),
        )
        for t, c in enumerate(confs)
    ]
    meta = {
        "episode_id": episode_id,
        "label": label,
        "part": episode_id.split("/")[0],
        "source": f"{episode_id}.avi",
    }
    return CachedEpisode(meta, scenes, [{"weapon_max_conf": c, "label": None} for c in confs])


def test_hard_negatives_are_windows_of_normal_train_clips_where_the_detector_fires():
    episodes = [
        _episode("train/NonFight/a", 0, [0.1, 0.45, 0.2]),
        _episode("train/Fight/b", 1, [0.9]),  # fights are not negatives: RWF has no weapon labels at all
        _episode("train/NonFight/c", 0, [0.05]),
    ]
    windows = hard_negative_windows(episodes, min_conf=0.3)
    assert windows == [
        {"clip": "train/NonFight/a.avi", "t_start": 1.0, "t_end": 2.0, "weapon_max_conf": 0.45}
    ]


def test_test_parts_never_become_training_data():
    with pytest.raises(ValueError, match="test"):
        hard_negative_windows([_episode("test/NonFight/x", 0, [0.9])], min_conf=0.3)
    with pytest.raises(ValueError, match="Cam7"):
        split_segments(["Cam1-A_Segment_0", "Cam7-B_Segment_1"], val_share=0.2, seed=2026)


def test_sevilla_segments_split_whole_into_train_and_val():
    segments = [f"Cam{c}-X_Segment_{i}" for c in (1, 5) for i in range(7)]
    split = split_segments(segments, val_share=0.2, seed=2026)
    assert set(split["train"]) | set(split["val"]) == set(segments)
    assert not set(split["train"]) & set(split["val"])
    assert len(split["val"]) == 3  # round(14 x 0.2)
    assert split == split_segments(segments, val_share=0.2, seed=2026)
