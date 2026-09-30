"""Reading the episode cache and one-feature window AUCs (session 03, step 2 check)."""

import json
from pathlib import Path

import pytest

from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.eval.cache import feature_aucs, load_episodes
from sentinel.schema import Person, SceneDescription, Summary, Window, Zone

CONFIG = load_config(Path(__file__).parents[1] / DEFAULT_CONFIG)
ZONE = Zone(type="street", hours="day", expected_density="medium")


def _scene(t: float, speed: float, n_persons: int = 1, energy: float | None = 1.0) -> dict:
    persons = tuple(
        Person(id=i, speed=speed, limb_accel=None, holds=None, conf=0.9) for i in range(n_persons)
    )
    scene = SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=ZONE,
        summary=Summary(
            n_persons=n_persons,
            crowd_motion=speed if n_persons else 0.0,
            motion_energy=energy,
            weapon_visible=False,
        ),
        persons=persons,
    )
    return scene.model_dump(mode="json")


def _write(root: Path, episode_id: str, label: int, speeds: list[float], perception_hash: str = "h1", **kw):
    path = root / "rwf2000" / f"{episode_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode_id": episode_id,
        "part": episode_id.split("/")[0],
        "label": label,
        "perception_version": "3.0",
        "perception_hash": perception_hash,
    }
    lines = [json.dumps({"meta": meta})]
    for i, s in enumerate(speeds):
        lines.append(json.dumps({"scene": _scene(float(i), s, **kw), "aux": {"frames": 10, "label": None}}))
    path.write_text("\n".join(lines) + "\n")


def test_episodes_are_read_back_with_scenes_and_aux(tmp_path: Path):
    _write(tmp_path, "train/Fight/a", 1, [1.0, 2.0])
    _write(tmp_path, "val/NonFight/b", 0, [0.1])
    episodes = load_episodes(tmp_path, "rwf2000", parts=["train"])
    assert [e.episode_id for e in episodes] == ["train/Fight/a"]
    assert episodes[0].label == 1 and len(episodes[0].scenes) == 2 and episodes[0].aux[0]["frames"] == 10
    assert episodes[0].scenes[1].persons[0].speed == 2.0


def test_mixing_episodes_of_different_perception_is_refused(tmp_path: Path):
    _write(tmp_path, "train/Fight/a", 1, [1.0], perception_hash="h1")
    _write(tmp_path, "train/NonFight/b", 0, [0.1], perception_hash="h2")
    with pytest.raises(RuntimeError, match="perception"):
        load_episodes(tmp_path, "rwf2000", parts=["train"])


def test_feature_auc_ranks_windows_by_one_feature_with_the_clip_label(tmp_path: Path):
    for i in range(5):
        _write(tmp_path, f"train/Fight/f{i}", 1, [2.0 + i, 3.0])
        _write(tmp_path, f"train/NonFight/n{i}", 0, [0.1 * i, 0.5])
    episodes = load_episodes(tmp_path, "rwf2000", parts=["train"])
    table = {row["feature"]: row for row in feature_aucs(episodes, CONFIG)}
    assert table["max_speed"]["auc"] == pytest.approx(1.0)
    assert table["max_speed"]["windows"] == 20
    assert table["weapon_held"]["auc"] == pytest.approx(0.5)  # constant: no information


def test_windows_without_people_are_left_out_of_a_features_but_kept_for_pixel_motion(tmp_path: Path):
    _write(tmp_path, "train/Fight/f", 1, [2.0], n_persons=0, energy=9.0)
    _write(tmp_path, "train/Fight/g", 1, [2.0], n_persons=1, energy=8.0)
    _write(tmp_path, "train/NonFight/n", 0, [0.1], n_persons=1, energy=1.0)
    episodes = load_episodes(tmp_path, "rwf2000", parts=["train"])
    table = {row["feature"]: row for row in feature_aucs(episodes, CONFIG)}
    assert table["max_speed"]["windows"] == 2  # A never sees a scene without people (SPEC §5)
    assert table["motion_energy"]["windows_all"] == 3 and table["motion_energy"]["auc_all"] == pytest.approx(
        1.0
    )
