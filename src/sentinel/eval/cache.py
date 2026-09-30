"""Reading the episode cache (data/episodes/) and quick statistics over it.

Every consumer of the experiment — aggregator A training, B, the cascade, the baselines, the weapon rule —
reads scenes through `load_episodes`, which refuses to mix files made by different perceptions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentinel.schema import SceneDescription

# windows without people: A (and B) return zero risk without looking (SPEC §5), so A's features are judged on
# windows with people only; pixel motion is also shown over all windows (the motion baseline sees them all)
PIXEL_FEATURES = ("motion_energy", "motion_area")
OUT_DEFAULT = Path("data/episodes")


@dataclass
class CachedEpisode:
    meta: dict
    scenes: list[SceneDescription]
    aux: list[dict]

    @property
    def episode_id(self) -> str:
        return self.meta["episode_id"]

    @property
    def label(self) -> int:
        return int(self.meta["label"])

    @property
    def part(self) -> str:
        return self.meta["part"]


def read_episode(path: Path) -> CachedEpisode:
    with path.open(encoding="utf-8") as f:
        meta = json.loads(f.readline())["meta"]
        scenes, aux = [], []
        for line in f:
            record = json.loads(line)
            scenes.append(SceneDescription.model_validate(record["scene"]))
            aux.append(record["aux"])
    return CachedEpisode(meta, scenes, aux)


def load_episodes(root: Path, dataset: str, parts: list[str] | None = None) -> list[CachedEpisode]:
    """Episodes of `dataset` (optionally only some parts), sorted by id; all must share one perception."""
    episodes = []
    for path in sorted((root / dataset).rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            meta = json.loads(f.readline())["meta"]
        if parts is None or meta["part"] in parts:
            episodes.append(read_episode(path))
    perceptions = {(e.meta["perception_version"], e.meta["perception_hash"]) for e in episodes}
    if len(perceptions) > 1:
        raise RuntimeError(f"episodes of {dataset} come from different perceptions: {sorted(perceptions)}")
    return sorted(episodes, key=lambda e: e.episode_id)


def _auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def feature_aucs(episodes: list[CachedEpisode], config) -> list[dict]:
    """One-feature window ROC-AUC for every feature of A (window label = episode label)."""
    from sentinel.aggregators.rules import FEATURE_NAMES, scene_to_features

    rows, labels, with_people = [], [], []
    for episode in episodes:
        for scene in episode.scenes:
            rows.append(scene_to_features(scene, config))
            labels.append(episode.label)
            with_people.append(scene.summary.n_persons > 0)
    x, y, people = np.array(rows), np.array(labels), np.array(with_people)
    table = []
    for j, name in enumerate(FEATURE_NAMES):
        row = {"feature": name, "auc": _auc(y[people], x[people, j]), "windows": int(people.sum())}
        if name in PIXEL_FEATURES:
            row |= {"auc_all": _auc(y, x[:, j]), "windows_all": len(y)}
        table.append(row)
    return table


def expected_episodes(dataset: str, parts: list[str], splits: Path = Path("data/splits")) -> int:
    """How many episodes the split files promise for these parts."""
    if dataset == "rwf2000":
        data = json.loads((splits / "rwf2000.json").read_text())
        return sum(len(data[p]) for p in parts)
    if dataset == "us_gun":
        data = json.loads((splits / "us_gun.json").read_text())
        return sum(len(data["segments"][p]) for p in parts)
    if dataset == "xd":
        return len(json.loads((splits / "xd_subset.json").read_text())["videos"])
    raise ValueError(f"unknown dataset {dataset!r}")


def check_complete(
    episodes: list[CachedEpisode],
    dataset: str,
    parts: list[str],
    splits: Path = Path("data/splits"),
    allow_partial: bool = False,
    warn=print,
) -> None:
    """Refuse to compute results on a cache that is still being extracted (unless explicitly allowed)."""
    expected = expected_episodes(dataset, parts, splits)
    if len(episodes) >= expected:
        return
    message = f"{dataset} {'+'.join(parts)}: {len(episodes)} of {expected} episodes in the cache"
    if not allow_partial:
        raise RuntimeError(message + " — wait for the extraction or pass --allow-partial")
    warn("WARNING: " + message + " (partial, --allow-partial)")
