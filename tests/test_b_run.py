"""B on whole val clips for the cascade grid (session 05, step 4): a stratified, seeded subsample of clips."""

import pytest

from sentinel.eval.b_run import pick_clips
from sentinel.eval.cache import CachedEpisode


def _ep(i: int, label: int, part: str = "val") -> CachedEpisode:
    return CachedEpisode({"episode_id": f"{part}/{i}", "part": part, "label": label}, [], [])


def test_clips_are_half_fights_half_calm_seeded_and_nested():
    episodes = [_ep(i, i % 2) for i in range(100)]
    clips = pick_clips(episodes, 20, seed=2026)
    assert len(clips) == 20 and sum(e.label for e in clips) == 10
    assert clips == pick_clips(episodes, 20, seed=2026)
    assert set(e.episode_id for e in pick_clips(episodes, 10, seed=2026)) <= set(e.episode_id for e in clips)


def test_only_val_clips():
    with pytest.raises(ValueError, match="val"):
        pick_clips([_ep(0, 0, "test")], 1, seed=1)
