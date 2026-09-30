"""Datasets -> splits (and, later, episodes).

Run from the repository root: `uv run sentinel eval build-episodes` writes data/splits/*.json.

RWF-2000 (decision 29.09): the original `val` (400 clips) is our test, untouched until the experiment;
the original `train` (1600) becomes our train / val 80/20, stratified by class with a fixed seed.
Clips are cut from longer source videos (`<video id>_<segment>.avi`), so whole source videos go to one
side: otherwise neighbouring segments of one video would sit in train and val at once.

Sevilla (US gun): per-frame data split by camera so that neighbouring frames never land in different parts:
Cam1 + Cam5 select the weapon weights and threshold, Cam7 is the held-out test.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

RWF_SEED = 2026
VAL_SHARE = 0.2
RWF_CLASSES = ("Fight", "NonFight")


@dataclass(frozen=True)
class Clip:
    path: str  # relative to the dataset root, e.g. "train/Fight/abc_0.avi"
    label: str  # "Fight" / "NonFight"
    original_split: str  # "train" / "val" as published
    group: str  # source video


def source_group(stem: str, renamed: dict[str, str] | None = None) -> str:
    """Source video of a clip: its file name without the trailing `_<segment>` index."""
    original = (renamed or {}).get(stem, stem)
    return re.sub(r"_\d+$", "", original)


def read_renamed(root: Path) -> dict[str, str]:
    """New stem -> original stem for clips renamed because their names exceeded the ext4 limit."""
    table = root / "renamed.tsv"
    if not table.exists():
        return {}
    renamed = {}
    for line in table.read_text(encoding="utf-8").splitlines()[1:]:
        new_path, original_hex, _size = line.split("\t")
        original = bytes.fromhex(original_hex).decode("utf-8", "surrogateescape")
        renamed[Path(new_path).stem] = original.removesuffix(".avi")
    return renamed


def collect_rwf2000(root: Path) -> list[Clip]:
    renamed = read_renamed(root)
    clips = []
    for split in ("train", "val"):
        for label in RWF_CLASSES:
            for f in sorted((root / split / label).glob("*.avi")):
                rel = f.relative_to(root).as_posix()
                clips.append(Clip(rel, label, split, source_group(f.stem, renamed)))
    return clips


def split_rwf2000(clips: list[Clip], val_share: float, seed: int) -> dict[str, list[str]]:
    """Our train / val from the original train (whole source videos, by class); test = original val."""
    test = sorted(c.path for c in clips if c.original_split == "val")
    pool = [c for c in clips if c.original_split == "train"]
    by_group: dict[str, list[Clip]] = defaultdict(list)
    for c in pool:
        by_group[c.group].append(c)
    target = {label: round(val_share * n) for label, n in Counter(c.label for c in pool).items()}
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    in_val: Counter[str] = Counter()
    val: list[str] = []
    for g in groups:  # take whole source videos while no class overshoots its target
        counts = Counter(c.label for c in by_group[g])
        if all(in_val[label] + n <= target[label] for label, n in counts.items()):
            in_val += counts
            val += [c.path for c in by_group[g]]
    val_set = set(val)
    return {
        "train": sorted(c.path for c in pool if c.path not in val_set),
        "val": sorted(val),
        "test": test,
    }


def split_us_gun(images: Path) -> dict:
    frames = sorted(p.name for p in images.glob("*.jpg"))
    by_camera = Counter(name.split("-", 1)[0] for name in frames)
    segments: dict[str, set[str]] = defaultdict(set)
    for name in frames:
        segments[name.split("-", 1)[0]].add(re.sub(r"_frame_\d+\.jpg$", "", name))
    parts = {"select": ["Cam1", "Cam5"], "test": ["Cam7"]}
    return {
        "dataset": "us-gun (Sevilla mock attack, frames at 2 fps)",
        "rule": "split by camera; file name prefix Cam<N>-",
        "parts": parts,
        "frames": {part: sum(by_camera[c] for c in cams) for part, cams in parts.items()},
        "segments": {part: sorted(s for c in cams for s in segments[c]) for part, cams in parts.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rwf", type=Path, default=Path("data/raw/rwf2000"))
    parser.add_argument("--us-gun", type=Path, default=Path("data/raw/us-gun/Images"))
    parser.add_argument("--out", type=Path, default=Path("data/splits"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    clips = collect_rwf2000(args.rwf)
    split = split_rwf2000(clips, VAL_SHARE, RWF_SEED)
    label = {c.path: c.label for c in clips}
    rwf = {
        "dataset": "rwf2000",
        "rule": (
            "test = original val; our train/val = original train split 80/20 by whole source videos "
            "(file name without trailing _<segment>), stratified by class"
        ),
        "seed": RWF_SEED,
        "val_share": VAL_SHARE,
        "source_videos": len({c.group for c in clips}),
        "counts": {part: dict(Counter(label[p] for p in paths)) for part, paths in split.items()},
        **split,
    }
    (args.out / "rwf2000.json").write_text(
        json.dumps(rwf, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    (args.out / "us_gun.json").write_text(
        json.dumps(split_us_gun(args.us_gun), ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(rwf["counts"]))


if __name__ == "__main__":
    main()
