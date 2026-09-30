"""Metadata for fine-tuning the weapon detector on Kaggle GPU (session 03, step 8).

    uv run sentinel eval weapon-finetune       # -> data/splits/weapon_finetune.json

No image or video leaves this machine: the notebook (notebooks/train_weapon_detector.ipynb) downloads the
public sources itself — Sevilla frames and the authors' synthetic Unity set from their Hugging Face page, RWF
from the same mirror we use, the WeaponShield checkpoint from its pinned GitHub commit — and this file only
says what to take:
- Sevilla Cam1 + Cam5, split into train / val by whole segments (seed 2026); Cam7 never;
- hard negatives: windows of our RWF **train** NonFight clips where the current detector scores a weapon at
  least `min_conf` (the scene threshold); the notebook cuts crops around those false boxes as background
  images. The test part of RWF is never used.
"""

import argparse
import json
import random
from pathlib import Path

from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, load_episodes


def hard_negative_windows(episodes: list[CachedEpisode], min_conf: float) -> list[dict]:
    """Windows of normal (NonFight) train clips where the detector scored a weapon >= min_conf."""
    out = []
    for episode in episodes:
        if episode.part == "test":
            raise ValueError(f"{episode.episode_id}: the test part must never become training data")
        if episode.label != 0:
            continue
        for scene, aux in zip(episode.scenes, episode.aux, strict=True):
            if aux["weapon_max_conf"] >= min_conf:
                out.append(
                    {
                        "clip": episode.meta["source"],
                        "t_start": scene.window.t_start,
                        "t_end": scene.window.t_end,
                        "weapon_max_conf": aux["weapon_max_conf"],
                    }
                )
    return out


def split_segments(segments: list[str], val_share: float, seed: int) -> dict[str, list[str]]:
    """Whole Sevilla segments into train / val; Cam7 is the deferred test and is refused."""
    if any(s.startswith("Cam7") for s in segments):
        raise ValueError("Cam7 is the deferred test camera and must never be used for training")
    ordered = sorted(segments)
    random.Random(seed).shuffle(ordered)
    n_val = round(len(ordered) * val_share)
    return {"train": sorted(ordered[n_val:]), "val": sorted(ordered[:n_val])}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--us-gun-split", type=Path, default=Path("data/splits/us_gun.json"))
    parser.add_argument("--min-conf", type=float, default=0.30)
    parser.add_argument("--val-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("data/splits/weapon_finetune.json"))
    args = parser.parse_args(argv)
    sevilla = json.loads(args.us_gun_split.read_text())["segments"]["select"]
    train = load_episodes(args.episodes, "rwf2000", ["train"])
    negatives = hard_negative_windows(train, args.min_conf)
    data = {
        "rule": "Sevilla Cam1 + Cam5 split by segments; hard negatives from our RWF train NonFight only; "
        "never Cam7, never the RWF test",
        "seed": args.seed,
        "sevilla_segments": split_segments(sevilla, args.val_share, args.seed),
        "hard_negatives": {"min_conf": args.min_conf, "windows": negatives},
        "weaponshield_weights": "https://raw.githubusercontent.com/Halok600/WeaponShield-AI/"
        "7a0b08ed8961a70fe8ceeeedb1b341584eff66f8/model/best.pt",
        "sources": {
            "sevilla_and_unity": "huggingface.co/datasets/jsalazar/"
            "US-Real-time-gun-detection-in-CCTV-An-open-problem-dataset",
            "rwf2000_mirror": "huggingface.co/datasets/DanJoshua/RWF-2000",
        },
    }
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    clips = len({w["clip"] for w in negatives})
    print(
        f"Sevilla: {len(data['sevilla_segments']['train'])} train / {len(data['sevilla_segments']['val'])} "
        f"val segments; hard negatives: {len(negatives)} windows in {clips} NonFight train clips"
    )


if __name__ == "__main__":
    main()
