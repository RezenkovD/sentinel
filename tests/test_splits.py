from collections import Counter

from eval.build_episodes import Clip, source_group, split_rwf2000


def test_source_group_drops_the_segment_index():
    assert source_group("-1l5631l3fg_2") == "-1l5631l3fg"
    assert source_group("0H2s9UJcNJ0_0") == "0H2s9UJcNJ0"
    assert source_group("rfthfd") == "rfthfd"
    # renamed clips (names too long for ext4) are grouped by their original name
    assert (
        source_group("renamed-abc_urlgot_7", {"renamed-abc_urlgot_7": "长名字_urlgot_7"}) == "长名字_urlgot"
    )


def _clips() -> list[Clip]:
    clips = []
    for v in range(40):  # 40 source videos of 5 clips each; every 4th video also has NonFight parts
        for i in range(5):
            label = "NonFight" if (v % 4 == 0 and i >= 3) or v >= 30 else "Fight"
            clips.append(Clip(f"train/{label}/vid{v}_{i}.avi", label, "train", f"vid{v}"))
    clips += [Clip(f"val/Fight/t{i}_0.avi", "Fight", "val", f"t{i}") for i in range(10)]
    return clips


def test_original_val_becomes_the_test_split_untouched():
    split = split_rwf2000(_clips(), val_share=0.2, seed=7)
    assert split["test"] == [f"val/Fight/t{i}_0.avi" for i in range(10)]
    assert not set(split["test"]) & (set(split["train"]) | set(split["val"]))


def test_source_videos_never_straddle_train_and_val():
    clips = _clips()
    split = split_rwf2000(clips, val_share=0.2, seed=7)
    group = {c.path: c.group for c in clips}
    assert not {group[p] for p in split["train"]} & {group[p] for p in split["val"]}
    assert len(split["train"]) + len(split["val"]) == 200


def test_val_holds_a_fifth_of_each_class_up_to_one_source_video():
    # Whole source videos move together, so val never overshoots the target and misses it by < one video.
    clips = _clips()
    split = split_rwf2000(clips, val_share=0.2, seed=7)
    label = {c.path: c.label for c in clips}
    total = Counter(c.label for c in clips if c.original_split == "train")
    in_val = Counter(label[p] for p in split["val"])
    largest_video = 5
    for cls in ("Fight", "NonFight"):
        target = round(0.2 * total[cls])
        assert target - largest_video < in_val[cls] <= target, (cls, in_val[cls], target)


def test_split_is_reproducible_for_a_seed_and_differs_between_seeds():
    clips = _clips()
    assert split_rwf2000(clips, 0.2, seed=7) == split_rwf2000(list(reversed(clips)), 0.2, seed=7)
    assert split_rwf2000(clips, 0.2, seed=7)["val"] != split_rwf2000(clips, 0.2, seed=8)["val"]
