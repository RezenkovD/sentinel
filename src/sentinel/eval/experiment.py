"""The test experiment (session 06, step 1): A and the baselines on the test parts — the test is opened once.

    uv run sentinel eval experiment freeze    # docs/experiment_freeze.json: what is frozen, with hashes
    uv run sentinel eval experiment rwf       # RWF-2000 test, 400 clips -> results/experiment/rwf_test.json
    uv run sentinel eval experiment sevilla   # Sevilla Cam7 (weapons) -> results/experiment/sevilla_test.json
    uv run sentinel eval experiment xd        # XD-Violence, 156 videos -> results/experiment/xd_test.json
    uv run sentinel eval experiment report    # docs/bench/experiment_test_<date>.md from the JSON results

Every run first checks the freeze record: the config, A, the model weights and the cached scenes of val and
test must hash exactly as when the test was opened, otherwise nothing is computed. Thresholds come from val
only: A's episode threshold (F1 on val, saved with the model), τ of `baseline_motion` (F1 on val, recomputed
here from the frozen val cache), τ of `baseline_weapon` (the detector's operating point on Sevilla dev).

Confidence intervals: 95 % percentile bootstrap over episodes (clips, segments, videos), 1000 resamples,
seed 2026; a window statistic resamples whole episodes with all their windows.
"""

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

import numpy as np

from sentinel.config import DEFAULT_CONFIG, Config, load_config
from sentinel.eval.cache import OUT_DEFAULT, CachedEpisode, _auc, check_complete, load_episodes
from sentinel.eval.train_a import episode_score

FREEZE = Path("docs/experiment_freeze.json")
OUT = Path("results/experiment")
SEED = 2026
RESAMPLES = 1000
GRACE_S = 5.0  # an alarm within M + 5 s after a violent segment still belongs to it (as in fsm-tune)
CONFIGS = ("baseline_motion", "baseline_weapon", "A")
NAMES_UK = {"baseline_motion": "Рух (базовий)", "baseline_weapon": "Зброя (базовий)", "A": "Агрегатор A"}


# --- statistics -----------------------------------------------------------------------------------------


def bootstrap(
    stat: Callable[[np.ndarray], float | None], n: int, resamples: int = RESAMPLES, seed: int = SEED
) -> list[float] | None:
    """95 % percentile interval of `stat(indices)` over resamples of n episodes with replacement; resamples
    where the statistic is undefined (e.g. one class only) are skipped."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        value = stat(rng.integers(0, n, n))
        if value is not None and not np.isnan(value):
            values.append(value)
    if not values:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predicted = scores >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def roc(labels: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    return {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}


def episode_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """Episode AUC and the metrics at the val threshold, each with its bootstrap interval, and the ROC."""
    point = at_threshold(scores, labels, threshold)

    def auc(idx: np.ndarray) -> float | None:
        return _auc(labels[idx], scores[idx])

    def metric(name: str) -> Callable[[np.ndarray], float]:
        return lambda idx: at_threshold(scores[idx], labels[idx], threshold)[name]

    return {
        "episodes": len(labels),
        "positives": int(labels.sum()),
        "threshold": threshold,
        "auc": _auc(labels, scores),
        "auc_ci": bootstrap(auc, len(labels)),
        **point,
        **{f"{m}_ci": bootstrap(metric(m), len(labels)) for m in ("precision", "recall", "f1")},
        "roc": roc(labels, scores),
    }


def window_auc(per_episode: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Window AUC over episodes' (labels, scores), with the interval resampling whole episodes."""

    def auc(idx: np.ndarray) -> float | None:
        if len(idx) == 0:
            return None
        return _auc(
            np.concatenate([per_episode[i][0] for i in idx]), np.concatenate([per_episode[i][1] for i in idx])
        )

    every = np.arange(len(per_episode))
    labels = np.concatenate([p[0] for p in per_episode]) if per_episode else np.array([])
    scores = np.concatenate([p[1] for p in per_episode]) if per_episode else np.array([])
    return {
        "episodes": len(per_episode),
        "windows": len(labels),
        "positive_windows": int(labels.sum()),
        "auc": auc(every),
        "auc_ci": bootstrap(auc, len(per_episode)),
        "roc": roc(labels, scores) if len(set(labels.tolist())) == 2 else None,
    }


# --- the freeze record ----------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_digest(root: Path, dataset: str, part: str) -> dict:
    """One hash over every cached episode file of a part (names and contents, sorted)."""
    h, files = hashlib.sha256(), 0
    for path in sorted((root / dataset / part).rglob("*.jsonl")):
        h.update(str(path.relative_to(root)).encode())
        h.update(sha256(path).encode())
        files += 1
    return {"files": files, "sha256": h.hexdigest()}


CACHE_PARTS = (
    ("rwf2000", "val"),
    ("rwf2000", "test"),
    ("us_gun", "select"),
    ("us_gun", "test"),
    ("xd", "test"),
)


def motion_tau(config: Config, episodes: Path) -> float:
    from sentinel.eval.baselines import motion_baseline

    val = load_episodes(episodes, "rwf2000", ["val"])
    check_complete(val, "rwf2000", ["val"])
    return motion_baseline(val)["threshold"]


# config sections the test results depend on; `llm`, `aggregation` (B and the cascade — val only), `notify`,
# `source` and `bench` may still change after the test is opened
FROZEN_SECTIONS = (
    "analysis", "models", "weapons", "features", "motion", "context",
    "aggregator_a", "incidents", "baselines",
)  # fmt: skip


def config_digest(config: Config) -> str:
    data = config.model_dump(mode="json", include=set(FROZEN_SECTIONS))
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_state(config_path: Path, config: Config, episodes: Path) -> dict:
    """Everything the test results depend on, hashed."""
    model = Path(config.aggregator_a.model)
    meta = json.loads(model.read_text(encoding="utf-8"))
    w, inc = config.weapons, config.incidents
    perceptions = {}
    for dataset in ("rwf2000", "us_gun", "xd"):
        eps = load_episodes(episodes, dataset)
        perceptions[dataset] = sorted(
            {f"{e.meta['perception_version']}:{e.meta['perception_hash']}" for e in eps}
        )
    return {
        "config": {"sections": list(FROZEN_SECTIONS), "sha256": config_digest(config)},
        "zones_sha256": sha256(Path(config.context.zones)),
        "aggregator_a": {
            "path": str(model),
            "sha256": sha256(model),
            "episode_rule": meta["episode_rule"],
            "episode_threshold": meta["episode_threshold"],
        },
        "weights_sha256": {
            "people": sha256(Path(config.models.people.weights)),
            "pose": sha256(Path(config.models.pose.weights)),
            "weapons": sha256(Path(w.weights)),
        },
        "perceptions": perceptions,
        "caches": {f"{d}/{p}": cache_digest(episodes, d, p) for d, p in CACHE_PARTS},
        "automaton": {
            "theta_high": inc.theta_high,
            "theta_low": inc.theta_low,
            "k": inc.k,
            "n": inc.n,
            "quiet_s": inc.quiet_s,
            "merge_after_close_s": inc.merge_after_close_s,
            "weapon_confirm_windows": inc.weapon_confirm_windows,
        },
        "baselines": {
            "motion_tau": motion_tau(config, episodes),
            "weapon_tau": config.baselines.weapon_conf,
        },
        "weapon_mode": {"mode": w.mode, "imgsz": w.imgsz, "every_n_frames": w.every_n_frames, "conf": w.conf},
    }


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def check_frozen(config_path: Path, config: Config, episodes: Path, freeze: Path = FREEZE) -> dict:
    """The freeze record, after checking that nothing it names has changed since."""
    if not freeze.exists():
        raise SystemExit(f"{freeze} is missing: run `sentinel eval experiment freeze` first")
    record = json.loads(freeze.read_text(encoding="utf-8"))
    now = frozen_state(config_path, config, episodes)
    changed = [k for k, v in record["state"].items() if now.get(k) != v]
    if changed:
        raise SystemExit(f"changed since the test was opened ({record['date']}): {', '.join(changed)}")
    return record


# --- RWF-2000 test --------------------------------------------------------------------------------------


def rwf_scores(episodes: list[CachedEpisode], aggregator, rule: str) -> dict[str, dict]:
    """Per configuration: episode scores and window (labels, scores) of every clip."""
    out = {name: {"episode": [], "windows": []} for name in CONFIGS}
    for e in episodes:
        labels = np.full(len(e.scenes), e.label)
        motion = np.array([s.summary.motion_energy or 0.0 for s in e.scenes])
        weapon = np.array([a.get("weapon_max_conf", 0.0) for a in e.aux])
        risk = np.array([aggregator.assess(s).risk for s in e.scenes])
        for name, scores, episode in (
            ("baseline_motion", motion, float(motion.max())),
            ("baseline_weapon", weapon, float(weapon.max())),
            ("A", risk, episode_score(risk.tolist(), rule)),
        ):
            out[name]["episode"].append(episode)
            out[name]["windows"].append((labels, scores))
    return out


def run_rwf(config: Config, record: dict, episodes: Path) -> dict:
    from sentinel.aggregators.rules import RulesAggregator

    test = load_episodes(episodes, "rwf2000", ["test"])
    check_complete(test, "rwf2000", ["test"])
    state = record["state"]
    a = state["aggregator_a"]
    aggregator = RulesAggregator.load(Path(a["path"]), config)
    thresholds = {
        "baseline_motion": state["baselines"]["motion_tau"],
        "baseline_weapon": state["baselines"]["weapon_tau"],
        "A": a["episode_threshold"],
    }
    scores = rwf_scores(test, aggregator, a["episode_rule"])
    labels = np.array([e.label for e in test])
    rows = {}
    for name in CONFIGS:
        rows[name] = episode_metrics(np.array(scores[name]["episode"]), labels, thresholds[name]) | {
            "window": window_auc(scores[name]["windows"])
        }
    episode = {name: np.array(scores[name]["episode"]) for name in CONFIGS}
    delta = {}
    for base in ("baseline_motion", "baseline_weapon"):  # paired: the same resampled clips for both

        def diff(idx: np.ndarray, base: str = base) -> float | None:
            a_auc, b_auc = _auc(labels[idx], episode["A"][idx]), _auc(labels[idx], episode[base][idx])
            return None if a_auc is None or b_auc is None else a_auc - b_auc

        delta[base] = {"delta_auc": diff(np.arange(len(labels))), "ci": bootstrap(diff, len(labels))}
    return {
        "dataset": "RWF-2000 test",
        "episode_rule_a": a["episode_rule"],
        "configs": rows,
        "a_minus_baseline_auc": delta,
    }


# --- Sevilla Cam7: the weapon scenario ------------------------------------------------------------------


def _onsets(alarms: Sequence[bool]) -> list[int]:
    return [i for i, a in enumerate(alarms) if a and (i == 0 or not alarms[i - 1])]


def weapon_segments(episodes: list[CachedEpisode], alarms_of: Callable[[CachedEpisode], list[bool]]) -> dict:
    """Segment recall (an alarm at or after the first annotated weapon window), times to alarm, alarms before
    any weapon, and alarm onsets in weapon-free windows per hour of weapon-free video."""
    hit, times, early, false_onsets, free_s = [], [], 0, 0, 0.0
    per_segment = []
    for e in episodes:
        alarms = alarms_of(e)
        labelled = [i for i, a in enumerate(e.aux) if a.get("label")]
        first = labelled[0]
        later = [i for i in range(first, len(alarms)) if alarms[i]]
        hit.append(bool(later))
        if later:
            times.append(e.scenes[later[0]].window.t_end - e.scenes[first].window.t_start)
        early += any(alarms[:first])
        free = [i for i, a in enumerate(e.aux) if not a.get("label")]
        seg_false = sum(1 for i in _onsets(alarms) if not e.aux[i].get("label"))
        seg_free = sum(e.scenes[i].window.t_end - e.scenes[i].window.t_start for i in free)
        false_onsets += seg_false
        free_s += seg_free
        per_segment.append(
            {
                "segment": e.episode_id,
                "windows": len(alarms),
                "weapon_windows": len(labelled),
                "alarm": bool(later),
                "time_to_alarm_s": times[-1] if later else None,
                "false_onsets": seg_false,
                "weapon_free_s": seg_free,
            }
        )
    hits = np.array(hit, dtype=float)
    return {
        "segments": len(episodes),
        "recall": float(hits.mean()),
        "recall_ci": bootstrap(lambda idx: float(hits[idx].mean()), len(hits)),
        "times_to_alarm_s": times,
        "median_time_to_alarm_s": float(np.median(times)) if times else None,
        "alarms_before_weapon": early,
        "false_onsets": false_onsets,
        "weapon_free_h": free_s / 3600,
        "false_per_h": false_onsets / (free_s / 3600) if free_s else None,
        "per_segment": per_segment,
    }


def weapon_view(config: Config, crops_root: Path | None) -> Callable[[CachedEpisode], CachedEpisode]:
    """Episodes as the frozen weapon mode sees them: the main cache (frame mode) as is; a crops cache at
    the frozen detector threshold (holders below it let go)."""
    if config.weapons.mode == "frame":
        return lambda e: e
    from sentinel.eval.weapon_rule import at_threshold as hold_at

    return lambda e: hold_at(e, config.weapons.conf)


def run_sevilla(config: Config, record: dict, episodes: Path, crops_root: Path | None) -> dict:
    from sentinel.eval.weapon_rule import weapon_alarms

    root = episodes if config.weapons.mode == "frame" else crops_root
    if root is None:
        raise SystemExit(
            "weapon mode is crops: pass --crops-episodes with Cam7 and RWF test NonFight extracted"
        )
    view = weapon_view(config, root)
    cam7 = [view(e) for e in load_episodes(root, "us_gun", ["test"])]
    check_complete(cam7, "us_gun", ["test"])
    normal = [view(e) for e in load_episodes(root, "rwf2000", ["test"]) if e.label == 0]
    if len(normal) < 200:
        raise SystemExit(f"{root}: {len(normal)} of 200 RWF test NonFight clips")
    windows = config.incidents.weapon_confirm_windows
    tau = record["state"]["baselines"]["weapon_tau"]
    main_cam7 = load_episodes(episodes, "us_gun", ["test"])  # the plain detector baseline: the main cache
    main_normal = [e for e in load_episodes(episodes, "rwf2000", ["test"]) if e.label == 0]

    def rule(e: CachedEpisode) -> list[bool]:
        return weapon_alarms(e.scenes, windows)

    def plain(e: CachedEpisode) -> list[bool]:
        return [a.get("weapon_max_conf", 0.0) >= tau for a in e.aux]

    def share(eps: list[CachedEpisode], alarms_of) -> dict:
        flags = np.array([any(alarms_of(e)) for e in eps], dtype=float)
        return {
            "clips": len(eps),
            "share": float(flags.mean()),
            "share_ci": bootstrap(lambda idx: float(flags[idx].mean()), len(flags)),
        }

    return {
        "dataset": "Sevilla Cam7 (test) + RWF-2000 test NonFight",
        "weapon_mode": record["state"]["weapon_mode"],
        "configs": {
            "rule": weapon_segments(cam7, rule) | {"rwf_nonfight_alarm": share(normal, rule)},
            "baseline_weapon": weapon_segments(main_cam7, plain)
            | {"rwf_nonfight_alarm": share(main_normal, plain)},
        },
    }


# --- XD-Violence ----------------------------------------------------------------------------------------


def segments_of(labels: Sequence[int]) -> list[tuple[int, int]]:
    """Maximal runs of violent windows as (first, last) indices."""
    runs, start = [], None
    for i, v in enumerate(labels):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(labels) - 1))
    return runs


def automaton_alarms(scenes, risks, cfg) -> list[bool]:
    """Window by window: is an incident open (ALARM or COOLDOWN) after this window?"""
    from sentinel.incidents import IncidentManager

    manager = IncidentManager(cfg, "cam-1", lambda *_, **__: None, "experiment")
    flags = []
    for scene, risk in zip(scenes, risks, strict=True):
        manager.on_window(scene.window.t_end, risk, scene)
        flags.append(manager.current is not None)
    return flags


def merge_segments(segs: list[tuple[int, int]], scenes, gap_s: float) -> list[tuple[int, int]]:
    """Violent runs separated by less than `gap_s` seconds of normal windows are one segment: the automaton
    keeps an incident open over such a gap (M + 5 s), so for the operator they are one event."""
    merged: list[tuple[int, int]] = []
    for first, last in segs:
        if merged and scenes[first].window.t_start - scenes[merged[-1][1]].window.t_end < gap_s:
            merged[-1] = (merged[-1][0], last)
        else:
            merged.append((first, last))
    return merged


def alarm_timing(episodes: list[CachedEpisode], alarms: list[list[bool]], quiet_s: float) -> dict:
    """Violent segments (runs of violent windows, merged across gaps shorter than M + 5 s): the share with an
    alarm, the share where the alarm was already on when the segment began (raised during the lead-in), and
    the delay from the segment's start to its first alarm window for the others. A false alarm is an alarm
    run that overlaps no violent window and does not start within M + 5 s after a segment; the rate is per
    hour of normal windows. Intervals resample whole videos."""
    tail = quiet_s + GRACE_S
    per_video = []
    for e, flags in zip(episodes, alarms, strict=True):
        labels = [int(a.get("label", 0)) for a in e.aux]
        segs = merge_segments(segments_of(labels), e.scenes, tail)
        detected, already, delays = [], [], []
        for first, last in segs:
            hits = [i for i in range(first, last + 1) if flags[i]]
            on = first > 0 and flags[first - 1]
            detected.append(bool(hits))
            already.append(on)
            if hits and not on:
                delays.append(e.scenes[hits[0]].window.t_end - e.scenes[first].window.t_start)
        false = 0
        for i in _onsets(flags):
            j = i
            while j + 1 < len(flags) and flags[j + 1]:
                j += 1
            t = e.scenes[i].window.t_end
            ends = [e.scenes[last].window.t_end for _, last in segs]
            false += not any(labels[i : j + 1]) and not any(end < t <= end + tail for end in ends)
        normal_s = sum(
            s.window.t_end - s.window.t_start for s, v in zip(e.scenes, labels, strict=True) if not v
        )
        per_video.append((detected, already, delays, false, normal_s))

    def pooled(k: int) -> Callable[[np.ndarray], list]:
        return lambda idx: [x for i in idx for x in per_video[i][k]]

    def share(k: int) -> Callable[[np.ndarray], float | None]:
        return lambda idx: float(np.mean(v)) if (v := pooled(k)(idx)) else None

    def med_delay(idx: np.ndarray) -> float | None:
        values = pooled(2)(idx)
        return float(np.median(values)) if values else None

    def fa_rate(idx: np.ndarray) -> float | None:
        hours = sum(per_video[i][4] for i in idx) / 3600
        return sum(per_video[i][3] for i in idx) / hours if hours else None

    every = np.arange(len(per_video))
    return {
        "videos": len(per_video),
        "segments": len(pooled(0)(every)),
        "detected": share(0)(every),
        "detected_ci": bootstrap(share(0), len(per_video)),
        "already_on": share(1)(every),
        "delays_s": pooled(2)(every),
        "median_delay_s": med_delay(every),
        "median_delay_ci": bootstrap(med_delay, len(per_video)),
        "false_onsets": sum(p[3] for p in per_video),
        "normal_h": sum(p[4] for p in per_video) / 3600,
        "false_per_h": fa_rate(every),
        "false_per_h_ci": bootstrap(fa_rate, len(per_video)),
    }


def run_xd(config: Config, record: dict, episodes: Path, crops_root: Path | None) -> dict:
    from sentinel.aggregators.rules import RulesAggregator

    videos = load_episodes(episodes, "xd")
    check_complete(videos, "xd", ["test"])
    strata = {
        v["name"]: v["stratum"] for v in json.loads(Path("data/splits/xd_subset.json").read_text())["videos"]
    }
    state = record["state"]
    aggregator = RulesAggregator.load(Path(state["aggregator_a"]["path"]), config)
    if config.weapons.mode == "crops":  # weapons (for the automaton) from the crops cache, at the frozen τ
        if crops_root is None:
            raise SystemExit("weapon mode is crops: pass --crops-episodes with XD extracted")
        view = weapon_view(config, crops_root)
        by_id = {e.episode_id: view(e) for e in load_episodes(crops_root, "xd")}
        weapon_scenes = [by_id[v.episode_id].scenes for v in videos]
    else:
        weapon_scenes = [v.scenes for v in videos]
    tau = {
        "baseline_motion": state["baselines"]["motion_tau"],
        "baseline_weapon": state["baselines"]["weapon_tau"],
    }
    per = {name: [] for name in CONFIGS}
    alarms = {name: [] for name in CONFIGS}
    for v, w_scenes in zip(videos, weapon_scenes, strict=True):
        labels = np.array([int(a.get("label", 0)) for a in v.aux])
        motion = np.array([s.summary.motion_energy or 0.0 for s in v.scenes])
        weapon = np.array([a.get("weapon_max_conf", 0.0) for a in v.aux])
        results = [aggregator.assess(s) for s in v.scenes]
        risk = np.array([r.risk for r in results])
        for name, scores in (("baseline_motion", motion), ("baseline_weapon", weapon), ("A", risk)):
            per[name].append((labels, scores))
        alarms["baseline_motion"].append((motion >= tau["baseline_motion"]).tolist())
        alarms["baseline_weapon"].append((weapon >= tau["baseline_weapon"]).tolist())
        alarms["A"].append(automaton_alarms(w_scenes, results, config.incidents))
    groups = {"all": None, "B1": "B1", "B2": "B2", "normal": "A"}  # B1 / B2 with the normal videos
    rows = {}
    for name in CONFIGS:
        rows[name] = {"window": {}, "alarms": {}}
        for group, stratum in groups.items():
            keep = [
                i
                for i, v in enumerate(videos)
                if stratum is None or strata[v.meta["source"]] in (stratum, "A")
            ]
            rows[name]["window"][group] = window_auc([per[name][i] for i in keep])
            rows[name]["alarms"][group] = alarm_timing(
                [videos[i] for i in keep], [alarms[name][i] for i in keep], config.incidents.quiet_s
            )
    return {
        "dataset": "XD-Violence test subset",
        "videos": len(videos),
        "strata": {s: sum(1 for v in videos if strata[v.meta["source"]] == s) for s in ("B1", "B2", "A")},
        "alarm_rule": {
            "A": "incident automaton (frozen θ, k of n, M; weapon rule)",
            "baselines": "window score ≥ τ, no automaton",
        },
        "configs": rows,
    }


# --- report ---------------------------------------------------------------------------------------------


def _n(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}".replace(".", ",")


def _ci(value, ci, digits: int = 3) -> str:
    if value is None:
        return "—"
    if not ci:
        return _n(value, digits)
    return f"{_n(value, digits)} [{_n(ci[0], digits)}; {_n(ci[1], digits)}]"


def rwf_table(result: dict) -> tuple[list[str], list[list[str]]]:
    head = [
        "Конфігурація",
        "AUC епізодів",
        "Поріг (val)",
        "F1",
        "Точність",
        "Повнота",
        "TP / FP / FN / TN",
        "AUC вікон",
    ]
    rows = []
    for name in CONFIGS:
        r = result["configs"][name]
        c = r["confusion"]
        rows.append(
            [
                NAMES_UK[name],
                _ci(r["auc"], r["auc_ci"]),
                _n(r["threshold"], 2),
                _ci(r["f1"], r["f1_ci"]),
                _ci(r["precision"], r["precision_ci"]),
                _ci(r["recall"], r["recall_ci"]),
                f"{c['tp']} / {c['fp']} / {c['fn']} / {c['tn']}",
                _ci(r["window"]["auc"], r["window"]["auc_ci"]),
            ]
        )
    for base, d in result.get("a_minus_baseline_auc", {}).items():
        rows.append([f"A − {NAMES_UK[base].lower()}", _ci(d["delta_auc"], d["ci"]), *["—"] * (len(head) - 2)])
    return head, rows


def sevilla_table(result: dict) -> tuple[list[str], list[list[str]]]:
    head = [
        "Метод",
        "Повнота сегментів",
        "Медіана часу до тривоги, с",
        "Тривог до появи зброї",
        "Хибних тривог на год без зброї (Cam7)",
        "NonFight RWF test з тривогою",
    ]
    m = result["weapon_mode"]
    kind = "вирізки " if m["mode"] == "crops" else "кадр "
    mode = f"{kind}{m['imgsz']}:{m['every_n_frames']}, поріг {_n(m['conf'], 2)}"
    names = {"rule": f"Правило «зброя в руці» ({mode})", "baseline_weapon": "Зброя (базовий, одне вікно)"}
    rows = []
    for key in ("rule", "baseline_weapon"):
        r = result["configs"][key]
        rows.append(
            [
                names[key],
                f"{_ci(r['recall'], r['recall_ci'], 2)} "
                f"({sum(s['alarm'] for s in r['per_segment'])} з {r['segments']})",
                _n(r["median_time_to_alarm_s"], 1),
                str(r["alarms_before_weapon"]),
                f"{_n(r['false_per_h'], 1)} ({r['false_onsets']} за {_n(r['weapon_free_h'], 2)} год)",
                f"{_ci(r['rwf_nonfight_alarm']['share'], r['rwf_nonfight_alarm']['share_ci'])} "
                f"з {r['rwf_nonfight_alarm']['clips']}",
            ]
        )
    return head, rows


def xd_tables(result: dict) -> dict[str, tuple[list[str], list[list[str]]]]:
    auc_head = ["Конфігурація", "AUC вікон, усі", "B1 (бійки) + норма", "B2 (стрілянина) + норма"]
    auc_rows = [
        [NAMES_UK[n]]
        + [
            _ci(result["configs"][n]["window"][g]["auc"], result["configs"][n]["window"][g]["auc_ci"])
            for g in ("all", "B1", "B2")
        ]
        for n in CONFIGS
    ]
    alarm_head = [
        "Конфігурація",
        "Група",
        "Сегментів з тривогою",
        "Тривога вже була на початку",
        "Медіана затримки, с",
        "Хибних тривог на год норми",
    ]
    alarm_rows = []
    for n in CONFIGS:
        for g, label in (
            ("all", "усі 156"),
            ("B1", "B1 + норма"),
            ("B2", "B2 + норма"),
            ("normal", "лише норма (31)"),
        ):
            r = result["configs"][n]["alarms"][g]
            segments = r["segments"]
            alarm_rows.append(
                [
                    NAMES_UK[n],
                    label,
                    f"{_ci(r['detected'], r['detected_ci'], 2)} з {segments}" if segments else "—",
                    _n(r["already_on"], 2) if segments else "—",
                    f"{_ci(r['median_delay_s'], r['median_delay_ci'], 1)} (n = {len(r['delays_s'])})"
                    if segments
                    else "—",
                    f"{_ci(r['false_per_h'], r['false_per_h_ci'], 1)} "
                    f"({r['false_onsets']} за {_n(r['normal_h'], 2)} год)",
                ]
            )
    return {"xd_auc": (auc_head, auc_rows), "xd_alarms": (alarm_head, alarm_rows)}


TITLES = {
    "rwf": "Таблиця. Виявлення бійок на RWF-2000 (тест, 400 кліпів)",
    "sevilla": "Таблиця. Виявлення зброї в руці: Севілья Cam7 (тест)",
    "xd_auc": "Таблиця. Ризик вікон на XD-Violence (156 відео)",
    "xd_alarms": "Таблиця. Тривоги на XD-Violence: затримка й хибні тривоги",
}


def tables(out: Path = OUT) -> dict[str, tuple[str, list[str], list[list[str]]]]:
    """Every experiment table available so far: name -> (Ukrainian title, header, rows)."""
    found = {}
    if (out / "rwf_test.json").exists():
        found["rwf"] = rwf_table(json.loads((out / "rwf_test.json").read_text()))
    if (out / "sevilla_test.json").exists():
        found["sevilla"] = sevilla_table(json.loads((out / "sevilla_test.json").read_text()))
    if (out / "xd_test.json").exists():
        found |= xd_tables(json.loads((out / "xd_test.json").read_text()))
    return {k: (TITLES[k], *v) for k, v in found.items()}


def markdown_table(head: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
        *["| " + " | ".join(r) + " |" for r in rows],
    ]


def write_report(record: dict, out: Path, path: Path) -> None:
    s = record["state"]
    lines = [
        f"# Експеримент на тесті: A і базові методи — {date.today().isoformat()}",
        "",
        f"Тест відкрито {record['date']} (коміт `{record['commit']}`); заморожене — "
        "[`experiment_freeze.json`](../experiment_freeze.json),"
        " кожен прогін перевіряє хеші. Пороги — лише з val: A — поріг F1 епізодів на val "
        f"({_n(s['aggregator_a']['episode_threshold'], 2)}, оцінка епізоду — "
        f"`{s['aggregator_a']['episode_rule']}` ризиків вікон); "
        f"рух — τ {_n(s['baselines']['motion_tau'], 2)} (F1 на val, найбільша енергія руху вікна); "
        f"зброя — τ {_n(s['baselines']['weapon_tau'], 2)} (робоча точка детектора, Севілья dev). "
        f"Автомат: θ_high {_n(s['automaton']['theta_high'], 2)}, θ_low {_n(s['automaton']['theta_low'], 2)}, "
        f"k {s['automaton']['k']} з n {s['automaton']['n']}, M {_n(s['automaton']['quiet_s'], 0)} с.",
        "",
        f"У дужках — 95 % довірчий інтервал: бутстреп за епізодами, {RESAMPLES} повторів, seed {SEED}.",
        "",
    ]
    for name, (title, head, rows) in tables(out).items():
        lines += [f"## {title}", "", *markdown_table(head, rows), ""]
        lines += NOTES.get(name, [])
    if path.exists() and MANUAL in (old := path.read_text(encoding="utf-8")):  # keep the written conclusions
        lines += [old[old.index(MANUAL) :].rstrip("\n")]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


MANUAL = "## Висновки (додано вручну)"
NOTES = {
    "rwf": [
        "Оцінка епізоду: рух — найбільша енергія руху вікна; зброя — найвища оцінка детектора у вікні; "
        "A — середній ризик вікон. AUC вікон — мітка кліпу для кожного його вікна. Рядки «A − …» — різниця "
        "AUC епізодів, інтервал — парний бутстреп (ті самі кліпи для обох методів).",
        "",
    ],
    "sevilla": [
        "Сегмент знайдено, якщо тривога є у вікні з анотованою зброєю або пізніше; час — від початку першого "
        "вікна зі зброєю до кінця вікна тривоги. Хибна тривога — початок тривоги у вікні без анотованої "
        "зброї. "
        "У Cam7 немає сегментів без зброї зовсім, тому окремо — частка спокійних кліпів RWF test (без зброї) "
        "з тривогою.",
        "",
    ],
    "xd_alarms": [
        "A — тривога автомата інцидентів (інцидент відкритий); базові методи — вікно з оцінкою ≥ τ, "
        "без автомата. Сегмент — неперервна послідовність вікон із насильством за покадровою розміткою; "
        "сегменти, між якими менше M + 5 с, злито в один. «Тривога вже була» — тривога горіла у вікні перед "
        "початком сегмента (її підняв вступ до сцени); затримка — від початку сегмента до першого вікна "
        "тривоги, лише для решти сегментів (n). Хибна тривога — неперервна тривога, що не перетинається з "
        "жодним вікном насильства й почалася не в межах M + 5 с після сегмента; частота — на годину вікон "
        "без насильства.",
        "",
    ],
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("freeze", "rwf", "sevilla", "xd", "report"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episodes", type=Path, default=OUT_DEFAULT)
    parser.add_argument(
        "--crops-episodes", type=Path, help="crops-mode cache (weapon parts), if the mode is crops"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.step == "freeze":
        if FREEZE.exists():
            raise SystemExit(f"{FREEZE} exists: the test is opened once")
        record = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
            "state": frozen_state(args.config, config, args.episodes),
        }
        FREEZE.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=1, ensure_ascii=False))
        return
    record = check_frozen(args.config, config, args.episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.step == "report":
        path = Path(f"docs/bench/experiment_test_{date.today().isoformat()}.md")
        write_report(record, args.out, path)
        print(path.read_text(encoding="utf-8"))
        return
    if args.step == "rwf":
        result = run_rwf(config, record, args.episodes)
    elif args.step == "sevilla":
        result = run_sevilla(config, record, args.episodes, args.crops_episodes)
    else:
        result = run_xd(config, record, args.episodes, args.crops_episodes)
    result |= {
        "computed": datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "freeze": record["date"],
    }
    name = {"rwf": "rwf_test", "sevilla": "sevilla_test", "xd": "xd_test"}[args.step]
    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    for key, (title, head, rows) in tables(args.out).items():
        if key.startswith(args.step[:2]) or (args.step == "xd" and key.startswith("xd")):
            print(title, *markdown_table(head, rows), sep="\n")


if __name__ == "__main__":
    main()
