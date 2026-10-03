"""Aggregator A: logistic model over scene features (SPEC §3–4).

`scene_to_features` is the only way from a scene description to A's inputs: a fixed order of named features.
A null feature takes its neutral value from the config (a calm scene); indicators say that the wrist
acceleration was missing ("no data about the hands" is not "calm hands") and that nobody was detected.

Weapons are not features of A and do not touch its risk (decisions 30.09 and 01.10): the weapon alarm is
the incident automaton's rule (`holds` confirmed in consecutive windows), the same for A, B and the cascade.
An empty frame where nothing moves (motion_area below `quiet_motion_area`, τ_quiet) is zero without
computation (SPEC §5); an empty frame with motion is scored from the pixel-motion features by a model fitted
on all windows (`empty_windows: score`), while a model fitted on windows with people only keeps every empty
frame at zero (`empty_windows: zero`).
"""

import json
from pathlib import Path

import numpy as np

from sentinel.config import Config
from sentinel.schema import SCHEMA_VERSION, RiskResult, SceneDescription

FEATURE_NAMES = (
    "n_persons",
    "crowd_motion",
    "max_speed",
    "pair_max_speed",
    "max_limb_speed",
    "max_limb_accel",
    "max_pose_motion",
    "max_person_motion",
    "motion_energy",
    "motion_area",
    "min_distance_h",
    "max_approach_speed",
    "max_contact_s",
    "n_close_pairs",
    "zone_prior",
    "limb_accel_missing",
    "no_persons",
)
FEATURE_NAMES_UK = {  # for tables in reports and the thesis
    "n_persons": "кількість людей",
    "crowd_motion": "середня швидкість людей",
    "max_speed": "найбільша швидкість людини",
    "pair_max_speed": "найбільша швидкість людини, що має пару",
    "max_limb_speed": "швидкість рук відносно тулуба",
    "max_limb_accel": "прискорення рук відносно тулуба",
    "max_pose_motion": "рух тіла відносно тулуба",
    "max_person_motion": "рух пікселів у рамці людини",
    "motion_energy": "енергія руху в кадрі",
    "motion_area": "частка кадру з рухом",
    "min_distance_h": "найменша відстань між людьми",
    "max_approach_speed": "швидкість зближення",
    "max_contact_s": "тривалість близького контакту",
    "n_close_pairs": "кількість пар упритул",
    "zone_prior": "апріорний ризик зони",
    "limb_accel_missing": "немає даних про руки",
    "no_persons": "людей у кадрі не знайдено",
}
_REASON_GROUP = {  # one phrase per group in an explanation
    "crowd_motion": "motion",
    "max_speed": "motion",
    "pair_max_speed": "motion",
    "max_limb_speed": "limbs",
    "max_limb_accel": "limbs",
    "max_pose_motion": "limbs",
    "max_person_motion": "pixels",
    "motion_energy": "pixels",
    "motion_area": "pixels",
}


def quiet_empty(scene: SceneDescription, quiet_motion_area: float) -> bool:
    """Nobody detected and nothing moves: the one case A answers with zero without computing (SPEC §5)."""
    area = scene.summary.motion_area
    return scene.summary.n_persons == 0 and (area is None or area < quiet_motion_area)


def scene_to_features(scene: SceneDescription, config: Config) -> np.ndarray:
    neutral = config.aggregator_a.neutral
    contact = config.features.contact_distance_h
    persons, pairs = scene.persons, scene.interactions

    def worst(values: list[float | None], name: str, pick=max) -> float:
        known = [v for v in values if v is not None]
        return float(pick(known)) if known else neutral[name]

    def value(v: float | None, name: str) -> float:
        return neutral[name] if v is None else float(v)

    close = [i for i in pairs if i.distance_h < contact]  # pairs in contact
    paired = {i for pair in pairs for i in pair.pair}  # people with someone closer than the pair limit
    prior = config.aggregator_a.zone_prior
    zone = scene.zone
    summary = scene.summary
    x = {
        "n_persons": float(summary.n_persons),
        "crowd_motion": value(summary.crowd_motion, "crowd_motion"),
        "max_speed": worst([p.speed for p in persons], "max_speed"),
        "pair_max_speed": worst([p.speed for p in persons if p.id in paired], "pair_max_speed"),
        "max_limb_speed": worst([p.limb_speed for p in persons], "max_limb_speed"),
        "max_limb_accel": worst([p.limb_accel for p in persons], "max_limb_accel"),
        "max_pose_motion": worst([p.pose_motion for p in persons], "max_pose_motion"),
        "max_person_motion": worst([p.motion for p in persons], "max_person_motion"),
        "motion_energy": value(summary.motion_energy, "motion_energy"),
        "motion_area": value(summary.motion_area, "motion_area"),
        "min_distance_h": worst([i.distance_h for i in pairs], "min_distance_h", pick=min),
        "max_approach_speed": worst([i.approach_speed for i in pairs], "max_approach_speed"),
        "max_contact_s": worst([i.duration_s for i in pairs], "max_contact_s"),
        "n_close_pairs": float(len(close)),
        "zone_prior": prior.hours.get(zone.hours, 0.0)
        + prior.density.get(zone.expected_density, 0.0)
        + prior.type.get(zone.type, 0.0),
        "limb_accel_missing": float(bool(persons) and all(p.limb_accel is None for p in persons)),
        "no_persons": float(summary.n_persons == 0),
    }
    return np.array([x[name] for name in FEATURE_NAMES], dtype=float)


def _reason(name: str, x: dict[str, float], scene: SceneDescription) -> str | None:
    """A plain operator phrase for a feature that raised the risk (no units; numbers stay in the journal)."""
    phrases = {
        "max_limb_accel": "різкі рухи рук",
        "max_limb_speed": "швидкі рухи рук",
        "max_pose_motion": "активні рухи тіла",
        "pair_max_speed": "швидкий рух поруч з іншою людиною",
        "max_person_motion": "інтенсивний рух людини",
        "motion_energy": "багато руху в кадрі",
        "motion_area": "рух на великій частині кадру",
        "min_distance_h": "люди дуже близько одне до одного",
        "max_approach_speed": "люди швидко зближуються",
        "max_contact_s": "тривалий близький контакт",
        "n_close_pairs": "кілька пар людей упритул",
        "crowd_motion": "швидкий рух людей",
        "max_speed": "швидкий рух людей",
        "n_persons": "багато людей",
        "zone_prior": "зона або час підвищеного ризику",
    }
    return phrases.get(name)


def make_aggregator(config: Config) -> "RulesAggregator":
    """The trained A named in the config, or the hand-set weights when none is set.

    A model trained on scenes of another perception (e.g. the OpenVINO cache while `sentinel run` uses torch)
    still loads, with a warning: its inputs may differ slightly from the ones it was fitted on.
    """
    import sys

    model = config.aggregator_a.model
    if model is None:
        return RulesAggregator.from_config(config)
    trained_on = json.loads(Path(model).read_text(encoding="utf-8")).get("perception_hash")
    if trained_on is not None:
        from sentinel.pipeline import perception_hash

        running = perception_hash(config)
        if running != trained_on:
            print(
                f"warning: A ({model}) was trained on perception {trained_on}, this run's perception is "
                f"{running} (backend, weights or settings differ)",
                file=sys.stderr,
            )
    return RulesAggregator.load(model, config)


class RulesAggregator:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        mean: np.ndarray,
        scale: np.ndarray,
        clip: float,
        config: Config,
        empty_windows: str = "zero",  # "score": fitted on all windows, empty frames with motion are scored
    ) -> None:
        if empty_windows not in ("zero", "score"):
            raise ValueError(f"empty_windows must be 'zero' or 'score', got {empty_windows!r}")
        self.weights, self.bias, self.mean, self.scale, self.clip = weights, bias, mean, scale, clip
        self.config = config
        self.empty_windows = empty_windows
        self._neutral_z = self._standardise(np.array([config.aggregator_a.neutral[n] for n in FEATURE_NAMES]))

    @classmethod
    def from_config(cls, config: Config) -> "RulesAggregator":
        a = config.aggregator_a
        for table in (a.weights, a.scale, a.neutral):
            if set(table) != set(FEATURE_NAMES):
                raise ValueError(
                    f"aggregator_a tables must list exactly {FEATURE_NAMES}, got {sorted(table)}"
                )
        weights = np.array([a.weights[n] for n in FEATURE_NAMES])
        scale = np.array([a.scale[n] for n in FEATURE_NAMES])
        return cls(weights, a.bias, np.zeros(len(FEATURE_NAMES)), scale, a.clip, config)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / self.scale, -self.clip, self.clip)

    def assess(self, scene: SceneDescription) -> RiskResult:
        thresholds = self.config.incidents
        if scene.summary.n_persons == 0 and (
            self.empty_windows == "zero" or quiet_empty(scene, self.config.aggregator_a.quiet_motion_area)
        ):
            return RiskResult(risk=0.0, level="normal", reason_uk="У кадрі нікого немає")
        x = scene_to_features(scene, self.config)
        z = self._standardise(x)
        risk = float(1.0 / (1.0 + np.exp(-(self.bias + self.weights @ z))))
        named = dict(zip(FEATURE_NAMES, x, strict=True))
        level = (
            "alarm"
            if risk >= thresholds.theta_high
            else "attention"
            if risk >= thresholds.theta_low
            else "normal"
        )
        return RiskResult(risk=risk, level=level, reason_uk=self._explain(z, named, scene))

    def _explain(self, z: np.ndarray, x: dict[str, float], scene: SceneDescription) -> str:
        """Two or three largest contributions against the neutral (calm) scene, as Ukrainian phrases."""
        contributions = self.weights * (z - self._neutral_z)
        floor = self.config.aggregator_a.reason_min_contribution
        phrases: list[str] = []
        groups: set[str] = set()
        for i in np.argsort(-contributions):
            name = FEATURE_NAMES[i]
            if contributions[i] < floor or len(phrases) == 3:
                break
            group = _REASON_GROUP.get(name, name)
            if group in groups or (phrase := _reason(name, x, scene)) is None:
                continue
            groups.add(group)
            phrases.append(phrase)
        if not phrases:
            return "Ознак небезпеки немає"
        text = "; ".join(phrases)
        return text[0].upper() + text[1:]

    def save(self, path: Path, extra: dict | None = None) -> None:
        """Weights with the feature list and schema version; `extra` adds metadata (training, thresholds)."""
        data = {
            "features": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip,
            "empty_windows": self.empty_windows,
            "schema_version": SCHEMA_VERSION,
            # what turns a scene into the inputs: pinned, so later config edits cannot shift a trained model
            "input_settings": {
                "neutral": dict(self.config.aggregator_a.neutral),
                "zone_prior": self.config.aggregator_a.zone_prior.model_dump(),
                "contact_distance_h": self.config.features.contact_distance_h,
                "quiet_motion_area": self.config.aggregator_a.quiet_motion_area,
            },
        } | (extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, config: Config) -> "RulesAggregator":
        """Load saved weights; refuse ones made for other features, another schema or another perception."""
        from sentinel.pipeline import PERCEPTION_VERSION, compatible_perceptions

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["features"]) != FEATURE_NAMES:
            raise ValueError(f"{path} was saved for features {data['features']}, expected {FEATURE_NAMES}")
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(
                f"{path} was trained on schema v{data['schema_version']}, this is v{SCHEMA_VERSION}"
            )
        trained_on = data.get("perception_version", PERCEPTION_VERSION)
        if not compatible_perceptions({trained_on, PERCEPTION_VERSION}):
            raise ValueError(
                f"{path} was trained on perception {data['perception_version']}, this is {PERCEPTION_VERSION}"
            )
        arrays = {k: np.array(data[k], dtype=float) for k in ("weights", "mean", "scale")}
        pinned = data.get("input_settings")
        if pinned is not None:
            a = config.aggregator_a
            config = config.model_copy(
                update={
                    "aggregator_a": a.model_copy(
                        update={
                            "neutral": pinned["neutral"],
                            "zone_prior": a.zone_prior.model_validate(pinned["zone_prior"]),
                            "quiet_motion_area": pinned.get("quiet_motion_area", a.quiet_motion_area),
                        }
                    ),
                    "features": config.features.model_copy(
                        update={"contact_distance_h": pinned["contact_distance_h"]}
                    ),
                }
            )
        return cls(
            arrays["weights"],
            float(data["bias"]),
            arrays["mean"],
            arrays["scale"],
            data["clip"],
            config,
            data.get("empty_windows", "zero"),
        )
