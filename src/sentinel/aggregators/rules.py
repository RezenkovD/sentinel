"""Aggregator A: logistic model over scene features plus the "weapon in hand" override rule (SPEC §3–4).

`scene_to_features` is the only way from a scene description to A's inputs: a fixed order of named features.
A null feature takes its neutral value from the config (a calm scene); for the wrist acceleration an extra
indicator says the value was missing, because "no data about the hands" is not the same as "calm hands".

v0 uses hand-set weights from the config. The same object is ready for a fitted logistic regression
(`fit` / `save` / `load`): features are standardised as z = clip((x − mean) / scale, ±clip) and
risk = sigmoid(bias + w · z); hand-set weights use mean = 0 and the configured scale.
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
    "facing_close",
    "max_contact_s",
    "n_close_pairs",
    "weapon_visible",
    "weapon_held",
    "zone_prior",
    "limb_accel_missing",
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
    "facing_close": "обличчям одне до одного впритул",
    "max_contact_s": "тривалість близького контакту",
    "n_close_pairs": "кількість пар упритул",
    "weapon_visible": "зброя в кадрі",
    "weapon_held": "зброя в руці",
    "zone_prior": "апріорний ризик зони",
    "limb_accel_missing": "немає даних про руки",
}
_WEAPON_UK = {"gun": "вогнепальна зброя", "knife": "ніж"}
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


def _num(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


def scene_to_features(scene: SceneDescription, config: Config) -> np.ndarray:
    neutral = config.aggregator_a.neutral
    contact = config.features.contact_distance_h
    persons, pairs = scene.persons, scene.interactions

    def worst(values: list[float | None], name: str, pick=max) -> float:
        known = [v for v in values if v is not None]
        return float(pick(known)) if known else neutral[name]

    def value(v: float | None, name: str) -> float:
        return neutral[name] if v is None else float(v)

    close = [i for i in pairs if i.distance_h < contact]
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
        "facing_close": float(any(i.facing is True for i in close)),
        "max_contact_s": worst([i.duration_s for i in pairs], "max_contact_s"),
        "n_close_pairs": float(len(close)),
        "weapon_visible": float(summary.weapon_visible),
        "weapon_held": float(any(p.holds is not None for p in persons)),
        "zone_prior": prior.hours.get(zone.hours, 0.0)
        + prior.density.get(zone.expected_density, 0.0)
        + prior.type.get(zone.type, 0.0),
        "limb_accel_missing": float(bool(persons) and all(p.limb_accel is None for p in persons)),
    }
    return np.array([x[name] for name in FEATURE_NAMES], dtype=float)


def _reason(name: str, x: dict[str, float], scene: SceneDescription) -> str | None:
    if name == "max_limb_accel":
        return "різкі рухи рук"
    if name == "max_limb_speed":
        return f"швидкі рухи рук ({_num(x[name])} зросту/с)"
    if name == "max_pose_motion":
        return f"активні рухи тіла ({_num(x[name])} зросту/с)"
    if name == "pair_max_speed":
        return f"швидкий рух поруч з іншою людиною ({_num(x[name])} зросту/с)"
    if name == "max_person_motion":
        return "інтенсивний рух у рамці людини"
    if name == "motion_energy":
        return "багато руху в кадрі"
    if name == "motion_area":
        return f"рух на {_num(x[name])} % кадру"
    if name == "min_distance_h":
        return f"двоє людей зблизилися до {_num(x[name])} зросту"
    if name == "max_approach_speed":
        return f"швидке зближення ({_num(x[name])} зросту/с)"
    if name == "facing_close":
        return "люди впритул обличчям одне до одного"
    if name == "max_contact_s":
        return f"тривалий близький контакт ({_num(x[name])} с)"
    if name == "n_close_pairs":
        return f"кілька пар упритул ({int(x[name])})"
    if name == "weapon_held":
        kinds = sorted({p.holds for p in scene.persons if p.holds})
        return "зброя в руці: " + ", ".join(_WEAPON_UK[k] for k in kinds)
    if name == "weapon_visible":
        return "у кадрі видно зброю"
    if name in ("crowd_motion", "max_speed"):
        return f"швидкий рух ({_num(x[name])} зросту/с)"
    if name == "n_persons":
        return f"багато людей ({int(x[name])})"
    if name == "zone_prior":
        return "зона або час підвищеного ризику"
    return None


def make_aggregator(config: Config) -> "RulesAggregator":
    """The trained A named in the config, or the hand-set weights when none is set."""
    model = config.aggregator_a.model
    return RulesAggregator.from_config(config) if model is None else RulesAggregator.load(model, config)


class RulesAggregator:
    def __init__(
        self,
        weights: np.ndarray,
        bias: float,
        mean: np.ndarray,
        scale: np.ndarray,
        clip: float,
        config: Config,
    ) -> None:
        self.weights, self.bias, self.mean, self.scale, self.clip = weights, bias, mean, scale, clip
        self.config = config
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
        if scene.summary.n_persons == 0:
            return RiskResult(risk=0.0, level="normal", reason_uk="У кадрі нікого немає")
        x = scene_to_features(scene, self.config)
        z = self._standardise(x)
        risk = float(1.0 / (1.0 + np.exp(-(self.bias + self.weights @ z))))
        named = dict(zip(FEATURE_NAMES, x, strict=True))
        if named["weapon_held"]:
            risk = max(risk, min(1.0, thresholds.theta_high + thresholds.weapon_override_margin))
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

    def fit(self, features: np.ndarray, labels: np.ndarray, seed: int = 0) -> None:
        """Logistic regression on standardised features (train split only)."""
        from sklearn.linear_model import LogisticRegression

        self.mean = features.mean(axis=0)
        self.scale = np.where(features.std(axis=0) > 0, features.std(axis=0), 1.0)
        model = LogisticRegression(random_state=seed, max_iter=1000).fit(self._standardise(features), labels)
        self.weights, self.bias = model.coef_[0], float(model.intercept_[0])
        self._neutral_z = self._standardise(
            np.array([self.config.aggregator_a.neutral[n] for n in FEATURE_NAMES])
        )

    def save(self, path: Path, extra: dict | None = None) -> None:
        """Weights with the feature list and schema version; `extra` adds metadata (training, thresholds)."""
        data = {
            "features": list(FEATURE_NAMES),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "clip": self.clip,
            "schema_version": SCHEMA_VERSION,
        } | (extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, config: Config) -> "RulesAggregator":
        """Load saved weights; refuse ones made for other features, another schema or another perception."""
        from sentinel.pipeline import PERCEPTION_VERSION

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if tuple(data["features"]) != FEATURE_NAMES:
            raise ValueError(f"{path} was saved for features {data['features']}, expected {FEATURE_NAMES}")
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError(
                f"{path} was trained on schema v{data['schema_version']}, this is v{SCHEMA_VERSION}"
            )
        if data.get("perception_version", PERCEPTION_VERSION) != PERCEPTION_VERSION:
            raise ValueError(
                f"{path} was trained on perception {data['perception_version']}, this is {PERCEPTION_VERSION}"
            )
        arrays = {k: np.array(data[k], dtype=float) for k in ("weights", "mean", "scale")}
        return cls(
            arrays["weights"], float(data["bias"]), arrays["mean"], arrays["scale"], data["clip"], config
        )
