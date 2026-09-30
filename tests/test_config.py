from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.config import load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_shipped_default_config_is_valid():
    config = load_config(DEFAULT)
    assert 8 <= config.analysis.fps <= 10  # SPEC, section 6
    assert config.models.people.weights.name.endswith(".pt")


def test_tracker_settings_are_versioned_with_the_project():
    # Tracker thresholds belong to configs/, not to whatever the installed Ultralytics ships.
    tracker = Path(load_config(DEFAULT).models.people.tracker)
    assert tracker.parent.name == "configs"
    assert (DEFAULT.parents[1] / tracker).is_file()


def test_misspelled_key_is_rejected(tmp_path: Path):
    text = DEFAULT.read_text().replace("window_s:", "windows_s:")
    path = tmp_path / "typo.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        load_config(path)


def test_shipped_zone_profiles_are_valid():
    from sentinel.agents.context import ZoneContext

    config = load_config(DEFAULT)
    zones = ZoneContext.from_yaml(DEFAULT.parents[1] / config.context.zones)
    assert zones.zone_for(config.source.camera_id).hours in {"day", "night"}


def test_feature_thresholds_come_from_the_config():
    features = load_config(DEFAULT).features.to_feature_config()
    assert features.contact_distance_h < features.pair_max_distance_h


def test_more_than_five_pairs_is_rejected_at_load_time(tmp_path: Path):
    # SPEC §5 allows at most 5 pairs in a scene description; a larger value would stop the run later
    text = DEFAULT.read_text().replace("  max_pairs: 5", "  max_pairs: 6")
    path = tmp_path / "pairs.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        load_config(path)


def test_baseline_thresholds_come_from_the_config():
    baselines = load_config(DEFAULT).baselines
    assert baselines.weapon_conf == 0.30 and baselines.motion_energy == 2.14
