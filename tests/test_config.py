from pathlib import Path

import pytest
from pydantic import ValidationError

from sentinel.config import load_config

DEFAULT = Path(__file__).parents[1] / "configs" / "default.yaml"


def test_shipped_default_config_is_valid():
    config = load_config(DEFAULT)
    assert 8 <= config.analysis.fps <= 10  # SPEC, section 6
    assert config.models.person.weights.name.endswith(".pt")


def test_tracker_settings_are_versioned_with_the_project():
    # Tracker thresholds belong to configs/, not to whatever the installed Ultralytics ships.
    tracker = Path(load_config(DEFAULT).models.person.tracker)
    assert tracker.parent.name == "configs"
    assert (DEFAULT.parents[1] / tracker).is_file()


def test_misspelled_key_is_rejected(tmp_path: Path):
    text = DEFAULT.read_text().replace("window_s:", "windows_s:")
    path = tmp_path / "typo.yaml"
    path.write_text(text)
    with pytest.raises(ValidationError):
        load_config(path)
