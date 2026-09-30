from pathlib import Path

import pytest

from sentinel.cli import eval_command


def test_eval_command_maps_a_dashed_name_to_its_module():
    module, argv = eval_command("build-episodes", ["--out", "x"], Path("configs/default.yaml"))
    assert module == "sentinel.eval.build_episodes"
    assert argv == ["--out", "x"]


def test_eval_command_forwards_the_top_level_config_to_scripts_that_take_one():
    module, argv = eval_command("sanity", ["--per-class", "2"], Path("other.yaml"))
    assert module == "sentinel.eval.sanity"
    assert argv == ["--config", "other.yaml", "--per-class", "2"]


def test_eval_command_keeps_an_explicit_script_config():
    _, argv = eval_command("sanity", ["--config", "mine.yaml"], Path("other.yaml"))
    assert argv == ["--config", "mine.yaml"]


def test_unknown_eval_command_is_rejected():
    with pytest.raises(SystemExit):
        eval_command("nope", [], Path("configs/default.yaml"))
