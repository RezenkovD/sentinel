import json
import os
from pathlib import Path

import pytest

from sentinel.config import load_config
from sentinel.models import export_openvino, resolve_weights, weights_to_export


def _fake_exporter(calls: list[Path]):
    def export(pt: Path) -> None:
        calls.append(pt)
        out = pt.with_name(f"{pt.stem}_openvino_model")
        out.mkdir(exist_ok=True)
        (out / f"{pt.stem}.xml").write_text("<net/>")

    return export


def test_export_writes_the_model_and_a_stamp_with_the_source_hash(tmp_path: Path):
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights v1")
    calls: list[Path] = []
    out = export_openvino(pt, _fake_exporter(calls))
    assert out == tmp_path / "net_openvino_model" and calls == [pt]
    stamp = json.loads((out / "sentinel_export.json").read_text())
    assert stamp["source"] == "net.pt" and len(stamp["source_sha256"]) == 64


def test_export_is_skipped_while_the_source_is_unchanged(tmp_path: Path):
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights v1")
    calls: list[Path] = []
    export_openvino(pt, _fake_exporter(calls))
    export_openvino(pt, _fake_exporter(calls))
    assert calls == [pt]


def test_changed_weights_are_exported_again(tmp_path: Path):
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights v1")
    calls: list[Path] = []
    export_openvino(pt, _fake_exporter(calls))
    pt.write_bytes(b"weights v2")
    export_openvino(pt, _fake_exporter(calls))
    assert calls == [pt, pt]


def test_torch_backend_uses_the_checkpoint_itself(tmp_path: Path):
    pt = tmp_path / "net.pt"
    assert resolve_weights(pt, "torch") == pt


def test_openvino_backend_uses_an_up_to_date_export(tmp_path: Path):
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights v1")
    out = export_openvino(pt, _fake_exporter([]))
    assert resolve_weights(pt, "openvino") == out


@pytest.mark.parametrize("stale", [False, True])
def test_missing_or_stale_export_is_an_error_not_a_silent_export(tmp_path: Path, stale: bool):
    pt = tmp_path / "net.pt"
    pt.write_bytes(b"weights v1")
    if stale:
        export_openvino(pt, _fake_exporter([]))
        pt.write_bytes(b"weights v2")
    with pytest.raises(RuntimeError, match="sentinel export"):
        resolve_weights(pt, "openvino")


def test_every_model_of_the_config_is_exported():
    config = load_config()
    names = {p.name for p in weights_to_export(config)}
    assert {config.models.people.weights.name, config.models.pose.weights.name} <= names
    assert config.weapons.weights.name in names


def test_openvino_runs_on_the_intel_cpu_device():
    config = load_config()
    models = config.models.model_copy(update={"backend": "openvino"})
    assert models.runtime_device == "intel:cpu"
    assert config.models.model_copy(update={"backend": "torch"}).runtime_device == "cpu"


def test_library_telemetry_is_off_for_ultralytics():
    import sentinel  # noqa: F401  (sets the environment before ultralytics is imported)

    assert os.environ["YOLO_OFFLINE"].lower() == "true"


def test_openvino_telemetry_has_no_consent_during_export():
    import openvino_telemetry

    from sentinel.models import no_openvino_telemetry

    with no_openvino_telemetry():
        telemetry = openvino_telemetry.Telemetry(
            app_name="sentinel-test", app_version="0", tid="none", enable_opt_in_dialog=False
        )
        assert telemetry.consent is False


def test_openvino_telemetry_is_off_by_the_environment_set_at_import():
    """Decision 30.09 (report 03), item 11: project scope, environment at import, no user settings touched.
    The OpenVINO converter declines telemetry in CI (`disable_in_ci=True`), so `CI=true` is set for us."""
    import openvino_telemetry

    import sentinel  # noqa: F401

    assert os.environ["CI"] == "true"
    telemetry = openvino_telemetry.Telemetry(
        app_name="sentinel-test", app_version="0", tid="none", enable_opt_in_dialog=False, disable_in_ci=True
    )
    telemetry.init(  # the class is a singleton: set it up the way the converter does
        tid="none",
        app_name="sentinel-test",
        app_version="0",
        backend="ga4",
        enable_opt_in_dialog=False,
        disable_in_ci=True,
    )
    assert telemetry.consent is False
