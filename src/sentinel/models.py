"""Model files per backend: PyTorch checkpoints or their OpenVINO exports (decision 30.09.1).

`sentinel export` converts every checkpoint the config uses to OpenVINO IR next to it
(`<name>_openvino_model/`) and stamps the export with the checkpoint's SHA-256. Running with
`models.backend: openvino` never exports silently: a missing or stale export (the checkpoint changed since)
is an error that names the command to run.
"""

import hashlib
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

STAMP = "sentinel_export.json"
Exporter = Callable[[Path], None]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_dir(pt: Path) -> Path:
    return pt.with_name(f"{pt.stem}_openvino_model")


def _up_to_date(pt: Path) -> bool:
    out = export_dir(pt)
    stamp = out / STAMP
    if not stamp.exists() or not (out / f"{pt.stem}.xml").exists():
        return False
    return json.loads(stamp.read_text(encoding="utf-8")).get("source_sha256") == sha256(pt)


@contextmanager
def no_openvino_telemetry() -> Iterator[None]:
    """OpenVINO's model converter sends usage statistics unless a consent file says no; here it says no.

    Only this process is affected: nothing is written to the user's settings.
    """
    from openvino_telemetry.utils.opt_in_checker import ConsentCheckResult, OptInChecker

    original = OptInChecker.check
    OptInChecker.check = lambda self, *args, **kwargs: ConsentCheckResult.DECLINED
    try:
        yield
    finally:
        OptInChecker.check = original


def _ultralytics_export(pt: Path) -> None:
    from ultralytics import YOLO

    with no_openvino_telemetry():
        # dynamic input shape: one export serves every imgsz; FP32 keeps the outputs close to PyTorch
        YOLO(str(pt)).export(format="openvino", dynamic=True, half=False)


def export_openvino(pt: Path, exporter: Exporter | None = None) -> Path:
    """Export `pt` to OpenVINO unless an export of these exact weights already exists; returns its folder."""
    pt = Path(pt)
    if not pt.exists():
        raise FileNotFoundError(f"weights not found: {pt} (see data/SOURCES.md)")
    out = export_dir(pt)
    if not _up_to_date(pt):
        (exporter or _ultralytics_export)(pt)
        import importlib.metadata as meta

        versions = {}
        for package in ("openvino", "ultralytics"):
            try:
                versions[package] = meta.version(package)
            except meta.PackageNotFoundError:
                versions[package] = None
        stamp = {"source": pt.name, "source_sha256": sha256(pt), **versions}
        (out / STAMP).write_text(json.dumps(stamp, indent=1), encoding="utf-8")
    return out


def resolve_weights(pt: Path, backend: str) -> Path:
    """What to load for `backend`: the checkpoint (torch) or its up-to-date OpenVINO export."""
    pt = Path(pt)
    if backend == "torch":
        return pt
    if backend != "openvino":
        raise ValueError(f"unknown model backend {backend!r} (torch | openvino)")
    if not _up_to_date(pt):
        raise RuntimeError(
            f"OpenVINO export of {pt} is missing or older than the checkpoint: run `sentinel export`"
        )
    return export_dir(pt)


def weights_to_export(config) -> list[Path]:
    """Every checkpoint the config can load: people, pose and weapon models."""
    paths = [config.models.people.weights, config.models.pose.weights, config.weapons.weights]
    return list(dict.fromkeys(Path(p) for p in paths))
