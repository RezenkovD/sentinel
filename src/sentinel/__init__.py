"""Sentinel: multi-agent situational danger assessment for CCTV video (weapons + aggression)."""

import os

# Ultralytics pip-installs missing optional packages at runtime (e.g. `lap` for ByteTrack), which silently
# changes the locked environment. Dependencies come only from uv.lock, so auto-install is off. The flag is
# read when ultralytics is first imported, hence it is set here, before any sentinel module imports it.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# Ultralytics sends anonymous usage events (model name, speeds, CPU) to Google Analytics while its global
# `sync` setting is on. Offline mode turns that off for our processes only, without touching user settings;
# nothing is downloaded at run time anyway (weights come from data/SOURCES.md).
os.environ.setdefault("YOLO_OFFLINE", "true")
# OpenVINO has no switch of its own: its model converter declines usage statistics when it runs in CI
# (`disable_in_ci=True`), so our processes say so (decision 30.09, report 03: project scope, environment at
# import, user settings untouched). Of our dependencies only torch.compile and pytest also read `CI`; neither
# changes what Sentinel computes.
os.environ.setdefault("CI", "true")

__version__ = "0.1.0"
