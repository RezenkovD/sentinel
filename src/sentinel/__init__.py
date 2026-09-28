"""Sentinel: multi-agent situational danger assessment for CCTV video (weapons + aggression)."""

import os

# Ultralytics pip-installs missing optional packages at runtime (e.g. `lap` for ByteTrack), which silently
# changes the locked environment. Dependencies come only from uv.lock, so auto-install is off. The flag is
# read when ultralytics is first imported, hence it is set here, before any sentinel module imports it.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

__version__ = "0.1.0"
