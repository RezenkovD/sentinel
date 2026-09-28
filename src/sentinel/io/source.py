"""Frame source agent: LatestFrameReader keeps only the freshest frame (live / paced / offline modes).

- live: a camera or RTSP stream is read by a background thread as fast as it delivers frames.
- paced: a video file is read by a background thread at its real speed, as if it were a live camera (demo).
- offline: a video file is read synchronously with no pauses and no races, so the frames picked on the
  analysis grid are deterministic (experiment on datasets: speed matters, real time does not).

In live and paced modes only the latest frame is kept: if analysis is slower than the source, older frames are
dropped instead of queued, so the lag never grows (SPEC, section 6). With `target_fps`, frames are handed out
no faster than the fixed analysis rate.
"""

import math
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import cv2
import numpy as np

Mode = Literal["live", "paced", "offline"]


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    t: float  # seconds: video time for files, time since start for live sources
    index: int  # position of the frame in the source stream


@dataclass(frozen=True)
class ReaderStats:
    received: int  # frames read from the source
    delivered: int  # frames handed to the analysis
    skipped: int  # frames received but never analysed (dropped as stale or off the analysis grid)


class LatestFrameReader:
    def __init__(self, source: str | int | Path, mode: Mode, target_fps: float | None = None) -> None:
        if mode not in ("live", "paced", "offline"):
            raise ValueError(f"unknown source mode: {mode!r}")
        if target_fps is not None and not target_fps > 0:
            raise ValueError(f"analysis rate must be positive, got {target_fps}")
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # webcam index as given by argparse or YAML, e.g. "0"
        is_file = isinstance(source, Path) or (isinstance(source, str) and "://" not in source)
        if is_file and not Path(source).exists():
            raise FileNotFoundError(source)
        self.mode = mode
        self.target_fps = target_fps
        self._cap = cv2.VideoCapture(int(source) if isinstance(source, int) else str(source))
        if not self._cap.isOpened():
            raise OSError(f"cannot open video source: {source}")
        self.source_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if mode != "live" and not self.source_fps > 0:
            raise ValueError(f"video has no frame rate, cannot use {mode} mode: {source}")

        self._received = 0
        self._delivered = 0
        self._skipped = 0
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._eof = False
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_tick = 0.0
        self._last_slot = -1

    @property
    def stats(self) -> ReaderStats:
        with self._cond:
            return ReaderStats(self._received, self._delivered, self._skipped)

    def read(self) -> Frame | None:
        """Next frame to analyse, or None when the source has ended.

        Raises RuntimeError (caused by the original error) if reading the source failed.
        """
        if self.mode == "offline":
            return self._read_offline()
        if self._thread is None:
            self._thread = threading.Thread(target=self._pump, name="frame-reader", daemon=True)
            self._thread.start()
        self._wait_for_tick()
        with self._cond:
            while self._pending is None and not self._eof:
                self._cond.wait()
            frame, self._pending = self._pending, None
            if frame is not None:
                self._delivered += 1
            elif self._error is not None:
                raise RuntimeError("reading the video source failed") from self._error
            return frame

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._cap.release()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        while (frame := self.read()) is not None:
            yield frame

    def _read_offline(self) -> Frame | None:
        while self._cap.grab():
            index = self._received
            self._received += 1
            slot = (
                index
                if self.target_fps is None
                else math.floor(index * self.target_fps / self.source_fps + 1e-9)
            )
            if slot == self._last_slot:
                self._skipped += 1
                continue
            ok, image = self._cap.retrieve()
            if not ok:
                break
            self._last_slot = slot
            self._delivered += 1
            return Frame(image=image, t=index / self.source_fps, index=index)
        return None

    def _wait_for_tick(self) -> None:
        if self.target_fps is None:
            return
        now = time.monotonic()
        if now < self._next_tick:
            time.sleep(self._next_tick - now)
        self._next_tick = max(self._next_tick, now) + 1.0 / self.target_fps

    def _pump(self) -> None:
        try:
            self._pump_frames()
        except Exception as exc:  # handed to the consumer: a dead thread must not leave read() waiting
            with self._cond:
                self._error = exc
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def _pump_frames(self) -> None:
        started = time.monotonic()
        while not self._stop.is_set():
            ok, image = self._cap.read()
            if not ok:
                break
            index = self._received
            if self.mode == "paced":
                t = index / self.source_fps
                delay = started + t - time.monotonic()
                if delay > 0 and self._stop.wait(delay):
                    break
            else:
                t = time.monotonic() - started
            with self._cond:
                if self._pending is not None:
                    self._skipped += 1
                self._pending = Frame(image=image, t=t, index=index)
                self._received += 1
                self._cond.notify_all()
