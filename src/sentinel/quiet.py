"""Quiet-window measurements (session 03, step 1): speed is measured only when the machine is otherwise idle.

Dmytro keeps working on this laptop (PyCharm, other projects), so a measurement waits for a quiet window by
itself: the 1-minute mean of *foreign* load — tasks running or waiting for I/O, as the kernel load average
counts them, minus our own threads — below `start_limit` for `hold_s` seconds in a row, on mains power with
the `performance` profile. A measurement during which that mean rose above `noisy_limit` is kept, marked noisy
and repeated later. Our own background work (episode extraction, see EXTRACT_PIDS) is paused (SIGSTOP)
for the duration of each measurement and resumed (SIGCONT) afterwards; every extraction announces itself with
a pid file in data/episodes/.extract/, checked against /proc/<pid>/cmdline before any signal is sent.

Foreign load is used instead of /proc/loadavg because the measurement itself and our extraction also
load the machine; the plain load average would call every measurement noisy.
"""

import os
import signal
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def system_busy(stat: Path = Path("/proc/stat")) -> int:
    """Tasks running or blocked on I/O right now (what the load average averages)."""
    values = {}
    for line in stat.read_text().splitlines():
        key, _, rest = line.partition(" ")
        if key in ("procs_running", "procs_blocked"):
            values[key] = int(rest.split()[0])
    return values.get("procs_running", 0) + values.get("procs_blocked", 0)


def own_busy_threads(pids: set[int], proc: Path = Path("/proc")) -> int:
    """Threads of our processes that are running or blocked on I/O."""
    busy = 0
    for pid in pids:
        try:
            tasks = list((proc / str(pid) / "task").iterdir())
        except OSError:  # the process is gone
            continue
        for task in tasks:
            try:
                stat = (task / "stat").read_text()
            except OSError:
                continue
            busy += stat[stat.rindex(")") + 2] in "RD"  # the name may contain spaces and parentheses
    return busy


@dataclass
class LoadWindow:
    """Rolling mean of load samples over the last `span_s` seconds, and how long it has stayed low."""

    span_s: float
    _samples: deque = field(default_factory=deque)
    _means: deque = field(default_factory=deque)  # (t, rolling mean) history for quiet_for

    def add(self, t: float, value: float) -> None:
        self._samples.append((t, value))
        while self._samples and self._samples[0][0] <= t - self.span_s:
            self._samples.popleft()
        self._means.append((t, self.mean(t)))
        while len(self._means) > 100_000:
            self._means.popleft()

    def mean(self, t: float) -> float:
        values = [v for ts, v in self._samples if ts > t - self.span_s]
        return sum(values) / len(values) if values else 0.0

    def quiet_for(self, t: float, limit: float) -> float:
        """Seconds since the rolling mean last was at or above `limit` (0 if it is now)."""
        since = None
        for ts, mean in reversed(self._means):
            if mean >= limit:
                break
            since = ts
        return 0.0 if since is None else t - since


@dataclass
class QuietRunner:
    """Measures each combination in a quiet window; noisy measurements are recorded and repeated."""

    sample: Callable[[], float]  # current foreign load (running + blocked tasks outside our processes)
    measure: Callable[[Any, Callable[[], None]], dict]  # measure(combo, during): `during` samples the load
    power: Callable[[], tuple[bool, str]]  # (on mains power, power profile)
    pause: Callable[[], None]  # pause our background work
    resume: Callable[[], None]
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    start_limit: float = 1.5
    hold_s: float = 120.0
    noisy_limit: float = 2.0
    span_s: float = 60.0
    step_s: float = 5.0
    log: Callable[[str], None] = lambda message: None
    on_row: Callable[[dict], None] = lambda row: None  # called with every finished row (to save it at once)
    hold_pause: bool = False  # keep our background work paused from the first measurement to the end

    def run(self, combos: Sequence[Any], deadline_s: float) -> list[dict]:
        paused = [False]
        try:
            return self._run(combos, deadline_s, paused)
        finally:
            if paused[0]:
                self.resume()

    def _run(self, combos: Sequence[Any], deadline_s: float, paused: list[bool]) -> list[dict]:
        rows: list[dict] = []
        pending = list(combos)
        window = LoadWindow(self.span_s)
        started = self.clock()
        waiting_logged = None
        while pending and self.clock() - started < deadline_s:
            t = self.clock()
            window.add(t, self.sample())
            on_mains, profile = self.power()
            if (
                not on_mains
                or profile != "performance"
                or window.quiet_for(t, self.start_limit) < self.hold_s
            ):
                if waiting_logged is None or t - waiting_logged >= 600:
                    self.log(
                        f"waiting: foreign load {window.mean(t):.2f}, mains {on_mains}, profile {profile}"
                    )
                    waiting_logged = t
                self.sleep(self.step_s)
                continue
            combo = pending[0]
            means = [window.mean(t)]

            def during(means: list[float] = means) -> None:
                now = self.clock()
                window.add(now, self.sample())
                means.append(window.mean(now))

            self.log(f"measuring {combo} (foreign load {means[0]:.2f})")
            if not paused[0]:
                self.pause()
                paused[0] = True
            row = self.measure(combo, during)
            if not self.hold_pause:
                self.resume()
                paused[0] = False
            noisy = max(means) > self.noisy_limit
            row = row | {
                "noisy": noisy,
                "foreign_load_mean": round(sum(means) / len(means), 2),
                "foreign_load_max": round(max(means), 2),
            }
            rows.append(row)
            self.on_row(row)
            self.log(
                f"{'noisy, will repeat' if noisy else 'done'}: {combo} (foreign load max {max(means):.2f})"
            )
            if not noisy:
                pending.pop(0)
            waiting_logged = None
        return rows


# --- the real machine ---

EXTRACT_PIDS = Path("data/episodes/.extract")  # one <pid>.pid per running `sentinel eval extract`


def register_extraction(directory: Path = EXTRACT_PIDS, pid: int | None = None) -> Path:
    """Announce a running extraction so that the quiet bench can pause it."""
    pid = os.getpid() if pid is None else pid
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pid}.pid"
    path.write_text(str(pid))
    return path


def unregister_extraction(directory: Path = EXTRACT_PIDS, pid: int | None = None) -> None:
    (directory / f"{os.getpid() if pid is None else pid}.pid").unlink(missing_ok=True)


def _is_extraction(pid: int) -> bool:
    """The process exists and really is a `sentinel … extract` (a reused PID must never be paused)."""
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    text = b" ".join(args).decode(errors="replace")
    return "sentinel" in text and "extract" in text


def extraction_pids(directory: Path = EXTRACT_PIDS) -> list[int]:
    """Live extraction processes; files of processes that are gone or are something else are removed."""
    pids = []
    for path in sorted(directory.glob("*.pid")) if directory.exists() else []:
        try:
            pid = int(path.read_text().strip())
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            continue
        if _is_extraction(pid):
            pids.append(pid)
        else:
            path.unlink(missing_ok=True)
    return pids


def foreign_load(own: Callable[[], set[int]]) -> float:
    """Busy tasks outside our processes; the reading thread itself is one of ours."""
    return max(0, system_busy() - own_busy_threads(own()))


def pause_extraction(directory: Path = EXTRACT_PIDS) -> None:
    for pid in extraction_pids(directory):
        os.kill(pid, signal.SIGSTOP)


def resume_extraction(directory: Path = EXTRACT_PIDS) -> None:
    for pid in extraction_pids(directory):
        os.kill(pid, signal.SIGCONT)


def machine_load_average() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])
