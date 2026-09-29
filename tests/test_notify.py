import json
import threading
import time
from pathlib import Path

import requests

from sentinel.incidents import IncidentEvent
from sentinel.notify import ConsoleNotifier, make_notifier
from sentinel.notify.telegram import TelegramNotifier

TOKEN = "123:SECRET"


def _event(kind: str, t: float = 12.0, keyframe: Path | None = None, **extra) -> IncidentEvent:
    fields = {
        "kind": kind,
        "incident_id": "r-cam-1-001",
        "source": "cam-1",
        "t": t,
        "level": "normal" if kind == "close" else "alarm",
        "max_risk": 0.82,
        "participants": (1, 2),
        "reason_uk": "Двоє людей зблизилися до 0,4 зросту; різкі рухи рук",
        "keyframe": keyframe,
        "started_at": 3.0,
        "duration_s": t - 3.0,
    }
    return IncidentEvent(**(fields | extra))


class FakeResponse:
    def __init__(self, status: int, payload: dict) -> None:
        self.status_code, self._payload = status, payload

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """Records Bot API calls; `failures` first calls raise a connection error."""

    def __init__(self, failures: int = 0, block: threading.Event | None = None) -> None:
        self.calls: list[tuple[str, dict, bool]] = []
        self.failures = failures
        self.block = block
        self.next_id = 100

    def post(self, url: str, data=None, files=None, timeout=None) -> FakeResponse:
        if self.block is not None:
            self.block.wait()
        if self.failures > 0:
            self.failures -= 1
            raise requests.ConnectionError("network down")
        method = url.rsplit("/", 1)[-1]
        self.calls.append((method, dict(data or {}), files is not None))
        self.next_id += 1
        return FakeResponse(200, {"ok": True, "result": {"message_id": self.next_id}})


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _telegram(session: FakeSession, clock: FakeClock | None = None, chats=("42",)) -> TelegramNotifier:
    return TelegramNotifier(
        TOKEN,
        list(chats),
        session=session,
        clock=clock or FakeClock(),
        sleep=lambda s: None,
        start_worker=False,
    )


def _photo(tmp_path: Path) -> Path:
    path = tmp_path / "keyframe.jpg"
    path.write_bytes(b"\xff\xd8fake")
    return path


def test_opening_sends_one_photo_with_a_ukrainian_caption_per_chat(tmp_path: Path):
    session = FakeSession()
    tg = _telegram(session, chats=("42", "43"))
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    assert [(m, d["chat_id"], photo) for m, d, photo in session.calls] == [
        ("sendPhoto", "42", True),
        ("sendPhoto", "43", True),
    ]
    caption = session.calls[0][1]["caption"]
    assert "ТРИВОГА" in caption and "зблизилися до 0,4 зросту" in caption and "0,82" in caption


def test_updates_edit_the_same_message_at_most_every_ten_seconds(tmp_path: Path):
    session, clock = FakeSession(), FakeClock()
    tg = _telegram(session, clock)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    message_id = 101
    clock.now = 11.0
    tg.handle(_event("update", t=20.0, max_risk=0.9))
    clock.now = 14.0
    tg.handle(_event("update", t=23.0, max_risk=0.95))  # too soon: kept for later
    tg.send_due_edits()
    edits = [d for m, d, _ in session.calls if m == "editMessageCaption"]
    assert len(edits) == 1 and edits[0]["message_id"] == message_id and "0,9" in edits[0]["caption"]
    clock.now = 21.5
    tg.send_due_edits()
    edits = [d for m, d, _ in session.calls if m == "editMessageCaption"]
    assert len(edits) == 2 and "0,95" in edits[1]["caption"]


def test_escalation_replies_with_a_new_photo_for_a_push_notification(tmp_path: Path):
    session = FakeSession()
    tg = _telegram(session)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    tg.handle(_event("escalate", t=15.0, keyframe=_photo(tmp_path), reason_uk="Зброя в руці: ніж"))
    method, data, photo = session.calls[1]
    assert method == "sendPhoto" and photo and data["reply_to_message_id"] == 101
    assert "Зброя в руці" in data["caption"]


def test_closing_edits_the_caption_to_finished_with_duration(tmp_path: Path):
    session, clock = FakeSession(), FakeClock()
    tg = _telegram(session, clock)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    clock.now = 2.0  # closing is never held back by the edit interval
    tg.handle(_event("close", t=45.0))
    method, data, _ = session.calls[-1]
    assert method == "editMessageCaption" and "Завершено" in data["caption"] and "42 с" in data["caption"]


def test_network_errors_are_retried(tmp_path: Path):
    session = FakeSession(failures=2)
    tg = _telegram(session)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    assert [m for m, _, _ in session.calls] == ["sendPhoto"]


def test_pipeline_never_waits_for_telegram(tmp_path: Path):
    gate = threading.Event()
    session = FakeSession(block=gate)  # the network hangs until the gate opens
    tg = TelegramNotifier(TOKEN, ["42"], session=session, sleep=lambda s: None)
    started = time.perf_counter()
    tg.on_event(_event("open", keyframe=_photo(tmp_path)))
    tg.on_event(_event("update", t=30.0))
    assert time.perf_counter() - started < 0.1
    gate.set()
    tg.close()
    assert [m for m, _, _ in session.calls][0] == "sendPhoto"


def test_token_never_appears_in_the_notifier_repr():
    assert TOKEN not in repr(_telegram(FakeSession()))


def test_console_notifier_prints_and_logs_every_event(tmp_path: Path, capsys):
    log = tmp_path / "incidents.jsonl"
    console = ConsoleNotifier(log)
    console.on_event(_event("open", keyframe=tmp_path / "k.jpg"))
    console.on_event(_event("close", t=45.0))
    console.close()
    printed = capsys.readouterr().out
    assert "ТРИВОГА" in printed and "Завершено" in printed
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["open", "close"] and rows[0]["keyframe"].endswith("k.jpg")


def test_without_a_token_notifications_go_to_the_console(tmp_path: Path):
    assert isinstance(make_notifier({}, tmp_path / "i.jsonl"), ConsoleNotifier)
    tg = make_notifier({"TELEGRAM_TOKEN": TOKEN, "TELEGRAM_CHAT_IDS": "42, 43"}, tmp_path / "i.jsonl")
    assert isinstance(tg, TelegramNotifier) and tg.chat_ids == ["42", "43"]
    tg.close()
