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
        "people": 2,
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
    assert (
        "Ймовірна бійка" in caption and "зблизилися до 0,4 зросту" in caption and "Оцінка: помірна" in caption
    )


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
    assert len(edits) == 1 and edits[0]["message_id"] == message_id and "триває 17 с" in edits[0]["caption"]
    clock.now = 21.5
    tg.send_due_edits()
    edits = [d for m, d, _ in session.calls if m == "editMessageCaption"]
    assert len(edits) == 2 and "триває 20 с" in edits[1]["caption"]


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
    assert "Ймовірна бійка" in printed and "Завершено" in printed
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["kind"] for r in rows] == ["open", "close"] and rows[0]["keyframe"].endswith("k.jpg")


def test_without_a_token_notifications_go_to_the_console(tmp_path: Path):
    assert isinstance(make_notifier({}, tmp_path / "i.jsonl"), ConsoleNotifier)
    tg = make_notifier({"TELEGRAM_TOKEN": TOKEN, "TELEGRAM_CHAT_IDS": "42, 43"}, tmp_path / "i.jsonl")
    assert isinstance(tg, TelegramNotifier) and tg.chat_ids == ["42", "43"]
    tg.close()


def test_incident_whose_opening_failed_is_sent_again_with_the_next_event(tmp_path: Path):
    session = FakeSession(failures=8)  # photo and text fallback both fail (4 attempts each): opening lost
    tg = _telegram(session)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    assert session.calls == []
    tg.handle(_event("escalate", t=15.0, keyframe=_photo(tmp_path), reason_uk="Зброя в руці: ніж"))
    method, data, photo = session.calls[0]
    assert method == "sendPhoto" and photo and "reply_to_message_id" not in data  # a fresh message
    tg.handle(_event("close", t=45.0))
    assert session.calls[-1][0] == "editMessageCaption"


def test_text_only_incident_is_edited_as_text(tmp_path: Path):
    session = FakeSession()
    tg = _telegram(session)
    tg.handle(_event("open", keyframe=None))  # no key frame: sent as a text message
    tg.handle(_event("close", t=45.0))
    assert [m for m, _, _ in session.calls] == ["sendMessage", "editMessageText"]
    assert "Завершено" in session.calls[-1][1]["text"]


def test_rate_limit_wait_is_capped():
    class RateLimited(FakeSession):
        def post(self, url, data=None, files=None, timeout=None):
            if not self.calls and self.failures == 0:
                self.failures = -1
                return FakeResponse(429, {"ok": False, "parameters": {"retry_after": 900}})
            return super().post(url, data, files, timeout)

    waits = []
    tg = TelegramNotifier(TOKEN, ["42"], session=RateLimited(), sleep=waits.append, start_worker=False)
    tg.handle(_event("open", keyframe=None))
    assert waits and max(waits) <= 60.0


def test_rejected_photo_falls_back_to_a_text_message(tmp_path: Path):
    session = FakeSession(failures=4)  # the photo upload keeps failing, the network then recovers
    tg = _telegram(session)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    assert [m for m, _, _ in session.calls] == ["sendMessage"]


def test_mute_holds_back_new_messages_but_not_the_closing_of_open_ones(tmp_path: Path):
    session, clock = FakeSession(), FakeClock()
    tg = _telegram(session, clock)
    tg.handle(_event("open"))  # before the mute: a message exists
    tg.mute(600.0)
    tg.handle(_event("escalate"))  # a push while muted: held back
    tg.handle(_event("open", incident_id="r-cam-1-002"))  # a new incident while muted: held back
    tg.handle(_event("close", t=40.0))  # the open message must not stay "ALARM"
    methods = [m for m, _, _ in session.calls]
    assert methods == ["sendMessage", "editMessageText"]
    clock.now = 700.0  # the mute is over
    tg.handle(_event("open", incident_id="r-cam-1-003"))
    assert [m for m, _, _ in session.calls][-1] == "sendMessage"


def test_the_caption_reads_like_an_operator_message():
    """Decision 03.10 (Dmytro): what happened, where, wall-clock time, how long, how many people, why in plain
    words, how sure and who decided — no track-id sums, no video offsets."""
    from sentinel.notify import caption

    origin = time.mktime((2026, 10, 3, 14, 32, 0, 0, 0, -1))  # the run started at 14:32:00
    opened = caption(_event("open", t=5.0, duration_s=0.0, decided_by="B"), place="Вхід", origin=origin)
    lines = opened.splitlines()
    assert lines[0] == "🔴 Ймовірна бійка — Вхід"
    assert lines[1] == "14:32:05 · ≈ 2 людей"
    assert lines[2] == "Двоє людей зблизилися до 0,4 зросту; різкі рухи рук"
    assert lines[3] == "Оцінка: помірна · рішення: B"
    ongoing = caption(_event("update", t=25.0, max_risk=0.95), place="Вхід", origin=origin)
    assert ongoing.splitlines()[1] == "14:32:25 · триває 22 с · ≈ 2 людей" and "висока" in ongoing
    armed = caption(_event("open", weapon=True), place="Вхід", origin=origin)
    assert armed.startswith("🔴 Зброя в руці — Вхід")
    closed = caption(_event("close", t=45.0), place="Вхід", origin=origin)
    assert closed.splitlines()[0] == "✅ Завершено о 14:32:45 (42 с) — Вхід"


def test_an_update_with_a_new_frame_replaces_the_photo_and_the_caption(tmp_path: Path):
    session, clock = FakeSession(), FakeClock()
    tg = _telegram(session, clock)
    tg.handle(_event("open", keyframe=_photo(tmp_path)))
    clock.now = 11.0
    tg.handle(_event("update", t=20.0, keyframe=_photo(tmp_path)))
    method, data, photo = session.calls[-1]
    media = json.loads(data["media"])
    assert method == "editMessageMedia" and photo and media["type"] == "photo"
    assert media["media"] == "attach://photo" and "триває 17 с" in media["caption"]
