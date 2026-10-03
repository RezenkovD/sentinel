"""Telegram notifier: one message per incident (sendPhoto, then editMessageCaption); Bot API via requests.

Network work happens in a background thread fed by a queue, so the pipeline never waits (SPEC §6, Н2).
Edits of an incident's caption are sent at most every `edit_interval_s`; a newer caption waiting for its
slot replaces the older one. An escalation (or an incident joined again) is a reply with a new photo, so
the phone gets a push notification; closing always edits the caption at once. Failed calls are retried
with a pause. The bot token is never logged or printed.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path

import requests

from sentinel.incidents import IncidentEvent
from sentinel.notify import caption

log = logging.getLogger(__name__)
API = "https://api.telegram.org"


class TelegramNotifier:
    def __init__(
        self,
        token: str,
        chat_ids: list[str],
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        edit_interval_s: float = 10.0,
        retries: int = 3,
        timeout_s: float = 10.0,
        max_wait_s: float = 60.0,
        start_worker: bool = True,
        place: str = "Камера",
        origin: float | None = None,  # wall-clock start of the run (epoch s): captions show the time of day
    ) -> None:
        self._token = token
        self.place, self.origin = place, origin
        self.chat_ids = chat_ids
        self._session = session or requests.Session()
        self._clock, self._sleep = clock, sleep
        self.edit_interval_s, self.retries, self.timeout_s = edit_interval_s, retries, timeout_s
        self.max_wait_s = max_wait_s  # cap on a 429 retry_after pause: the queue must keep moving
        self._messages: dict[str, dict[str, int]] = {}  # incident -> chat -> message id
        self._photo: dict[str, bool] = {}  # incident -> its message carries a photo (else plain text)
        self._last_edit: dict[str, float] = {}
        self._pending: dict[str, str] = {}  # incident -> newest caption waiting for its edit slot
        self._queue: queue.Queue[IncidentEvent | None] = queue.Queue()
        self._muted_until = float("-inf")  # /mute: no new messages or pushes until then
        self._worker: threading.Thread | None = None
        if start_worker:
            self._worker = threading.Thread(target=self._run, name="telegram", daemon=True)
            self._worker.start()

    def __repr__(self) -> str:
        return f"TelegramNotifier(chats={self.chat_ids})"

    def _caption(self, event: IncidentEvent) -> str:
        return caption(event, self.place, self.origin)

    def mute(self, seconds: float) -> None:
        """No new incident messages and no escalation pushes for `seconds`; captions of messages already sent
        are still edited, so an open incident never stays "ALARM" on the phone."""
        self._muted_until = self._clock() + seconds

    def muted(self) -> bool:
        return self._clock() < self._muted_until

    def on_event(self, event: IncidentEvent) -> None:
        self._queue.put(event)  # never blocks the pipeline

    def close(self, timeout: float = 30.0) -> None:
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout)

    def _run(self) -> None:
        while True:
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                self.send_due_edits()
                continue
            if event is None:
                self.send_due_edits(force=True)
                return
            try:
                self.handle(event)
            except Exception:  # a broken message must not stop later notifications
                log.exception("telegram: failed to handle %s %s", event.kind, event.incident_id)
            self.send_due_edits()

    # --- message logic (synchronous; the worker thread calls it) ---

    def handle(self, event: IncidentEvent) -> None:
        text = self._caption(event)
        if self.muted():
            if event.kind in ("open", "summary") or not self._messages.get(event.incident_id):
                log.info("telegram: muted, %s %s not sent", event.kind, event.incident_id)
                return
            if event.kind == "escalate" or event.merged:  # no push: only the caption of the existing message
                self._edit(event.incident_id, self._caption(_as_update(event)))
                return
        if event.kind == "open" or not self._messages.get(event.incident_id):
            # the opening message (or a new one if sending the opening failed: the incident must not be lost)
            self._open(event.incident_id, text, event.keyframe)
        elif event.kind == "escalate" or (event.kind == "update" and event.merged):
            for chat, mid in self._messages[event.incident_id].items():
                self._send(chat, text, event.keyframe, reply_to=mid)
            self._edit(event.incident_id, self._caption(_as_update(event)))
        elif event.kind == "update":
            self._edit(event.incident_id, text)
        elif event.kind == "close":
            self._edit(event.incident_id, text, force=True)
        elif event.kind == "summary":
            self._edit(event.incident_id, text, force=True)

    def _open(self, incident_id: str, text: str, photo: Path | None) -> None:
        sent = {
            chat: result for chat in self.chat_ids if (result := self._send(chat, text, photo)) is not None
        }
        if sent:
            self._messages[incident_id] = {chat: mid for chat, (mid, _) in sent.items()}
            self._photo[incident_id] = next(iter(sent.values()))[1]
            self._last_edit[incident_id] = self._clock()

    def send_due_edits(self, force: bool = False) -> None:
        for incident_id, text in list(self._pending.items()):
            if (
                force
                or self._clock() - self._last_edit.get(incident_id, float("-inf")) >= self.edit_interval_s
            ):
                self._edit(incident_id, text, force=True)

    def _edit(self, incident_id: str, text: str, force: bool = False) -> None:
        messages = self._messages.get(incident_id)
        if not messages:
            return
        if (
            not force
            and self._clock() - self._last_edit.get(incident_id, float("-inf")) < self.edit_interval_s
        ):
            self._pending[incident_id] = text
            return
        self._pending.pop(incident_id, None)
        for chat, mid in messages.items():
            if self._photo.get(incident_id, True):
                self._post("editMessageCaption", {"chat_id": chat, "message_id": mid, "caption": text})
            else:
                self._post("editMessageText", {"chat_id": chat, "message_id": mid, "text": text})
        self._last_edit[incident_id] = self._clock()

    def _send(
        self, chat: str, text: str, photo: Path | None, reply_to: int | None = None
    ) -> tuple[int, bool] | None:
        """(message id, has photo) or None; a rejected photo falls back to a text message."""
        data: dict = {"chat_id": chat}
        if reply_to is not None:
            data["reply_to_message_id"] = reply_to
            data["allow_sending_without_reply"] = True  # the operator may have deleted the original
        if photo is not None and Path(photo).exists():
            result = self._post("sendPhoto", data | {"caption": text}, photo=Path(photo))
            if result is not None:
                return result.get("message_id"), True
        result = self._post("sendMessage", data | {"text": text})
        return None if result is None else (result.get("message_id"), False)

    def _post(self, method: str, data: dict, photo: Path | None = None) -> dict | None:
        url = f"{API}/bot{self._token}/{method}"
        for attempt in range(self.retries + 1):
            wait = min(2.0**attempt, 30.0)
            try:
                if photo is not None:
                    with photo.open("rb") as f:
                        response = self._session.post(
                            url, data=data, files={"photo": f}, timeout=self.timeout_s
                        )
                else:
                    response = self._session.post(url, data=data, timeout=self.timeout_s)
                payload = response.json()
                if response.status_code == 200 and payload.get("ok"):
                    return payload.get("result", {})
                if response.status_code == 429:
                    wait = min(float(payload.get("parameters", {}).get("retry_after", wait)), self.max_wait_s)
                elif response.status_code < 500:
                    log.error("telegram %s rejected: %s", method, payload.get("description"))
                    return None
            except (requests.RequestException, ValueError) as exc:
                log.warning("telegram %s attempt %d failed: %s", method, attempt + 1, type(exc).__name__)
            if attempt < self.retries:
                self._sleep(wait)
        log.error("telegram %s: giving up after %d attempts", method, self.retries + 1)
        return None


def _as_update(event: IncidentEvent) -> IncidentEvent:
    """Caption of the main message after an escalation: an ongoing alarm."""
    from dataclasses import replace

    return replace(event, kind="update", merged=False)
