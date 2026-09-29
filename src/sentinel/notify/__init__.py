"""Operator notifications: one message per incident (SPEC §7).

`make_notifier` picks Telegram when TELEGRAM_TOKEN and TELEGRAM_CHAT_IDS are set (.env), else the console.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from sentinel.incidents import IncidentEvent

TITLES = {
    "open": "🚨 ТРИВОГА",
    "update": "🚨 ТРИВОГА триває",
    "escalate": "⚠️ Ескалація",
    "close": "✅ Завершено",
    "summary": "🚨 Забагато інцидентів",
}
CAPTION_LIMIT = 1024  # Telegram photo caption limit


class Notifier(Protocol):
    def on_event(self, event: IncidentEvent) -> None: ...

    def close(self) -> None: ...


def _num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _clock(seconds: float) -> str:
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def caption(event: IncidentEvent) -> str:
    """Ukrainian text of a notification (photo caption / message)."""
    if event.kind == "summary":
        return (
            f"{TITLES['summary']} · {event.source}\n"
            f"Ще {event.suppressed} за останню хвилину (без окремих повідомлень) — див. журнал інцидентів"
        )
    head = f"{TITLES[event.kind]} · {event.source}"
    if event.merged:
        head += " (поновлено)"
    if event.stream_ended:
        head += " (кінець відео / потоку)"
    if event.kind in ("open", "escalate"):
        when = f"Час: {_clock(event.t)}"
    else:
        when = (
            f"Час: {_clock(event.started_at)}–{_clock(event.t)}, тривалість {int(round(event.duration_s))} с"
        )
    lines = [
        head,
        when,
        f"Ризик: макс. {_num(event.max_risk)} · учасників: {event.people}",
        f"Причина: {event.reason_uk}",
    ]
    return "\n".join(lines)[:CAPTION_LIMIT]


class ConsoleNotifier:
    """Prints every event and appends it to a JSONL log (the default until there is a bot token)."""

    def __init__(self, jsonl: Path) -> None:
        self.jsonl = Path(jsonl)
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: IncidentEvent) -> None:
        print(caption(event).replace("\n", " | "), flush=True)
        row = asdict(event) | {"keyframe": str(event.keyframe) if event.keyframe else None}
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        pass


def make_notifier(env: Mapping[str, str], jsonl: Path, edit_interval_s: float = 10.0) -> Notifier:
    token = (env.get("TELEGRAM_TOKEN") or "").strip()
    chats = [c.strip() for c in (env.get("TELEGRAM_CHAT_IDS") or "").split(",") if c.strip()]
    if token and chats:
        from sentinel.notify.telegram import TelegramNotifier

        return TelegramNotifier(token, chats, edit_interval_s=edit_interval_s)
    return ConsoleNotifier(jsonl)
