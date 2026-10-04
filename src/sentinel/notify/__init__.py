"""Operator notifications: one message per incident (SPEC §7).

`make_notifier` picks Telegram when TELEGRAM_TOKEN and TELEGRAM_CHAT_IDS are set (.env), else the console.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from sentinel.incidents import IncidentEvent

TITLES = {
    "fight": "🔴 Ймовірна бійка",
    "weapon": "🔴 Зброя в руці",
    "escalate": "⚠️ Ескалація: зброя в руці",
}
CAPTION_LIMIT = 1024  # Telegram photo caption limit


class Notifier(Protocol):
    def on_event(self, event: IncidentEvent) -> None: ...

    def close(self) -> None: ...


def _num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def _clock(seconds: float, origin: float | None) -> str:
    """Wall-clock time `seconds` into the run (origin: its start, epoch s); without it, mm:ss into the run."""
    if origin is not None:
        return time.strftime("%H:%M:%S", time.localtime(origin + seconds))
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes:02d}:{secs:02d}"


def _sureness(risk: float) -> str:
    return "висока" if risk >= 0.9 else "помірна"


def caption(event: IncidentEvent, place: str = "Камера", origin: float | None = None) -> str:
    """Ukrainian operator message (photo caption / text): what happened, where, when, how long, how many
    people, why in plain words, how sure and who decided (decision 03.10 on the format)."""
    if event.kind == "summary":
        return (
            f"🔴 Забагато інцидентів — {place}\n"
            f"Ще {event.suppressed} за останню хвилину без окремих повідомлень — див. журнал інцидентів"
        )
    seconds = int(round(event.duration_s))
    if event.kind == "close":
        head = f"✅ Завершено о {_clock(event.t, origin)} ({seconds} с) — {place}"
        if event.stream_ended:
            head += " · кінець відео / потоку"
        return "\n".join([head, event.reason_uk])[:CAPTION_LIMIT]
    title = TITLES["escalate"] if event.kind == "escalate" else TITLES["weapon" if event.weapon else "fight"]
    head = f"{title} — {place}" + (" · поновлено" if event.merged else "")
    when = (
        _clock(event.t, origin) + (f" · триває {seconds} с" if seconds else "") + f" · ≈ {event.people} людей"
    )
    sure = "зброю підтверджено у 2 вікнах поспіль" if event.weapon else _sureness(event.max_risk)
    lines = [
        head,
        when,
        event.reason_uk,
        f"Оцінка: {sure} · рішення: {event.decided_by}",
    ]
    return "\n".join(lines)[:CAPTION_LIMIT]


class ConsoleNotifier:
    """Prints every event and appends it to a JSONL log (the default until there is a bot token)."""

    def __init__(self, jsonl: Path, place: str = "Камера", origin: float | None = None) -> None:
        self.place, self.origin = place, origin
        self.jsonl = Path(jsonl)
        self.jsonl.parent.mkdir(parents=True, exist_ok=True)

    def on_event(self, event: IncidentEvent) -> None:
        print(caption(event, self.place, self.origin).replace("\n", " | "), flush=True)
        row = asdict(event) | {"keyframe": str(event.keyframe) if event.keyframe else None}
        with self.jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        pass


def make_notifier(
    env: Mapping[str, str],
    jsonl: Path,
    edit_interval_s: float = 10.0,
    place: str = "Камера",
    origin: float | None = None,
) -> Notifier:
    token = (env.get("TELEGRAM_TOKEN") or "").strip()
    chats = [c.strip() for c in (env.get("TELEGRAM_CHAT_IDS") or "").split(",") if c.strip()]
    if token and chats:
        from sentinel.notify.telegram import TelegramNotifier

        return TelegramNotifier(token, chats, edit_interval_s=edit_interval_s, place=place, origin=origin)
    return ConsoleNotifier(jsonl, place, origin)
