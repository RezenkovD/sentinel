"""Telegram bot commands (SPEC §3): `/mode A|B|cascade`, `/status`, `/mute 10m` — getUpdates long polling.

A background thread polls the Bot API; only messages from the allowed chats (TELEGRAM_CHAT_IDS) are acted on,
others are ignored without logging their text. The controller is the running `sentinel run`: it switches the
aggregation mode without a restart, reports its state and mutes new notifications. The bot token is never
logged or printed.
"""

import logging
import re
import threading
from typing import Protocol

import requests

log = logging.getLogger(__name__)
API = "https://api.telegram.org"
HELP = "Команди: /mode A | B | cascade — режим агрегації; /status — стан; /mute 10m — без нових сповіщень."


class Controller(Protocol):
    def set_mode(self, mode: str) -> str | None: ...  # None on success, else the reason in Ukrainian

    def status(self) -> dict: ...

    def mute(self, seconds: float) -> None: ...


def parse_duration(text: str) -> float | None:
    match = re.fullmatch(r"(\d+)\s*([smh])", text.strip())
    return None if match is None else int(match[1]) * {"s": 1.0, "m": 60.0, "h": 3600.0}[match[2]]


def _num(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def status_text(s: dict) -> str:
    queue = s.get("b_queue")
    b = (
        f"черга B: {queue['queued']}, зайнятий: {'так' if queue['busy'] else 'ні'}, "
        f"витіснено: {queue['dropped']}"
        if queue
        else "B недоступний"
    )
    return (
        f"Режим: {s['mode']} · RTF {_num(s['rtf'])} · {s['fps']:.1f} кадр/с · відкритих інцидентів: "
        f"{s['open_incidents']} · {b}"
    )


class BotCommands:
    def __init__(
        self,
        token: str,
        chat_ids: list[str],
        controller: Controller,
        session=None,
        poll_timeout_s: int = 25,
        start: bool = True,
    ) -> None:
        self._token, self.chat_ids, self.controller = token, set(chat_ids), controller
        self._session = session or requests.Session()
        self.poll_timeout_s = poll_timeout_s
        self._offset: int | None = None
        self._stop = threading.Event()
        self._thread = None
        if start:
            self._thread = threading.Thread(target=self._loop, name="telegram-commands", daemon=True)
            self._thread.start()

    def __repr__(self) -> str:
        return f"BotCommands(chats={sorted(self.chat_ids)})"

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - commands must never stop the pipeline
                log.warning("telegram commands: %s", type(exc).__name__)
                self._stop.wait(5.0)

    def poll_once(self) -> None:
        params = {"timeout": self.poll_timeout_s, "allowed_updates": '["message"]', "offset": self._offset}
        response = self._session.get(
            f"{API}/bot{self._token}/getUpdates", params=params, timeout=self.poll_timeout_s + 10
        )
        for update in response.json().get("result", []):
            self._offset = update["update_id"] + 1
            message = update.get("message") or {}
            chat = str(message.get("chat", {}).get("id"))
            if chat not in self.chat_ids:
                continue
            self._reply(chat, self.answer(message.get("text") or ""))

    def answer(self, text: str) -> str:
        command, _, argument = text.strip().partition(" ")
        command = command.split("@", 1)[0]
        if command == "/mode":
            mode = argument.strip()
            if mode not in ("A", "B", "cascade"):
                return "Режим: A | B | cascade"
            refused = self.controller.set_mode(mode)
            return f"Режим не змінено: {refused}" if refused else f"Режим: {mode}"
        if command == "/status":
            return status_text(self.controller.status())
        if command == "/mute":
            seconds = parse_duration(argument or "10m")
            if seconds is None:
                return "Приклад: /mute 10m (s, m, h)"
            self.controller.mute(seconds)
            period = argument or "10m"
            return f"Нові сповіщення вимкнено на {period}; закриття відкритих інцидентів надходитимуть."
        return HELP

    def _reply(self, chat: str, text: str) -> None:
        self._session.post(
            f"{API}/bot{self._token}/sendMessage", data={"chat_id": chat, "text": text}, timeout=10
        )
