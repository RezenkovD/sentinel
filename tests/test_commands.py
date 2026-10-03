"""Telegram bot commands over getUpdates long polling, with a fake HTTP session (session 05, step 6)."""

from sentinel.notify.commands import BotCommands, parse_duration


class FakeResponse:
    def __init__(self, body: dict) -> None:
        self.status_code, self._body = 200, body

    def json(self) -> dict:
        return self._body


class FakeSession:
    def __init__(self, updates: list[list[dict]]) -> None:
        self.updates, self.posts, self.offsets = list(updates), [], []

    def get(self, url, params=None, timeout=None):
        self.offsets.append(params.get("offset"))
        return FakeResponse({"ok": True, "result": self.updates.pop(0) if self.updates else []})

    def post(self, url, data=None, timeout=None):
        self.posts.append((url.rsplit("/", 1)[1], data))
        return FakeResponse({"ok": True, "result": {"message_id": 1}})


class Controller:
    def __init__(self) -> None:
        self.mode, self.muted_until = "A", None

    def set_mode(self, mode: str) -> str | None:
        self.mode = mode
        return None

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "rtf": 0.98,
            "fps": 9.7,
            "open_incidents": 1,
            "b_queue": {"queued": 1, "busy": True, "dropped": 3},
        }

    def mute(self, seconds: float) -> None:
        self.muted_until = seconds


def _update(uid: int, chat: int, text: str) -> dict:
    return {"update_id": uid, "message": {"chat": {"id": chat}, "text": text}}


def _bot(updates):
    session, controller = FakeSession(updates), Controller()
    return BotCommands("token", ["111"], controller, session=session, start=False), session, controller


def test_mode_status_and_mute_answer_only_allowed_chats():
    bot, session, controller = _bot(
        [[_update(5, 111, "/mode B"), _update(6, 999, "/mode cascade"), _update(7, 111, "/status")]]
    )
    bot.poll_once()
    assert controller.mode == "B"  # the stranger's /mode cascade is ignored
    replies = [d["text"] for method, d in session.posts if method == "sendMessage"]
    assert all(d["chat_id"] == "111" for _, d in session.posts)
    assert replies[0] == "Режим: B" and "RTF 0,98" in replies[1] and "черга B: 1" in replies[1]
    bot.poll_once()
    assert session.offsets == [None, 8]  # the next poll confirms everything seen so far


def test_mute_takes_a_duration_and_unknown_commands_get_help():
    bot, session, controller = _bot([[_update(1, 111, "/mute 10m"), _update(2, 111, "/hello")]])
    bot.poll_once()
    assert controller.muted_until == 600
    assert "/mode" in session.posts[-1][1]["text"]


def test_a_bad_mode_is_refused():
    bot, session, controller = _bot([[_update(1, 111, "/mode C")]])
    bot.poll_once()
    assert controller.mode == "A" and "A | B | cascade" in session.posts[0][1]["text"]


def test_durations():
    assert (parse_duration("10m"), parse_duration("30s"), parse_duration("1h"), parse_duration("x")) == (
        600.0,
        30.0,
        3600.0,
        None,
    )
