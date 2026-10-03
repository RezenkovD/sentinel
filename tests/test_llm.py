"""Aggregator B with a fake HTTP session: no network, no keys (session 05, step 2)."""

import json
from pathlib import Path

from sentinel.aggregators.llm import (
    BAggregator,
    BWorker,
    LlmCache,
    LlmClient,
    canonical_json,
    render_prompt,
)
from sentinel.config import DEFAULT_CONFIG, load_config
from sentinel.schema import Person, SceneDescription, Summary, Window, Zone

CONFIG = load_config(DEFAULT_CONFIG)
STATS = {
    "fields": {"persons.speed": {"p50": 0.15, "p90": 0.57, "p99": 1.32}},
    "shares": {"persons.holds": 0.001},
}


def _scene(t: float = 0.0, speed: float = 0.4) -> SceneDescription:
    return SceneDescription(
        window=Window(t_start=t, t_end=t + 1.0, source="cam-1"),
        zone=Zone(type="street", hours="day", expected_density="medium"),
        summary=Summary(n_persons=1, crowd_motion=speed, weapon_visible=False),
        persons=(Person(id=3, speed=speed, limb_accel=None, holds=None, conf=0.8),),
    )


class FakeResponse:
    def __init__(self, status: int, body: dict | None = None, headers: dict | None = None, text: str = ""):
        self.status_code, self._body, self.headers, self.text = status, body or {}, headers or {}, text

    def json(self) -> dict:
        return self._body


def _ok(content: str, reasoning: int = 40) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 900,
                "completion_tokens": 120,
                "completion_tokens_details": {"reasoning_tokens": reasoning},
            },
        },
    )


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses, self.calls = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002 - mirrors requests
        self.calls.append(json)
        return self.responses.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.t, self.slept = 1000.0, []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


GOOD = json.dumps({"risk": 0.82, "level": "alarm", "reason_uk": "Людина 3 б'є людину 4"}, ensure_ascii=False)


def _b(
    tmp_path: Path,
    responses: list[FakeResponse],
    wait: bool = True,
    clock: FakeClock | None = None,
    **limits,
):
    clock = clock or FakeClock()
    session = FakeSession(responses)
    provider = CONFIG.llm.providers["groq"].model_copy(update=limits)
    client = LlmClient(provider, "test-key", session=session, clock=clock, sleep=clock.sleep)
    cache = LlmCache(tmp_path, provider.model, "b_v1")
    return BAggregator(CONFIG, client, cache, "groq", stats=STATS, wait_on_limits=wait), session, clock


def test_the_scene_goes_to_b_as_canonical_json_without_aux():
    text = canonical_json(_scene())
    assert text == json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert '"speed":0.4' in text and "aux" not in text


def test_the_prompt_carries_its_reference_values():
    prompt = render_prompt("b_v1", STATS)
    assert "persons.speed: 0.15 / 0.57 / 1.32" in prompt and "persons.holds: 0.1 %" in prompt
    assert "{reference}" not in prompt


def test_a_valid_answer_is_parsed_recorded_and_answered_from_the_cache_next_time(tmp_path: Path):
    b, session, _ = _b(tmp_path, [_ok(GOOD)])
    first = b.answer(_scene())
    assert first.status == "ok" and first.result.risk == 0.82 and first.result.level == "alarm"
    assert first.usage == {"input": 900, "output": 120, "reasoning": 40} and not first.cached
    request = session.calls[0]
    assert request["temperature"] == 0.0 and request["reasoning_effort"] == "low"
    assert request["response_format"]["type"] == "json_schema"
    again = b.answer(_scene())  # same scene: no second request
    assert again.cached and again.result == first.result and len(session.calls) == 1
    record = json.loads((tmp_path / "openai_gpt-oss-120b" / "b_v1.jsonl").read_text().splitlines()[0])
    assert record["provider"] == "groq" and record["usage"]["reasoning"] == 40 and "latency_s" in record


def test_an_invalid_answer_is_retried_once_with_the_error_then_counted_invalid(tmp_path: Path):
    bad = json.dumps({"risk": True, "level": "alarm", "reason_uk": "так"})
    b, session, _ = _b(tmp_path, [_ok(bad), _ok(GOOD)])
    assert b.answer(_scene()).status == "ok" and len(session.calls) == 2
    assert "risk" in session.calls[1]["messages"][-1]["content"]  # the error goes back to the model
    record = json.loads((tmp_path / "openai_gpt-oss-120b" / "b_v1.jsonl").read_text().splitlines()[0])
    assert record["attempts"] == 2 and "risk" in record["errors"][0] and record["error"] is None
    long = json.dumps({"risk": 0.5, "level": "attention", "reason_uk": "а" * 201}, ensure_ascii=False)
    b2, _, _ = _b(tmp_path / "2", [_ok(long), _ok("не JSON")])
    answer = b2.answer(_scene())
    assert answer.status == "invalid" and answer.result is None


def test_a_rate_limit_is_waited_out_in_the_experiment_and_gives_up_in_live(tmp_path: Path):
    limited = FakeResponse(429, headers={"retry-after": "7"})
    b, session, clock = _b(tmp_path, [limited, _ok(GOOD)])
    assert b.answer(_scene()).status == "ok" and 7.0 in clock.slept
    live, _, live_clock = _b(tmp_path / "live", [limited], wait=False)
    answer = live.answer(_scene())
    assert answer.status == "unavailable" and live_clock.slept == []


def test_requests_per_minute_are_spread_out_before_the_provider_says_no(tmp_path: Path):
    b, _, clock = _b(tmp_path, [_ok(GOOD) for _ in range(31)], tpm=None, tpd=None)
    for i in range(30):
        b.answer(_scene(float(i)))
    assert clock.slept == []
    b.answer(_scene(30.0))
    assert sum(clock.slept) >= 59.0  # the 31st request of a 30-per-minute plan waits for the window


def test_tokens_per_minute_are_kept_too(tmp_path: Path):
    b, _, clock = _b(tmp_path, [_ok(GOOD) for _ in range(9)], rpm=None, tpd=None)  # 1020 tokens a call
    for i in range(9):
        b.answer(_scene(float(i)))
    assert sum(clock.slept) >= 59.0  # 8000 tokens a minute: the 8th call waits


def test_live_does_not_wait_for_a_full_minute(tmp_path: Path):
    b, _, clock = _b(tmp_path, [_ok(GOOD)], wait=False, rpm=1, tpm=None, tpd=None)
    assert b.answer(_scene(0.0)).status == "ok"
    assert b.answer(_scene(1.0)).status == "unavailable" and clock.slept == []


def test_structured_output_falls_back_to_a_json_object_if_the_provider_refuses_it(tmp_path: Path):
    refused = FakeResponse(400, text='{"error": {"message": "response_format json_schema is not supported"}}')
    b, session, _ = _b(tmp_path, [refused, _ok(GOOD)])
    assert b.answer(_scene()).status == "ok"
    assert session.calls[1]["response_format"] == {"type": "json_object"}


def test_the_cache_key_depends_on_prompt_model_provider_and_scene(tmp_path: Path):
    b, _, _ = _b(tmp_path, [])
    key = b.key(_scene())
    assert key != b.key(_scene(speed=0.5))
    assert key != BAggregator(CONFIG, b.client, b.cache, "cerebras", stats=STATS).key(_scene())
    b.prompt_version = "b_v2"
    assert key != b.key(_scene())


class SlowB:
    """Stands for BAggregator in the worker tests: answers after `delay` fake seconds."""

    def __init__(self, clock: FakeClock, delay: float) -> None:
        self.clock, self.delay, self.seen = clock, delay, []

    def answer(self, scene):
        from sentinel.aggregators.llm import BAnswer
        from sentinel.schema import RiskResult

        self.seen.append(scene.window.t_start)
        self.clock.t += self.delay
        result = RiskResult(risk=0.9, level="alarm", reason_uk="бійка")
        return BAnswer("ok", result, None, {}, self.delay, "groq", "m", "b_v1", False, None)


def test_the_live_queue_keeps_the_two_newest_scenes():
    clock = FakeClock()
    worker = BWorker(SlowB(clock, 0.5), maxlen=2, timeout_s=8.0, clock=clock, start=False)
    for t in (0.0, 1.0, 2.0):
        worker.submit(_scene(t))
    assert worker.dropped == 1 and [s.window.t_start for s, _ in worker.pending()] == [1.0, 2.0]
    worker.run_once()
    worker.run_once()
    assert [round(s.window.t_start) for s, _ in worker.take()] == [1, 2]


def test_an_answer_later_than_the_timeout_is_not_used():
    clock = FakeClock()
    worker = BWorker(SlowB(clock, 9.0), maxlen=2, timeout_s=8.0, clock=clock, start=False)
    worker.submit(_scene())
    worker.run_once()
    ((scene, answer),) = worker.take()
    assert answer.status == "timeout" and answer.result is None
