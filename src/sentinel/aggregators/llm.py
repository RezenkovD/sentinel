"""Aggregator B: an LLM reads the scene description as text, answers strict JSON (SPEC §3–4, 01.10).

Input — exactly the `SceneDescription` A gets, as canonical JSON (sorted keys, the schema's rounding); no
aux, no images. The prompt is versioned (`prompts/<version>.md`) and carries only the usual values of the
fields on calm windows of our train (`prompts/reference_stats.json`): B is zero-shot with respect to labels.

Output — `{risk, level, reason_uk}` validated by `RiskResult`, plus reason_uk ≤ 200 characters; an invalid
answer is retried once with the error, then counted as `invalid`. Every answer (ok or invalid) goes into the
cache `cache/llm/<model>/<prompt version>.jsonl` with its tokens, latency, provider and time, keyed by
sha256(prompt version, model, provider, scene JSON).

Rate limits come from the config (requests and tokens per minute are spread out here; a 429 waits as the
provider asks). In the experiment B waits and retries — a missing answer is `unavailable` and counted, never
replaced by A. In `live` it never waits: `BWorker` runs it in the background with a queue of at most two
scenes (the newest pushes out the oldest), and an answer later than the timeout is dropped so the window
keeps A.
"""

import hashlib
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from sentinel.config import Config, LlmProviderConfig
from sentinel.schema import RiskResult, SceneDescription

PROMPT_VERSION = "b_v1"
PROMPTS = Path(__file__).parent / "prompts"
REASON_MAX = 200
SCHEMA = {
    "type": "object",
    "properties": {
        "risk": {"type": "number", "minimum": 0, "maximum": 1},
        "level": {"type": "string", "enum": ["normal", "attention", "alarm"]},
        "reason_uk": {"type": "string"},
    },
    "required": ["risk", "level", "reason_uk"],
    "additionalProperties": False,
}


def canonical_json(scene: SceneDescription) -> str:
    return json.dumps(
        scene.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def load_reference_stats() -> dict:
    return json.loads((PROMPTS / "reference_stats.json").read_text(encoding="utf-8"))


def render_prompt(version: str, stats: dict) -> str:
    lines = [f"- {k}: {v['p50']:g} / {v['p90']:g} / {v['p99']:g}" for k, v in stats["fields"].items()]
    lines += [f"- {k}: {100 * v:.1f} %" for k, v in stats["shares"].items()]
    return (PROMPTS / f"{version}.md").read_text(encoding="utf-8").replace("{reference}", "\n".join(lines))


@dataclass
class BAnswer:
    status: str  # ok | invalid | unavailable | timeout
    result: RiskResult | None
    raw: str | None
    usage: dict  # {"input", "output", "reasoning"} tokens of all requests for this answer
    latency_s: float
    provider: str
    model: str
    prompt_version: str
    cached: bool
    error: str | None


class RateLimited(Exception):
    def __init__(self, retry_after: float | None) -> None:
        super().__init__(f"rate limited (retry after {retry_after})")
        self.retry_after = retry_after


class LlmError(Exception):
    pass


class _Window:
    """Requests and tokens of the last `span_s` seconds."""

    def __init__(self, span_s: float) -> None:
        self.span_s, self.items = span_s, deque()

    def _trim(self, now: float) -> None:
        while self.items and self.items[0][0] <= now - self.span_s:
            self.items.popleft()

    def count(self, now: float) -> int:
        self._trim(now)
        return len(self.items)

    def tokens(self, now: float) -> int:
        self._trim(now)
        return sum(n for _, n in self.items)

    def wait_for(self, now: float) -> float:
        return self.items[0][0] + self.span_s - now if self.items else 0.0


class LlmClient:
    """One OpenAI-compatible provider over `requests`, keeping under its per-minute and per-day limits."""

    def __init__(
        self,
        provider: LlmProviderConfig,
        api_key: str,
        session=None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        import requests

        self.provider, self.api_key = provider, api_key
        self.session = session or requests.Session()
        self.clock, self.sleep = clock, sleep
        self.minute, self.day = _Window(60.0), _Window(86400.0)
        self.lock = threading.Lock()

    def room(self, tokens: int) -> float:
        """Seconds to wait before a request of about `tokens` fits the limits (0: now)."""
        p, now = self.provider, self.clock()
        waits = [0.0]
        if p.rpm and self.minute.count(now) >= p.rpm:
            waits.append(self.minute.wait_for(now))
        if p.tpm and self.minute.tokens(now) + tokens > p.tpm and self.minute.items:
            waits.append(self.minute.wait_for(now))
        if (p.rpd and self.day.count(now) >= p.rpd) or (p.tpd and self.day.tokens(now) + tokens > p.tpd):
            waits.append(self.day.wait_for(now))
        return max(waits)

    def chat(self, body: dict) -> tuple[dict, float]:
        """POST /chat/completions -> (JSON, latency s); 429 -> RateLimited, other errors -> LlmError."""
        started = self.clock()
        response = self.session.post(
            f"{self.provider.base_url}/chat/completions",
            json=body | {"model": self.provider.model},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60,
        )
        latency = self.clock() - started
        if response.status_code == 429:
            retry = response.headers.get("retry-after")
            raise RateLimited(float(retry) if retry else None)
        if response.status_code != 200:
            raise LlmError(f"HTTP {response.status_code}: {response.text[:300]}")
        data = response.json()
        used = data.get("usage", {})
        now = self.clock()
        with self.lock:
            self.minute.items.append((now, int(used.get("total_tokens") or 0) or self._sum(used)))
            self.day.items.append((now, self.minute.items[-1][1]))
        return data, latency

    @staticmethod
    def _sum(used: dict) -> int:
        return int(used.get("prompt_tokens") or 0) + int(used.get("completion_tokens") or 0)


class LlmCache:
    """Append-only JSONL of every answer; read back whole on start."""

    def __init__(self, root: Path, model: str, prompt_version: str) -> None:
        self.path = Path(root) / model.replace("/", "_") / f"{prompt_version}.jsonl"
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["key"]] = record
        self.lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, record: dict) -> None:
        with self.lock:
            self.records[record["key"]] = record
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())


def validate(text: str) -> RiskResult:
    """The strict output contract of B; raises ValueError with a message the model can act on."""
    try:
        result = RiskResult.model_validate_json(text)
    except ValidationError as e:
        raise ValueError(f"the answer is not the required JSON: {e.errors(include_url=False)}") from e
    if len(result.reason_uk) > REASON_MAX:
        raise ValueError(
            f"reason_uk has {len(result.reason_uk)} characters, at most {REASON_MAX} are allowed"
        )
    return result


class BAggregator:
    def __init__(
        self,
        config: Config,
        client: LlmClient,
        cache: LlmCache,
        provider: str,
        stats: dict | None = None,
        prompt_version: str = PROMPT_VERSION,
        wait_on_limits: bool = True,
        max_waits: int = 20,
    ) -> None:
        self.config, self.client, self.cache, self.provider = config, client, cache, provider
        self.prompt_version, self.wait_on_limits, self.max_waits = prompt_version, wait_on_limits, max_waits
        self.model = config.llm.providers[provider].model
        self.prompt = render_prompt(prompt_version, stats if stats is not None else load_reference_stats())
        self.structured = True  # json_schema until the provider refuses it

    def key(self, scene: SceneDescription) -> str:
        text = "\n".join([self.prompt_version, self.model, self.provider, canonical_json(scene)])
        return hashlib.sha256(text.encode()).hexdigest()

    def _body(self, messages: list[dict]) -> dict:
        llm = self.config.llm
        body = {
            "messages": messages,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "reasoning_effort": llm.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "risk_result", "strict": True, "schema": SCHEMA},
            }
            if self.structured
            else {"type": "json_object"},
        }
        if llm.seed is not None:
            body["seed"] = llm.seed
        return body

    def _request(self, messages: list[dict], estimate: int) -> tuple[dict, float]:
        """One request, waiting for the limits (experiment) or giving up at once (live)."""
        for _ in range(self.max_waits):
            wait = self.client.room(estimate)
            if wait > 0:
                if not self.wait_on_limits:
                    raise RateLimited(wait)
                self.client.sleep(wait)
                continue
            try:
                return self.client.chat(self._body(messages))
            except RateLimited as e:
                if not self.wait_on_limits:
                    raise
                self.client.sleep(e.retry_after or 10.0)
            except LlmError as e:
                if self.structured and ("response_format" in str(e) or "json_schema" in str(e)):
                    self.structured = False  # this provider/model has no structured output: JSON mode
                    continue
                raise
        raise RateLimited(None)

    def answer(self, scene: SceneDescription, use_cache: bool = True) -> BAnswer:
        key = self.key(scene)
        if use_cache and (record := self.cache.get(key)) is not None:
            result = RiskResult.model_validate(record["result"]) if record["result"] else None
            return BAnswer(
                record["status"], result, record["raw"], record["usage"], record["latency_s"],
                self.provider, self.model, self.prompt_version, True, record.get("error"),
            )  # fmt: skip
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": canonical_json(scene)},
        ]
        usage = {"input": 0, "output": 0, "reasoning": 0}
        latency, raw, error, result, errors, attempts = 0.0, None, None, None, [], 0
        try:
            for attempt in range(2):
                attempts += 1
                data, took = self._request(messages, estimate=len(self.prompt) // 3 + 600)
                latency += took
                used = data.get("usage", {})
                usage["input"] += int(used.get("prompt_tokens") or 0)
                usage["output"] += int(used.get("completion_tokens") or 0)
                details = used.get("completion_tokens_details") or {}
                usage["reasoning"] += int(details.get("reasoning_tokens") or 0)
                raw = data["choices"][0]["message"].get("content") or ""
                try:
                    result, error = validate(raw), None
                    break
                except ValueError as e:
                    error = str(e)
                    errors.append(error)
                    if attempt == 0:  # one retry, telling the model what was wrong
                        messages = messages + [
                            {"role": "assistant", "content": raw},
                            {
                                "role": "user",
                                "content": f"Invalid answer: {error}. Reply again with the JSON only.",
                            },
                        ]
        except (RateLimited, LlmError) as e:
            return BAnswer(
                "unavailable", None, raw, usage, latency, self.provider, self.model, self.prompt_version,
                False, str(e),
            )  # fmt: skip
        status = "ok" if result is not None else "invalid"
        self.cache.put(
            {
                "key": key,
                "status": status,
                "result": result.model_dump(mode="json") if result else None,
                "raw": raw,
                "error": error,
                "attempts": attempts,
                "errors": errors,
                "usage": usage,
                "latency_s": round(latency, 3),
                "provider": self.provider,
                "model": self.model,
                "prompt_version": self.prompt_version,
                "time": datetime.now().isoformat(timespec="seconds"),
            }
        )
        return BAnswer(
            status, result, raw, usage, latency, self.provider, self.model, self.prompt_version, False, error
        )


def make_b(
    config: Config, wait_on_limits: bool, provider: str | None = None, prompt_version: str = PROMPT_VERSION
):
    """B for the configured provider; the key comes from the environment (.env), never from the config."""
    from dotenv import load_dotenv

    load_dotenv()
    name = provider or config.llm.provider
    p = config.llm.providers[name]
    key = os.environ.get(p.key_env)
    if not key:
        raise RuntimeError(f"{p.key_env} is not set in .env: aggregator B ({name}) cannot run")
    client = LlmClient(p, key)
    cache = LlmCache(config.llm.cache_dir, p.model, prompt_version)
    return BAggregator(
        config, client, cache, name, prompt_version=prompt_version, wait_on_limits=wait_on_limits
    )


class BWorker:
    """B in the background for `live`: at most `maxlen` scenes wait (a newer one pushes out the oldest), the
    pipeline never waits, and an answer later than `timeout_s` after its scene was queued is dropped."""

    def __init__(self, b, maxlen: int = 2, timeout_s: float = 8.0, clock=time.monotonic, start: bool = True):
        self.b, self.maxlen, self.timeout_s, self.clock = b, maxlen, timeout_s, clock
        self.queue: deque[tuple[SceneDescription, float]] = deque()
        self.done: deque[tuple[SceneDescription, BAnswer]] = deque()
        self.dropped = 0
        self.busy = False
        self.cond = threading.Condition()
        self.stop = False
        if start:
            threading.Thread(target=self._loop, name="aggregator-b", daemon=True).start()

    def submit(self, scene: SceneDescription) -> None:
        with self.cond:
            self.queue.append((scene, self.clock()))
            while len(self.queue) > self.maxlen:
                self.queue.popleft()
                self.dropped += 1
            self.cond.notify()

    def pending(self) -> list[tuple[SceneDescription, float]]:
        with self.cond:
            return list(self.queue)

    def take(self) -> list[tuple[SceneDescription, BAnswer]]:
        """Answers that arrived since the last call (in order of arrival)."""
        with self.cond:
            out, self.done = list(self.done), deque()
            return out

    def state(self) -> dict:
        with self.cond:
            return {"queued": len(self.queue), "busy": self.busy, "dropped": self.dropped}

    def run_once(self) -> None:
        with self.cond:
            if not self.queue:
                return
            scene, queued = self.queue.popleft()
            self.busy = True
        try:
            answer = self.b.answer(scene)
        except Exception as e:  # noqa: BLE001 - B must never stop the pipeline
            answer = BAnswer("unavailable", None, None, {}, 0.0, "", "", "", False, repr(e))
        if self.clock() - queued > self.timeout_s:
            answer = BAnswer(
                "timeout", None, answer.raw, answer.usage, answer.latency_s, answer.provider, answer.model,
                answer.prompt_version, answer.cached, f"answered after {self.clock() - queued:.1f} s",
            )  # fmt: skip
        with self.cond:
            self.busy = False
            self.done.append((scene, answer))

    def _loop(self) -> None:
        while not self.stop:
            with self.cond:
                while not self.queue and not self.stop:
                    self.cond.wait(timeout=1.0)
            self.run_once()
