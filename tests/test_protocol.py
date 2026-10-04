"""Trials protocol (session 06, step 2): the rendered protocol, without models or video."""

from sentinel.eval.protocol import render

ENV = {
    "image": "sha256:abc",
    "image_created": "2026-10-04",
    "git": "abc1234",
    "python": "3.12.7",
    "os": "Linux",
    "cpu": "Intel",
    "cpus": 16,
    "versions": {"ultralytics": "8.4.164"},
    "backend": "openvino",
    "perception": "3.1:ad09",
    "config_sha256": "c" * 64,
    "frozen_config_sha256": "f" * 64,
    "a_sha256": "a" * 64,
    "weights_sha256": {"people": "p" * 64},
}


def _row(scenario: str, expected: bool, incidents: int, rtf: float = 0.6) -> dict:
    return {
        "scenario": scenario, "clip": f"{scenario}.mp4", "expected": expected, "duration_s": 5.0, "rtf": rtf,
        "ms_mean": 50.0, "ms_p95": 80.0, "analysis_fps": 10.0, "skipped_share": 0.66, "latency_p50_ms": 60.0,
        "latency_p95_ms": 110.0, "incidents": incidents,
        "messages": ["🔴 Ймовірна бійка — Камера 1\n12:00:01"] if incidents else [],
    }  # fmt: skip


def test_protocol_has_every_section_and_checks_each_clip_against_its_scenario():
    rows = [_row("бійка", True, 1), _row("спокій", False, 0), _row("зброя", True, 0)]
    text = render(ENV, {"ok": True, "summary": "377 passed", "s": 12.0}, rows, "04.10.2026 23:30:00", False)
    for section in (
        "## 1. Мета",
        "## 2. Умови",
        "## 3. Засоби",
        "## 4. Порядок",
        "## 5. Результати",
        "## 6. Висновок",
    ):
        assert section in text
    assert "у 2 з 3 кліпів" in text and "`зброя.mp4` (зброя)" in text  # the miss is named in the conclusion
    assert "пройдено — `377 passed`" in text
    assert "🔴 Ймовірна бійка — Камера 1 · 12:00:01" in text  # one line per message


def test_a_failed_suite_or_a_slow_clip_is_not_reported_as_success():
    rows = [_row("бійка", True, 1, rtf=1.3)]
    text = render(ENV, {"ok": True, "summary": "ok", "s": 1.0}, rows, "x", False)
    assert "Вимоги не виконано повністю" in text
    text = render(ENV, {"ok": False, "summary": "1 failed", "s": 1.0}, [_row("бійка", True, 1)], "x", False)
    assert "НЕ пройдено" in text and "Вимоги не виконано повністю" in text
