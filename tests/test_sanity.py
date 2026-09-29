import json

from eval.sanity import window_records


def test_window_records_skip_the_end_of_stream_line():
    window = {"t": 1.0, "scene": {"summary": {"weapon_visible": False}}, "risk": {"risk": 0.2}, "events": []}
    end = {"t": 1.0, "stream_end": True, "state": "НОРМА", "events": [{"kind": "close", "incident_id": "x"}]}
    lines = [json.dumps(window), json.dumps(end, ensure_ascii=False)]
    assert window_records(lines) == [window]
