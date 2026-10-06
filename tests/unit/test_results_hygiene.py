"""Committed result files hold numbers, ids and URLs, never document text."""

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def test_no_committed_result_holds_a_long_string():
    # the OWI licence reserves document text to its owners; a report that
    # quotes it (a generated answer) must not be committed
    long = {
        path.name: text[:60]
        for path in sorted(RESULTS.glob("*.json"))
        for text in strings(json.loads(path.read_text()))
        if len(text) > 200
    }
    assert long == {}
