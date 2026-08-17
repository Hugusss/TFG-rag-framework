"""Integrity tests for the committed evaluation query set.

These validate the file itself (schema, invariants) and never touch
the dataset, per the no-full-corpus rule for tests.
"""

import json
from pathlib import Path

QUERIES = Path(__file__).resolve().parents[2] / "evaluation" / "queries.jsonl"


def load_entries():
    lines = QUERIES.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines]


def test_thirty_queries_with_unique_ids():
    entries = load_entries()
    assert len(entries) == 30
    ids = [e["query_id"] for e in entries]
    assert len(set(ids)) == 30


def test_required_fields_present():
    for entry in load_entries():
        assert set(entry) >= {
            "query_id", "query", "relevant_document_ids", "notes", "answerable"
        }
        assert entry["query"].strip()
        assert entry["notes"].strip()
        assert isinstance(entry["answerable"], bool)


def test_answerability_matches_relevance():
    for entry in load_entries():
        if entry["answerable"]:
            assert entry["relevant_document_ids"], entry["query_id"]
        else:
            assert entry["relevant_document_ids"] == [], entry["query_id"]


def test_relevant_ids_are_wellformed_and_deduplicated():
    for entry in load_entries():
        ids = entry["relevant_document_ids"]
        assert len(ids) == len(set(ids)), entry["query_id"]
        for document_id in ids:
            assert len(document_id) == 64, entry["query_id"]
            int(document_id, 16)  # sha-256 hex, like every corpus id


def test_paraphrase_pairs_share_their_targets():
    entries = {e["query_id"]: e for e in load_entries()}
    pairs = [e for e in entries.values() if "paraphrase_of" in e]
    assert len(pairs) == 3
    for entry in pairs:
        original = entries[entry["paraphrase_of"]]
        assert entry["relevant_document_ids"] == original["relevant_document_ids"]


def test_type_mix_counts():
    entries = load_entries()
    unanswerable = [e for e in entries if not e["answerable"]]
    multi = [e for e in entries if len(e["relevant_document_ids"]) > 1]
    assert len(unanswerable) == 4
    assert len(multi) >= 4  # the multi-document queries (q014-q017, q019)
