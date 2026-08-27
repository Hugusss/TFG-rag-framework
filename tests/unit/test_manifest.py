"""Unit tests for the metrics manifest and result-file writing."""

import json

from rag_framework.config import load_config
from rag_framework.metrics.manifest import run_manifest, write_report


def test_write_report_never_overwrites(tmp_path):
    first = write_report({"run": 1}, tmp_path, "test")
    second = write_report({"run": 2}, tmp_path, "test")
    assert first != second
    assert json.loads(first.read_text())["run"] == 1
    assert json.loads(second.read_text())["run"] == 2


def test_manifest_carries_rule9_fields(tmp_path):
    import yaml

    config_file = tmp_path / "c.yaml"
    config_file.write_text(
        yaml.safe_dump(
            {
                "dataset": {"loader": "owi", "path": "p", "version": "corpus-v1"},
                "chunking": {"strategy": "publisher_offsets"},
                "embedding": {"provider": "precomputed", "model": "m"},
                "vector_store": {"type": "chroma", "path": "s", "collection": "c"},
                "retrieval": {"mode": "sequential"},
                "metrics": {"output": "r"},
            }
        ),
        encoding="utf-8",
    )
    manifest = run_manifest(load_config(config_file))
    assert set(manifest) >= {
        "timestamp",
        "git_commit",
        "dataset_version",
        "machine",
        "versions",
        "config",
    }
    assert manifest["dataset_version"] == "corpus-v1"
    assert manifest["versions"]["rag-framework"]
