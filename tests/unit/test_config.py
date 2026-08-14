"""Unit tests for the configuration seam: parsing and fail-fast validation."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from rag_framework.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]


def base() -> dict:
    """A complete, valid configuration; tests mutate copies of it."""
    return {
        "dataset": {"loader": "owi", "path": "./data/owi"},
        "chunking": {
            "strategy": "recursive",
            "target_tokens": 500,
            "overlap_tokens": 50,
            "minimum_tokens": 50,
        },
        "embedding": {
            "provider": "local",
            "model": "jinaai/jina-embeddings-v5-text-small",
            "batch_size": 64,
            "normalize": True,
        },
        "vector_store": {
            "type": "chroma",
            "path": "./state/chroma",
            "collection": "owi-main",
        },
        "retrieval": {"mode": "sequential", "k": 10, "partitions": 1, "workers": 1},
        "generation": {"enabled": False, "provider": "mock"},
        "metrics": {"output": "./results"},
    }


def write(tmp_path: Path, data) -> Path:
    file = tmp_path / "config.yaml"
    file.write_text(yaml.safe_dump(data), encoding="utf-8")
    return file


def load(tmp_path: Path, data):
    return load_config(write(tmp_path, data))


class TestHappyPath:
    def test_full_config_parses_into_typed_values(self, tmp_path):
        config = load(tmp_path, base())
        assert config.dataset.loader == "owi"
        assert config.chunking.target_tokens == 500
        assert config.embedding.normalize is True
        assert config.vector_store.collection == "owi-main"
        assert config.retrieval.k == 10
        assert config.generation.enabled is False
        assert config.metrics.output == "./results"

    def test_shipped_local_chroma_config_parses(self):
        config = load_config(REPO_ROOT / "configs" / "local_chroma.yaml")
        assert config.chunking.strategy == "publisher_offsets"
        assert config.embedding.provider == "precomputed"
        assert config.retrieval.mode == "sequential"
        assert config.generation.enabled is False

    def test_dataset_version_optional_for_reproducibility(self, tmp_path):
        data = base()
        assert load(tmp_path, data).dataset.version is None
        data["dataset"]["version"] = "owi-v2.0.0-gpu-spa-2026-07-28"
        config = load(tmp_path, data)
        assert config.dataset.version == "owi-v2.0.0-gpu-spa-2026-07-28"

    def test_generation_section_is_optional_with_spec_defaults(self, tmp_path):
        data = base()
        del data["generation"]
        config = load(tmp_path, data)
        assert config.generation.enabled is False
        assert config.generation.provider == "mock"

    def test_config_objects_are_frozen(self, tmp_path):
        config = load(tmp_path, base())
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.retrieval.k = 99


class TestFileAndSyntaxErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_invalid_yaml_syntax(self, tmp_path):
        file = tmp_path / "config.yaml"
        file.write_text("dataset: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_config(file)

    def test_top_level_must_be_mapping(self, tmp_path):
        file = tmp_path / "config.yaml"
        file.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(file)

    def test_duplicate_keys_are_rejected_not_last_wins(self, tmp_path):
        text = yaml.safe_dump(base()).replace(
            "retrieval:", "retrieval:\n  k: 99", 1
        )
        file = tmp_path / "config.yaml"
        file.write_text(text, encoding="utf-8")  # k appears twice
        with pytest.raises(ConfigError, match="duplicate key: k"):
            load_config(file)

    def test_non_string_section_name_reports_config_error(self, tmp_path):
        # a stray bare-number line becomes an int key in YAML; this must
        # be a named ConfigError, never a TypeError traceback
        file = tmp_path / "config.yaml"
        file.write_text(
            yaml.safe_dump(base()) + "2024:\n  x: 1\n", encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="2024"):
            load_config(file)

    def test_non_string_key_inside_section_reports_config_error(self, tmp_path):
        data = base()
        data["retrieval"][5] = 1
        data["retrieval"]["partitons"] = 4  # mixed types must still sort
        with pytest.raises(ConfigError, match="unknown key"):
            load(tmp_path, data)

    def test_section_with_null_body_rejected(self, tmp_path):
        data = base()
        del data["generation"]
        file = tmp_path / "config.yaml"
        # "generation:" with no body parses as a null value, not a mapping
        file.write_text(yaml.safe_dump(data) + "generation:\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="generation"):
            load_config(file)

    def test_scalar_section_rejected(self, tmp_path):
        data = base()
        data["dataset"] = "hello"
        with pytest.raises(ConfigError, match="dataset"):
            load(tmp_path, data)

    def test_explicit_null_is_rejected_like_wrong_type(self, tmp_path):
        # optional keys are expressed by omission, never by "key: null"
        data = base()
        data["embedding"]["batch_size"] = None
        with pytest.raises(ConfigError, match="embedding.batch_size"):
            load(tmp_path, data)


class TestStructuralErrorsNameTheKey:
    def test_missing_section(self, tmp_path):
        data = base()
        del data["dataset"]
        with pytest.raises(ConfigError, match="dataset"):
            load(tmp_path, data)

    def test_unknown_section(self, tmp_path):
        data = base()
        data["datset"] = {"loader": "owi"}  # typo must fail loudly
        with pytest.raises(ConfigError, match="datset"):
            load(tmp_path, data)

    def test_missing_key(self, tmp_path):
        data = base()
        del data["dataset"]["path"]
        with pytest.raises(ConfigError, match="dataset.path"):
            load(tmp_path, data)

    def test_unknown_key(self, tmp_path):
        data = base()
        data["retrieval"]["partitons"] = 4  # typo must not silently no-op
        with pytest.raises(ConfigError, match="retrieval.partitons"):
            load(tmp_path, data)

    def test_wrong_type_is_never_coerced(self, tmp_path):
        data = base()
        data["retrieval"]["k"] = "10"
        with pytest.raises(ConfigError, match="retrieval.k"):
            load(tmp_path, data)

    def test_bool_does_not_pass_as_int(self, tmp_path):
        data = base()
        data["retrieval"]["k"] = True
        with pytest.raises(ConfigError, match="retrieval.k"):
            load(tmp_path, data)

    def test_empty_string_rejected(self, tmp_path):
        data = base()
        data["dataset"]["loader"] = "  "
        with pytest.raises(ConfigError, match="dataset.loader"):
            load(tmp_path, data)


class TestChunkingShapes:
    def test_publisher_offsets_rejects_token_sizes(self, tmp_path):
        data = base()
        data["chunking"] = {"strategy": "publisher_offsets", "target_tokens": 500}
        with pytest.raises(ConfigError, match="not applicable"):
            load(tmp_path, data)

    def test_recursive_requires_all_token_sizes(self, tmp_path):
        data = base()
        del data["chunking"]["overlap_tokens"]
        with pytest.raises(ConfigError, match="chunking.overlap_tokens"):
            load(tmp_path, data)

    def test_overlap_must_be_smaller_than_target(self, tmp_path):
        data = base()
        data["chunking"]["overlap_tokens"] = 500
        with pytest.raises(ConfigError, match="overlap_tokens"):
            load(tmp_path, data)

    def test_unknown_strategy(self, tmp_path):
        data = base()
        data["chunking"] = {"strategy": "semantic"}
        with pytest.raises(ConfigError, match="chunking.strategy"):
            load(tmp_path, data)

    def test_spec_short_example_without_minimum_tokens_is_rejected(self, tmp_path):
        # Documented deviation (ADR-004): the spec's section-15 example
        # omits minimum_tokens, but its section-9 example includes it and
        # an experiment must state its minimum chunk size explicitly.
        data = base()
        del data["chunking"]["minimum_tokens"]
        with pytest.raises(ConfigError, match="chunking.minimum_tokens"):
            load(tmp_path, data)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("target_tokens", 0, "positive"),
            ("overlap_tokens", -1, "negative"),
            ("minimum_tokens", -1, "negative"),
            ("minimum_tokens", 501, "exceed"),
        ],
    )
    def test_invalid_token_counts(self, tmp_path, key, value, message):
        data = base()
        data["chunking"][key] = value
        with pytest.raises(ConfigError, match=message):
            load(tmp_path, data)


class TestRetrievalShapes:
    def test_unknown_mode(self, tmp_path):
        data = base()
        data["retrieval"]["mode"] = "parallel"
        with pytest.raises(ConfigError, match="retrieval.mode"):
            load(tmp_path, data)

    def test_sequential_forbids_multiple_partitions(self, tmp_path):
        data = base()
        data["retrieval"]["partitions"] = 4
        with pytest.raises(ConfigError, match="sequential"):
            load(tmp_path, data)

    def test_sequential_forbids_multiple_workers(self, tmp_path):
        data = base()
        data["retrieval"]["workers"] = 4
        with pytest.raises(ConfigError, match="sequential"):
            load(tmp_path, data)

    @pytest.mark.parametrize("key", ["partitions", "workers"])
    def test_partitions_and_workers_must_be_at_least_one(self, tmp_path, key):
        data = base()
        data["retrieval"]["mode"] = "collective"
        data["retrieval"][key] = 0
        with pytest.raises(ConfigError, match=f"retrieval.{key}"):
            load(tmp_path, data)

    def test_batch_size_must_be_positive(self, tmp_path):
        data = base()
        data["embedding"]["batch_size"] = 0
        with pytest.raises(ConfigError, match="embedding.batch_size"):
            load(tmp_path, data)

    def test_collective_allows_partitions_and_workers(self, tmp_path):
        data = base()
        data["retrieval"] = {"mode": "collective", "k": 10, "partitions": 4, "workers": 2}
        config = load(tmp_path, data)
        assert config.retrieval.partitions == 4
        assert config.retrieval.workers == 2

    def test_k_must_be_positive(self, tmp_path):
        data = base()
        data["retrieval"]["k"] = 0
        with pytest.raises(ConfigError, match="retrieval.k"):
            load(tmp_path, data)
