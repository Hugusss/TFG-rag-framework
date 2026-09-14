"""Unit tests for the configuration seam: parsing and fail-fast validation."""

import dataclasses
from pathlib import Path

import pytest
import yaml

from rag_framework.config import (
    ConfigError,
    layout_mode,
    load_config,
    resolved_execution,
)

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
        "retrieval": {"k": 10},
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
        assert config.vector_store.partitions is None
        assert layout_mode(config) == "sequential"
        assert config.generation.enabled is False

    def test_dataset_version_optional_for_reproducibility(self, tmp_path):
        data = base()
        assert load(tmp_path, data).dataset.version is None
        data["dataset"]["version"] = "owi-v2.0.0-gpu-spa-2026-07-28"
        config = load(tmp_path, data)
        assert config.dataset.version == "owi-v2.0.0-gpu-spa-2026-07-28"

    def test_vector_store_ef_search_optional_and_validated(self, tmp_path):
        data = base()
        assert load(tmp_path, data).vector_store.ef_search is None
        data["vector_store"]["ef_search"] = 800
        assert load(tmp_path, data).vector_store.ef_search == 800
        for bad in (0, -1, True, "800"):
            data["vector_store"]["ef_search"] = bad
            with pytest.raises(ConfigError, match="vector_store.ef_search"):
                load(tmp_path, data)

    def test_dataset_sample_percent_defaults_to_full_and_validates(self, tmp_path):
        data = base()
        assert load(tmp_path, data).dataset.sample_percent == 100
        data["dataset"]["sample_percent"] = 25
        assert load(tmp_path, data).dataset.sample_percent == 25
        for bad in (0, 101, True, "50"):
            data["dataset"]["sample_percent"] = bad
            with pytest.raises(ConfigError, match="dataset.sample_percent"):
                load(tmp_path, data)

    def test_generation_section_is_optional_with_spec_defaults(self, tmp_path):
        data = base()
        del data["generation"]
        config = load(tmp_path, data)
        assert config.generation.enabled is False
        assert config.generation.provider == "mock"

    def test_ollama_generation_keys_validate(self, tmp_path):
        data = base()
        data["generation"] = {"enabled": True, "provider": "ollama", "model": "m"}
        config = load(tmp_path, data)
        assert config.generation.model == "m"
        assert config.generation.endpoint == "http://localhost:11434"
        assert config.generation.timeout_seconds == 120

        del data["generation"]["model"]
        with pytest.raises(ConfigError, match="generation.model"):
            load(tmp_path, data)

        data["generation"] = {"provider": "ollama", "model": "m", "timeout_seconds": 0}
        with pytest.raises(ConfigError, match="generation.timeout_seconds"):
            load(tmp_path, data)

        data["generation"] = {"provider": "mock", "endpoint": "http://x"}
        with pytest.raises(ConfigError, match="generation.endpoint"):
            load(tmp_path, data)

    def test_shipped_ollama_config_parses(self):
        config = load_config("configs/generation_ollama.yaml")
        assert config.generation.enabled and config.generation.provider == "ollama"
        assert config.generation.model == "ministral-3"

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
        # Deliberate strictness (ADR-004): a shorter configuration form
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


class TestEmbeddingShapes:
    def test_batch_size_must_be_positive(self, tmp_path):
        data = base()
        data["embedding"]["batch_size"] = 0
        with pytest.raises(ConfigError, match="embedding.batch_size"):
            load(tmp_path, data)


class TestIndexLayoutAndExecution:
    """ADR-014: the partition layout belongs to the index section, the
    execution settings to the query section, and the mode is derived."""

    def test_absent_partitions_is_the_monolithic_layout(self, tmp_path):
        config = load(tmp_path, base())
        assert config.vector_store.partitions is None
        assert layout_mode(config) == "sequential"
        assert resolved_execution(config) == (None, None)

    def test_one_partition_is_the_partitioned_layout(self, tmp_path):
        # P=1 is not the monolith: it takes the partitioned code path, so
        # a scaling curve's first point runs the same code as the rest
        data = base()
        data["vector_store"]["partitions"] = 1
        config = load(tmp_path, data)
        assert config.vector_store.partitions == 1
        assert layout_mode(config) == "collective"

    def test_partitions_must_be_at_least_one_and_an_int(self, tmp_path):
        data = base()
        for bad in (0, -2, True, "4"):
            data["vector_store"]["partitions"] = bad
            with pytest.raises(ConfigError, match="vector_store.partitions"):
                load(tmp_path, data)

    def test_partitions_under_retrieval_names_its_new_home(self, tmp_path):
        data = base()
        data["retrieval"]["partitions"] = 4
        with pytest.raises(ConfigError, match="declare vector_store.partitions: 4"):
            load(tmp_path, data)
        # the old sequential shape spelled its defaults; one partition
        # under the old schema was the monolith, not the P=1 layout
        data["retrieval"]["partitions"] = 1
        with pytest.raises(ConfigError, match="delete the key, a monolithic"):
            load(tmp_path, data)

    def test_workers_are_free_of_the_partition_count(self, tmp_path):
        data = base()
        data["vector_store"]["partitions"] = 8
        data["retrieval"] = {"k": 10, "workers": 3}
        config = load(tmp_path, data)
        assert config.retrieval.workers == 3
        assert resolved_execution(config) == ("threads", 3)

    def test_absent_execution_keys_resolve_to_threads_and_one(self, tmp_path):
        data = base()
        data["vector_store"]["partitions"] = 2
        config = load(tmp_path, data)
        assert config.retrieval.executor is None and config.retrieval.workers is None
        assert resolved_execution(config) == ("threads", 1)

    def test_serial_executor_searches_the_same_index_one_partition_at_a_time(self, tmp_path):
        data = base()
        data["vector_store"]["partitions"] = 8
        data["retrieval"] = {"k": 10, "executor": "serial"}
        assert resolved_execution(load(tmp_path, data)) == ("serial", 1)
        data["retrieval"]["workers"] = 1
        assert resolved_execution(load(tmp_path, data)) == ("serial", 1)
        data["retrieval"]["workers"] = 4
        with pytest.raises(ConfigError, match="retrieval.workers"):
            load(tmp_path, data)  # a knob the serial executor would ignore

    def test_unknown_executor(self, tmp_path):
        data = base()
        data["vector_store"]["partitions"] = 2
        data["retrieval"]["executor"] = "lithops"
        with pytest.raises(ConfigError, match="retrieval.executor"):
            load(tmp_path, data)

    def test_workers_must_be_at_least_one(self, tmp_path):
        data = base()
        data["vector_store"]["partitions"] = 2
        data["retrieval"]["workers"] = 0
        with pytest.raises(ConfigError, match="retrieval.workers"):
            load(tmp_path, data)

    @pytest.mark.parametrize("key,value", [("workers", 2), ("executor", "threads")])
    def test_execution_keys_need_a_partitioned_index(self, tmp_path, key, value):
        data = base()
        data["retrieval"][key] = value
        with pytest.raises(ConfigError, match=f"retrieval.{key}.*partitioned index"):
            load(tmp_path, data)

    def test_legacy_mode_key_is_accepted_when_it_agrees_and_refused_otherwise(self, tmp_path):
        # configurations written for the previous schema (the ones the
        # written report reproduces) keep loading while they tell the truth
        data = base()
        data["retrieval"]["mode"] = "sequential"
        assert layout_mode(load(tmp_path, data)) == "sequential"
        data["retrieval"]["mode"] = "collective"
        with pytest.raises(ConfigError, match="retrieval.mode: 'collective' contradicts"):
            load(tmp_path, data)
        data["vector_store"]["partitions"] = 2
        assert layout_mode(load(tmp_path, data)) == "collective"
        data["retrieval"]["mode"] = "sequential"
        with pytest.raises(ConfigError, match="retrieval.mode"):
            load(tmp_path, data)
        data["retrieval"]["mode"] = "parallel"
        with pytest.raises(ConfigError, match="retrieval.mode: unknown mode"):
            load(tmp_path, data)

    def test_shipped_collective_configs_declare_the_layout(self):
        for partitions in (1, 2, 4, 8):
            config = load_config(f"configs/collective_{partitions}.yaml")
            assert layout_mode(config) == "collective"
            assert config.vector_store.partitions == partitions
            assert config.vector_store.path.endswith(f"-p{partitions}")
            # as many threads as partitions is the partition-scaling
            # experiment's choice for these files, not a rule
            assert resolved_execution(config) == ("threads", partitions)

    def test_shipped_serial_config_shares_the_eight_partition_index(self):
        threads = load_config("configs/collective_8.yaml")
        serial = load_config("configs/collective_8_serial.yaml")
        assert serial.vector_store == threads.vector_store
        assert resolved_execution(serial) == ("serial", 1)

    def test_every_shipped_config_parses(self):
        for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
            load_config(path)

    def test_k_must_be_positive(self, tmp_path):
        data = base()
        data["retrieval"]["k"] = 0
        with pytest.raises(ConfigError, match="retrieval.k"):
            load(tmp_path, data)
