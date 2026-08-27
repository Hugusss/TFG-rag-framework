"""Configuration seam: a YAML file becomes a validated, typed configuration.

:func:`load_config` is the only entry point. It parses the file into the
frozen dataclasses defined here and raises :class:`ConfigError` naming
the exact offending key (``retrieval.k``, not "invalid config") for every
failure: missing file, malformed YAML, unknown sections or keys, wrong
types, or values that contradict one another. Scalars are never coerced,
so a typo cannot quietly change what an experiment runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Any invalid configuration; the message names the offending key."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys.

    Stock PyYAML resolves duplicates silently as last-wins, which would
    let a stray second ``k:`` line change an experiment without a trace.
    """

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:
                raise ConfigError(f"unhashable mapping key: {key!r}") from None
            if duplicate:
                raise ConfigError(f"duplicate key: {key}")
            seen.add(key)
        return super().construct_mapping(node, deep)


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """``version`` names the corpus a result was produced from, e.g.
    ``owi-v2.0.0-gpu-spa-2026-07-28`` — the path alone is machine-local
    and identifies nothing."""

    loader: str
    path: str
    version: str | None = None
    sample_percent: int = 100


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Token sizes are required by ``recursive`` and forbidden by
    ``publisher_offsets``, which reuses the publisher's own boundaries."""

    strategy: str
    target_tokens: int | None = None
    overlap_tokens: int | None = None
    minimum_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str
    model: str
    normalize: bool = True
    batch_size: int | None = None


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    type: str
    path: str
    collection: str
    ef_search: int | None = None


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    mode: str
    k: int = 10
    partitions: int = 1
    workers: int = 1
    executor: str = "threads"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    enabled: bool = False
    provider: str = "mock"
    model: str | None = None
    endpoint: str = "http://localhost:11434"
    timeout_seconds: int = 120


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    output: str


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    dataset: DatasetConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    metrics: MetricsConfig


_MISSING = object()


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name, _MISSING)
    if value is _MISSING:
        raise ConfigError(f"missing section: {name}")
    if not isinstance(value, dict):
        raise ConfigError(f"{name}: must be a mapping")
    return value


def _forbid_unknown(section: dict, allowed: set[str], path: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        # keys may be non-strings (YAML types bare numbers): sort by str
        keys = ", ".join(f"{path}.{key}" for key in sorted(unknown, key=str))
        raise ConfigError(f"unknown key(s): {keys}")


def _take(section: dict, key: str, expected: type, path: str, default=_MISSING):
    value = section.get(key, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ConfigError(f"missing key: {path}.{key}")
        return default
    # an optional key is expressed by omission, so an explicit `null`
    # falls through to the type check and is rejected like any wrong type
    # bool is a subclass of int in Python; "k: true" must not pass as int
    if expected is int and isinstance(value, bool):
        raise ConfigError(f"{path}.{key}: expected int, got bool")
    if not isinstance(value, expected):
        raise ConfigError(
            f"{path}.{key}: expected {expected.__name__},"
            f" got {type(value).__name__}"
        )
    if expected is str and not value.strip():
        raise ConfigError(f"{path}.{key}: must not be empty")
    return value


def _parse_dataset(section: dict) -> DatasetConfig:
    _forbid_unknown(
        section, {"loader", "path", "version", "sample_percent"}, "dataset"
    )
    sample_percent = _take(section, "sample_percent", int, "dataset", default=100)
    if not 1 <= sample_percent <= 100:
        raise ConfigError(
            f"dataset.sample_percent: must be between 1 and 100, got {sample_percent}"
        )
    return DatasetConfig(
        loader=_take(section, "loader", str, "dataset"),
        path=_take(section, "path", str, "dataset"),
        version=_take(section, "version", str, "dataset", default=None),
        sample_percent=sample_percent,
    )


def _parse_chunking(section: dict) -> ChunkingConfig:
    _forbid_unknown(
        section,
        {"strategy", "target_tokens", "overlap_tokens", "minimum_tokens"},
        "chunking",
    )
    strategy = _take(section, "strategy", str, "chunking")
    size_keys = ("target_tokens", "overlap_tokens", "minimum_tokens")

    if strategy == "publisher_offsets":
        present = [key for key in size_keys if key in section]
        if present:
            keys = ", ".join(f"chunking.{key}" for key in present)
            raise ConfigError(
                f"{keys}: not applicable to strategy publisher_offsets,"
                " which reuses the publisher's chunk boundaries"
            )
        return ChunkingConfig(strategy=strategy)

    if strategy == "recursive":
        target = _take(section, "target_tokens", int, "chunking")
        overlap = _take(section, "overlap_tokens", int, "chunking")
        # required, not defaulted: the minimum size silently decides which
        # tail fragments are dropped, so an experiment must state it
        minimum = _take(section, "minimum_tokens", int, "chunking")
        if target <= 0:
            raise ConfigError("chunking.target_tokens: must be positive")
        if overlap < 0:
            raise ConfigError("chunking.overlap_tokens: must not be negative")
        if minimum < 0:
            raise ConfigError("chunking.minimum_tokens: must not be negative")
        if overlap >= target:
            raise ConfigError(
                "chunking.overlap_tokens: must be smaller than target_tokens"
            )
        if minimum > target:
            raise ConfigError(
                "chunking.minimum_tokens: must not exceed target_tokens"
            )
        return ChunkingConfig(
            strategy=strategy,
            target_tokens=target,
            overlap_tokens=overlap,
            minimum_tokens=minimum,
        )

    raise ConfigError(
        f"chunking.strategy: unknown strategy '{strategy}'"
        " (known: publisher_offsets, recursive)"
    )


def _parse_embedding(section: dict) -> EmbeddingConfig:
    _forbid_unknown(
        section, {"provider", "model", "normalize", "batch_size"}, "embedding"
    )
    batch_size = _take(section, "batch_size", int, "embedding", default=None)
    if batch_size is not None and batch_size <= 0:
        raise ConfigError("embedding.batch_size: must be positive")
    return EmbeddingConfig(
        provider=_take(section, "provider", str, "embedding"),
        model=_take(section, "model", str, "embedding"),
        normalize=_take(section, "normalize", bool, "embedding", default=True),
        batch_size=batch_size,
    )


def _parse_vector_store(section: dict) -> VectorStoreConfig:
    _forbid_unknown(section, {"type", "path", "collection", "ef_search"}, "vector_store")
    ef_search = _take(section, "ef_search", int, "vector_store", default=None)
    if ef_search is not None and ef_search < 1:
        raise ConfigError("vector_store.ef_search: must be positive")
    return VectorStoreConfig(
        type=_take(section, "type", str, "vector_store"),
        path=_take(section, "path", str, "vector_store"),
        collection=_take(section, "collection", str, "vector_store"),
        ef_search=ef_search,
    )


def _parse_retrieval(section: dict) -> RetrievalConfig:
    _forbid_unknown(
        section, {"mode", "k", "partitions", "workers", "executor"}, "retrieval"
    )
    mode = _take(section, "mode", str, "retrieval")
    if mode not in ("sequential", "collective"):
        raise ConfigError(
            f"retrieval.mode: unknown mode '{mode}'"
            " (known: sequential, collective)"
        )
    k = _take(section, "k", int, "retrieval", default=10)
    partitions = _take(section, "partitions", int, "retrieval", default=1)
    workers = _take(section, "workers", int, "retrieval", default=1)
    if k <= 0:
        raise ConfigError("retrieval.k: must be positive")
    if partitions < 1:
        raise ConfigError("retrieval.partitions: must be at least 1")
    if workers < 1:
        raise ConfigError("retrieval.workers: must be at least 1")
    if mode == "sequential" and (partitions != 1 or workers != 1):
        raise ConfigError(
            "retrieval.partitions/workers: must be 1 when mode is sequential"
        )
    if mode == "sequential" and "executor" in section:
        raise ConfigError(
            "retrieval.executor: only meaningful when mode is collective"
        )
    executor = _take(section, "executor", str, "retrieval", default="threads")
    if executor not in ("serial", "threads"):
        raise ConfigError(
            f"retrieval.executor: unknown executor '{executor}'"
            " (known: serial, threads)"
        )
    if executor == "serial" and workers != 1:
        raise ConfigError(
            "retrieval.workers: must be 1 when executor is serial"
        )
    return RetrievalConfig(
        mode=mode, k=k, partitions=partitions, workers=workers, executor=executor
    )


def _parse_generation(section: dict) -> GenerationConfig:
    _forbid_unknown(
        section,
        {"enabled", "provider", "model", "endpoint", "timeout_seconds"},
        "generation",
    )
    provider = _take(section, "provider", str, "generation", default="mock")
    if provider != "ollama":
        for key in ("model", "endpoint", "timeout_seconds"):
            if key in section:
                raise ConfigError(
                    f"generation.{key}: only meaningful when provider is ollama"
                )
    model = _take(section, "model", str, "generation", default=None)
    endpoint = _take(
        section, "endpoint", str, "generation", default="http://localhost:11434"
    )
    timeout_seconds = _take(
        section, "timeout_seconds", int, "generation", default=120
    )
    if provider == "ollama" and not model:
        raise ConfigError(
            "generation.model: required when provider is ollama"
        )
    if timeout_seconds < 1:
        raise ConfigError("generation.timeout_seconds: must be positive")
    return GenerationConfig(
        enabled=_take(section, "enabled", bool, "generation", default=False),
        provider=provider,
        model=model,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
    )


def _parse_metrics(section: dict) -> MetricsConfig:
    _forbid_unknown(section, {"output"}, "metrics")
    return MetricsConfig(output=_take(section, "output", str, "metrics"))


_REQUIRED_SECTIONS = (
    "dataset",
    "chunking",
    "embedding",
    "vector_store",
    "retrieval",
    "metrics",
)
_OPTIONAL_SECTIONS = ("generation",)


def load_config(path: str | Path) -> PipelineConfig:
    """Parse and validate a pipeline configuration file.

    Raises :class:`ConfigError` for every failure mode; never returns a
    partially valid configuration.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {file}") from None
    except OSError as error:
        raise ConfigError(f"config file unreadable: {file}: {error}") from None

    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {file}: {error}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{file}: top level must be a mapping of sections")

    unknown = set(raw) - set(_REQUIRED_SECTIONS) - set(_OPTIONAL_SECTIONS)
    if unknown:
        names = ", ".join(str(name) for name in sorted(unknown, key=str))
        raise ConfigError(f"unknown section(s): {names}")

    if "generation" in raw:
        generation = _parse_generation(_section(raw, "generation"))
    else:
        generation = GenerationConfig()

    return PipelineConfig(
        dataset=_parse_dataset(_section(raw, "dataset")),
        chunking=_parse_chunking(_section(raw, "chunking")),
        embedding=_parse_embedding(_section(raw, "embedding")),
        vector_store=_parse_vector_store(_section(raw, "vector_store")),
        retrieval=_parse_retrieval(_section(raw, "retrieval")),
        generation=generation,
        metrics=_parse_metrics(_section(raw, "metrics")),
    )
