"""Integration test: full ingest over a synthetic corpus.

Builds a schema-faithful miniature of the OWI v2.0.0 layout (records +
text + embeddings parquet), writes a real YAML config, and runs the
pipeline end to end into a temporary Chroma collection, including the
re-run that proves ingestion is idempotent.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from rag_framework.__main__ import main
from rag_framework.orchestration.pipeline import RAGPipeline

PARTITION = "year=2026/month=7/day=28/language=spa"

_OFFSETS_TYPE = pa.list_(pa.struct([("start", pa.int32()), ("end", pa.int32())]))

DOCS = [
    # (id, main_content, offsets)
    ("aaa", "primer documento con dos ventanas", [(0, 16), (16, 33)]),
    ("bbb", "segundo documento entero", [(0, 24)]),
]
VECTORS = {  # (id, position) -> 4-d vector
    ("aaa", 0): [0.5, 0.5, 0.0, 0.0],
    ("aaa", 1): [0.0, 0.5, 0.5, 0.0],
    ("bbb", 0): [0.0, 0.0, 0.5, 0.5],
}


def write_corpus(root: Path, docs=DOCS) -> None:
    records_dir = root / "owie-ds" / PARTITION
    text_dir = root / "owi-ds" / PARTITION
    records_dir.mkdir(parents=True)
    text_dir.mkdir(parents=True)

    pq.write_table(
        pa.table(
            {
                "id": pa.array([d[0] for d in docs], pa.string()),
                "chunk_offsets": pa.array(
                    [
                        [{"start": s, "end": e} for s, e in d[2]]
                        for d in docs
                    ],
                    _OFFSETS_TYPE,
                ),
            }
        ),
        records_dir / "metadata_0_records.parquet",
    )
    pq.write_table(
        pa.table(
            {
                "id": pa.array([d[0] for d in docs], pa.string()),
                "url": pa.array([f"https://e.org/{d[0]}" for d in docs], pa.string()),
                "title": pa.array([f"t-{d[0]}" for d in docs], pa.string()),
                "main_content": pa.array([d[1] for d in docs], pa.string()),
                "language": pa.array(["spa"] * len(docs), pa.string()),
            }
        ),
        text_dir / "metadata_0.parquet",
    )
    keys = sorted(VECTORS)
    pq.write_table(
        pa.table(
            {
                "record_id": pa.array([k[0] for k in keys], pa.string()),
                "chunk_idx": pa.array([k[1] for k in keys], pa.int32()),
                "embedding": pa.array(
                    [
                        [np.float16(x) for x in VECTORS[k]]
                        for k in keys
                    ],
                    pa.list_(pa.float16()),
                ),
            }
        ),
        records_dir / "metadata_0_embeddings.parquet",
    )


def write_config(tmp_path: Path, retrieval=None) -> Path:
    config = {
        "dataset": {"loader": "owi", "path": str(tmp_path / "corpus")},
        "chunking": {"strategy": "publisher_offsets"},
        "embedding": {
            "provider": "precomputed",
            "model": "test-model",
            "normalize": True,
        },
        "vector_store": {
            "type": "chroma",
            "path": str(tmp_path / "state"),
            "collection": "test-col",
        },
        "retrieval": retrieval or {"mode": "sequential", "k": 10},
        "metrics": {"output": str(tmp_path / "results")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_full_ingest_and_idempotent_rerun(tmp_path):
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)

    pipeline = RAGPipeline.from_config(config_path)
    report = pipeline.ingest()

    assert report.documents_read == 2
    assert report.documents_rejected == 0
    assert report.duplicates_skipped == 0
    assert report.chunks_created == 3
    assert report.final_vector_count == 3
    assert report.bytes_processed > 0
    assert report.index_size_bytes > 0
    assert report.total_time_seconds >= 0

    # re-ingesting unchanged input must not create duplicates — and it
    # must hold on a fresh pipeline instance, not just in-process...
    second_pipeline = RAGPipeline.from_config(config_path)
    second = second_pipeline.ingest()
    assert second.final_vector_count == 3
    assert second.chunks_created == 3
    assert second_pipeline.vectors_before == 3  # auditable from the payload

    # ...and on the SAME instance (counter-reset contract end to end)
    third = second_pipeline.ingest()
    assert third.final_vector_count == 3
    assert third.documents_read == 2


def test_rejections_reach_the_written_report(tmp_path, capsys):
    # one record whose text half is empty: rejected with a reason, and
    # the reason must survive into the report payload, or a rejection
    # becomes an unexplained count
    docs = DOCS + [("ccc", "", [(0, 4)])]
    write_corpus(tmp_path / "corpus", docs=docs)
    config_path = write_config(tmp_path)

    assert main(["ingest", "--config", str(config_path)]) == 0
    payload = json.loads(
        next((tmp_path / "results").glob("ingest-*.json")).read_text()
    )
    assert payload["report"]["documents_read"] == 3
    assert payload["report"]["documents_rejected"] == 1
    assert payload["report"]["final_vector_count"] == 3
    (sample,) = payload["details"]["loader_rejection_sample"]
    assert "empty main_content" in sample["reason"]


def test_cli_writes_report_with_manifest(tmp_path, capsys):
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)

    exit_code = main(["ingest", "--config", str(config_path)])
    assert exit_code == 0

    reports = list((tmp_path / "results").glob("ingest-*.json"))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    assert payload["report"]["final_vector_count"] == 3
    manifest = payload["manifest"]
    assert set(manifest) >= {"timestamp", "git_commit", "machine", "versions", "config"}
    assert manifest["config"]["vector_store"]["collection"] == "test-col"
    assert payload["details"]["windows_clipped"] == 0

    printed = capsys.readouterr().out
    assert '"final_vector_count": 3' in printed


def test_cli_config_error_is_a_clean_failure(tmp_path, capsys):
    exit_code = main(["ingest", "--config", str(tmp_path / "missing.yaml")])
    assert exit_code == 1


class StubQueryEncoder:
    """Stands in for LocalEmbeddingProvider: same constructor shape,
    fixed vector aimed at chunk ("aaa", 0) of the fixture corpus."""

    def __init__(self, model_id, batch_size=None):
        self.model_id = model_id
        self.normalized = True
        self.device = "stub"
        self.batch_size = batch_size

    def embed_query(self, text):
        return VECTORS[("aaa", 0)]


def test_query_cli_end_to_end(tmp_path, capsys, monkeypatch):
    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    exit_code = main(
        ["query", "--config", str(config_path), "--question", "una pregunta",
         "--retrieval-only", "--k", "2"]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "[1] score=1.0000" in printed
    assert "https://e.org/aaa" in printed

    (report,) = (tmp_path / "results").glob("query-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["query"] == "una pregunta"
    assert payload["retrieval_mode"] == "sequential"
    assert payload["sources"][0]["document_id"] == "aaa"
    assert payload["sources"][0]["rank"] == 1
    assert payload["metrics"]["results_returned"] == 2
    assert payload["metrics"]["embed_seconds"] >= 0
    assert payload["metrics"]["search_seconds"] >= 0
    # retrieval and generation stay separately timed
    assert payload["metrics"]["total_seconds"] >= payload["metrics"]["retrieval_seconds"]
    assert payload["answer"] is None
    # query vectors are attributed to the encoder that produced them
    assert payload["query_encoder"]["device"] == "stub"
    assert payload["embedding_identity"]["device"] == "none"
    assert payload["manifest"]["dataset_version"]


def test_query_with_mock_generation(tmp_path, monkeypatch, capsys):
    import yaml as yaml_module

    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    data = yaml_module.safe_load(config_path.read_text())
    data["generation"] = {"enabled": True, "provider": "mock"}
    config_path.write_text(yaml_module.safe_dump(data), encoding="utf-8")
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    exit_code = main(
        ["query", "--config", str(config_path), "--question", "una pregunta"]
    )
    assert exit_code == 0
    printed = capsys.readouterr().out
    assert "[mock answer]" in printed

    (report,) = (tmp_path / "results").glob("query-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["answer"].startswith("[mock answer]")
    assert payload["answer"].count("https://e.org/") == len(payload["sources"])
    # generation latency reported separately from retrieval (section 12)
    assert payload["metrics"]["generation_seconds"] >= 0
    assert payload["metrics"]["total_seconds"] >= (
        payload["metrics"]["retrieval_seconds"]
        + payload["metrics"]["generation_seconds"]
    )

    # --retrieval-only overrides a generation-enabled config
    exit_code = main(
        ["query", "--config", str(config_path), "--question", "q",
         "--retrieval-only"]
    )
    assert exit_code == 0
    reports = sorted((tmp_path / "results").glob("query-*.json"))
    retrieval_only = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert retrieval_only["answer"] is None
    assert retrieval_only["metrics"]["generation_seconds"] is None

    # an unknown generation provider still fails at construction
    data["generation"] = {"enabled": True, "provider": "llm"}
    config_path.write_text(yaml_module.safe_dump(data), encoding="utf-8")
    assert main(
        ["query", "--config", str(config_path), "--question", "q"]
    ) == 1


def test_query_without_generation_section_stays_retrieval_only(
    tmp_path, monkeypatch, capsys
):
    # the config-default path (no generation section, no flag) must
    # behave exactly like retrieval-only: answer null, timing null
    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()
    assert main(
        ["query", "--config", str(config_path), "--question", "q"]
    ) == 0
    (report,) = (tmp_path / "results").glob("query-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["answer"] is None
    assert payload["metrics"]["generation_seconds"] is None


def test_generation_failure_is_a_clean_cli_error(tmp_path, monkeypatch, capsys):
    import yaml as yaml_module

    from rag_framework.generation.base import GenerationError
    from rag_framework.orchestration import pipeline as pipeline_module

    class BrokenGenerator:
        def generate(self, query, context):
            raise GenerationError("model timed out")

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    monkeypatch.setattr(
        pipeline_module, "build_generator", lambda config: BrokenGenerator()
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    data = yaml_module.safe_load(config_path.read_text())
    data["generation"] = {"enabled": True, "provider": "mock"}
    config_path.write_text(yaml_module.safe_dump(data), encoding="utf-8")
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    exit_code = main(
        ["query", "--config", str(config_path), "--question", "q"]
    )
    assert exit_code == 1  # message, never a traceback (section 22)


def test_query_against_missing_collection_fails_loudly(
    tmp_path, monkeypatch, capsys
):
    import yaml as yaml_module

    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    capsys.readouterr()

    # a typo'd collection name must never silently run an empty
    # experiment with a success exit code
    data = yaml_module.safe_load(config_path.read_text())
    data["vector_store"]["collection"] = "test-col-typo"
    config_path.write_text(yaml_module.safe_dump(data), encoding="utf-8")
    exit_code = main(
        ["query", "--config", str(config_path), "--question", "q"]
    )
    assert exit_code == 1
    assert not list((tmp_path / "results").glob("query-*.json"))


def test_query_k_zero_rejected_like_config_would(tmp_path, monkeypatch, capsys):
    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    exit_code = main(
        ["query", "--config", str(config_path), "--question", "q", "--k", "0"]
    )
    assert exit_code == 1  # a typo must not silently run an empty experiment
    assert not list((tmp_path / "results").glob("query-*.json"))


def test_report_write_failure_keeps_the_metrics(tmp_path, capsys):
    # an unwritable metrics dir must not cost the run its numbers: the
    # report is printed first, and the failure is a clean exit 3
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    blocker = tmp_path / "results"
    blocker.write_text("a file where the output dir should be")

    exit_code = main(["ingest", "--config", str(config_path)])
    assert exit_code == 3
    printed = capsys.readouterr().out
    assert '"final_vector_count": 3' in printed  # metrics survived


def test_collective_mode_end_to_end(tmp_path, capsys, monkeypatch):
    """Both retrieval modes are reachable by changing configuration alone."""
    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(
        tmp_path,
        retrieval={"mode": "collective", "k": 10, "partitions": 2, "workers": 2},
    )
    assert main(["ingest", "--config", str(config_path)]) == 0
    ingest_payload = json.loads(
        next((tmp_path / "results").glob("ingest-*.json")).read_text()
    )
    assert ingest_payload["report"]["final_vector_count"] == 3
    capsys.readouterr()

    pipeline = RAGPipeline.from_config(config_path)
    assert pipeline.vector_store.partitions == 2
    result = pipeline.query("una pregunta", k=2, retrieval_only=True)
    assert sum(pipeline.vector_store.partition_counts()) == 3
    assert abs(result.sources[0].score - 1.0) < 1e-6
    assert all(s.partition_id in ("0", "1") for s in result.sources)
    assert all(s.worker_id.startswith("w_") for s in result.sources)
    assert result.metrics["merge_seconds"] is not None
    assert result.metrics["candidates_returned"] == 3
    assert len(result.metrics["partition_searches"]) == 2

    exit_code = main(
        ["query", "--config", str(config_path), "--question", "una pregunta",
         "--retrieval-only", "--k", "2"]
    )
    assert exit_code == 0
    (report,) = (tmp_path / "results").glob("query-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["retrieval_mode"] == "collective"
    assert payload["partitions"] == 2 and payload["executor"] == "threads"
    assert payload["sources"][0]["partition_id"] in ("0", "1")


def test_correctness_benchmark_end_to_end(tmp_path, monkeypatch):
    """Experiment E runs through the benchmark on baseline + P=2 layouts."""
    import importlib.util

    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    baseline = write_config(tmp_path)
    assert main(["ingest", "--config", str(baseline)]) == 0
    collective_dir = tmp_path / "collective"
    collective_dir.mkdir()
    config = yaml.safe_load(baseline.read_text())
    config["vector_store"]["path"] = str(collective_dir / "state")
    config["retrieval"] = {"mode": "collective", "k": 10, "partitions": 2, "workers": 1, "executor": "serial"}
    collective = collective_dir / "config.yaml"
    collective.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(["ingest", "--config", str(collective)]) == 0
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query_id": "q1", "query": "x", "answerable": True, "relevant_document_ids": ["aaa"]}) + "\n"
    )

    spec = importlib.util.spec_from_file_location(
        "run_correctness", Path(__file__).resolve().parents[2] / "benchmarks" / "run_correctness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        "--baseline", str(baseline), "--collective", str(collective),
        "--queries", str(queries), "--output", str(tmp_path / "results"), "--k", "3",
    ]) == 0

    (report,) = (tmp_path / "results").glob("correctness-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["vectors"] == 3
    assert payload["baseline_vs_exact"]["summary"]["recall"] == 1.0
    (layout,) = payload["layouts"]
    assert layout["partitions"] == 2
    assert layout["vs_exact"]["identical_order"] == 1
    assert all(v == 0 for v in layout["differences"].values())
    assert "git_commit" in payload["manifest"]


def test_scaling_benchmark_end_to_end(tmp_path, monkeypatch):
    """Experiments B and C run through the benchmark on tiny layouts."""
    import importlib.util

    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    baseline = write_config(tmp_path)
    assert main(["ingest", "--config", str(baseline)]) == 0
    layouts = []
    for partitions in (1, 2):
        layout_dir = tmp_path / f"p{partitions}"
        layout_dir.mkdir()
        config = yaml.safe_load(baseline.read_text())
        config["vector_store"]["path"] = str(layout_dir / "state")
        config["retrieval"] = {"mode": "collective", "k": 10, "partitions": partitions, "workers": partitions}
        path = layout_dir / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        assert main(["ingest", "--config", str(path)]) == 0
        layouts.append(str(path))
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "query": "x", "answerable": True, "relevant_document_ids": ["aaa"]}) + "\n")

    spec = importlib.util.spec_from_file_location(
        "run_scaling", Path(__file__).resolve().parents[2] / "benchmarks" / "run_scaling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        "--baseline", str(baseline), "--layouts", *layouts, "--worker-layout", layouts[1],
        "--workers", "1", "2", "--queries", str(queries), "--output", str(tmp_path / "results"),
        "--k", "3", "--repetitions", "2", "--warm-up", "1",
    ]) == 0

    (partitions,) = (tmp_path / "results").glob("scaling-partitions-*.json")
    payload = json.loads(partitions.read_text(encoding="utf-8"))
    assert [r["label"] for r in payload["rows"]] == ["sequential", "P=1", "P=2"]
    assert payload["rows"][0]["timings"]["search_seconds"]["n"] == 2
    p2 = payload["rows"][2]
    assert p2["partition_counts"] == [2, 1] or p2["partition_counts"] == [1, 2]
    assert p2["partition_size_imbalance"]["max_over_mean"] == pytest.approx(4 / 3)
    assert set(p2["timings"]) >= {"merge_seconds", "worker_max_seconds", "partition_mean_seconds", "worker_time_imbalance"}
    assert p2["index_size_bytes"] > 0 and payload["repetitions"] == 2

    (workers,) = (tmp_path / "results").glob("scaling-workers-*.json")
    payload = json.loads(workers.read_text(encoding="utf-8"))
    assert [r["label"] for r in payload["rows"]] == ["serial x1", "threads x1", "threads x2"]
    assert payload["rows"][0]["executor"] == "serial"


def test_dataset_scaling_benchmark_end_to_end(tmp_path, monkeypatch):
    """Experiment A builds fresh subsets and measures them."""
    import importlib.util

    from rag_framework.models import stable_bucket
    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "query": "x", "answerable": True, "relevant_document_ids": ["aaa"]}) + "\n")
    # a percent that keeps "aaa" but not necessarily "bbb"
    keep_aaa = stable_bucket("aaa", 100) + 1

    spec = importlib.util.spec_from_file_location(
        "run_dataset_scaling", Path(__file__).resolve().parents[2] / "benchmarks" / "run_dataset_scaling.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "--config", str(config_path), "--percents", str(keep_aaa), "100",
        "--state", str(tmp_path / "scaling"), "--queries", str(queries),
        "--output", str(tmp_path / "results"), "--k", "3", "--repetitions", "2", "--warm-up", "1",
    ]
    assert module.main(argv) == 0
    assert module.main(argv) == 0  # a re-run builds into a new run folder

    (report,) = sorted((tmp_path / "results").glob("scaling-dataset-*.json"))[-1:]
    payload = json.loads(report.read_text(encoding="utf-8"))
    small, full = payload["rows"]
    assert full["percent"] == 100 and full["documents"] == 2 and full["vectors"] == 3
    assert full["documents_sampled_out"] == 0
    assert small["documents"] + small["documents_sampled_out"] == 2
    assert small["relevant_coverage"]["fraction"] == 1.0
    assert small["quality"]["mean_recall_at_k"] == 1.0
    assert full["throughput"]["queries_per_second"] > 0
    assert full["latency"]["search_seconds"]["n"] == 2
    assert full["index_size_bytes"] > 0
    assert payload["index_policy"].startswith("every subset is built fresh")
    runs = list((tmp_path / "scaling").iterdir())
    assert len(runs) == 2 and all((run / "p100").exists() for run in runs)


def test_quality_benchmark_end_to_end(tmp_path, monkeypatch):
    """run_quality reports index and exact quality side by side."""
    import importlib.util

    from rag_framework.orchestration import pipeline as pipeline_module

    monkeypatch.setattr(
        pipeline_module, "LocalEmbeddingProvider", StubQueryEncoder
    )
    write_corpus(tmp_path / "corpus")
    config_path = write_config(tmp_path)
    assert main(["ingest", "--config", str(config_path)]) == 0
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "query": "x", "answerable": True, "relevant_document_ids": ["aaa"]}) + "\n")

    spec = importlib.util.spec_from_file_location(
        "run_quality", Path(__file__).resolve().parents[2] / "benchmarks" / "run_quality.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([
        "--config", str(config_path), "--label", "tiny", "--queries", str(queries),
        "--output", str(tmp_path / "results"), "--k", "3", "--repetitions", "2", "--warm-up", "0",
    ]) == 0
    (report,) = (tmp_path / "results").glob("quality-tiny-*.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["vectors"] == 3
    assert payload["quality"]["mean_recall_at_k"] == 1.0
    assert payload["quality"]["exact_mean_recall_at_k"] == 1.0
    assert payload["exactness"]["mean_overlap_at_k"] == 1.0
    assert payload["relevant_coverage"]["in_corpus"] == 1
    assert payload["latency"]["search_seconds"]["n"] == 2
