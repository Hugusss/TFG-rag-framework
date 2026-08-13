"""Unit tests for the OWI loader, on tiny synthetic parquet pairs.

The fixtures reproduce the v2.0.0 layout faithfully (hive partitions,
paired records/text shards, an embeddings sibling the loader must
ignore) so no test ever touches the real dataset.
"""

import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag_framework.loaders.base import LoaderError
from rag_framework.loaders.owi import OwiLoader

PARTITION = "year=2026/month=7/day=28/language=spa"

_OFFSETS_TYPE = pa.list_(
    pa.struct([("start", pa.int32()), ("end", pa.int32())])
)


def write_pair(
    root: Path,
    docs: list[dict],
    *,
    partition: str = PARTITION,
    index: int = 0,
    with_text_file: bool = True,
    with_embeddings_file: bool = True,
    extra_text_docs: list[dict] | None = None,
    text_docs: list[dict] | None = None,
):
    """Write one records/text shard pair in the v2.0.0 layout.

    Each doc dict: {id, main_content, url, title, offsets}; None values
    become parquet nulls. ``text_docs`` overrides the text-file roster
    entirely (to model records without a text counterpart).
    """
    records_dir = root / "owie-dataset" / partition
    text_dir = root / "owi-dataset" / partition
    records_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    def offsets_value(d: dict):
        if "raw_offsets" in d:  # pass through verbatim (None, null structs)
            return d["raw_offsets"]
        return [{"start": s, "end": e} for s, e in d.get("offsets", [])]

    records = pa.table(
        {
            "id": pa.array([d["id"] for d in docs], pa.string()),
            "chunk_offsets": pa.array(
                [offsets_value(d) for d in docs], _OFFSETS_TYPE
            ),
        }
    )
    pq.write_table(records, records_dir / f"metadata_{index}_records.parquet")

    if with_embeddings_file:
        embeddings = pa.table({"record_id": pa.array([], pa.string())})
        pq.write_table(
            embeddings, records_dir / f"metadata_{index}_embeddings.parquet"
        )

    if with_text_file:
        roster = (docs if text_docs is None else text_docs) + (extra_text_docs or [])
        text = pa.table(
            {
                "id": pa.array([d["id"] for d in roster], pa.string()),
                "url": pa.array([d.get("url") for d in roster], pa.string()),
                "title": pa.array([d.get("title") for d in roster], pa.string()),
                "main_content": pa.array(
                    [d.get("main_content") for d in roster], pa.string()
                ),
                "language": pa.array(
                    [d.get("language", "spa") for d in roster], pa.string()
                ),
            }
        )
        pq.write_table(text, text_dir / f"metadata_{index}.parquet")


def doc(doc_id: str, **overrides) -> dict:
    result = {
        "id": doc_id,
        "main_content": f"content of {doc_id}",
        "url": f"https://example.org/{doc_id}",
        "title": f"title {doc_id}",
        "offsets": [(0, 5), (5, 10)],
    }
    result.update(overrides)
    return result


class TestHappyPath:
    def test_yields_documents_with_adopted_ids_and_joined_text(self, tmp_path):
        write_pair(tmp_path, [doc("aaa"), doc("bbb")])
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["aaa", "bbb"]
        assert documents[0].text == "content of aaa"
        assert loader.rejections == []
        assert loader.duplicates_skipped == 0

    def test_metadata_preserved(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")])
        (document,) = OwiLoader().load(tmp_path)
        assert document.metadata["url"] == "https://example.org/aaa"
        assert document.metadata["title"] == "title aaa"
        assert document.metadata["language"] == "spa"
        assert document.metadata["date"] == "2026-07-28"
        assert document.metadata["chunk_offsets"] == [(0, 5), (5, 10)]
        assert "metadata_0_records.parquet" in document.metadata["source_records_file"]
        assert "metadata_0.parquet" in document.metadata["source_text_file"]
        assert document.metadata["ingestion_version"]

    def test_missing_url_and_title_are_absent_keys(self, tmp_path):
        write_pair(tmp_path, [doc("aaa", url=None, title=None)])
        (document,) = OwiLoader().load(tmp_path)
        assert "url" not in document.metadata
        assert "title" not in document.metadata

    def test_empty_offsets_are_not_a_rejection(self, tmp_path):
        # a document without publisher chunks is still a document; the
        # publisher_offsets chunker decides what to do with it
        write_pair(tmp_path, [doc("aaa", offsets=[])])
        loader = OwiLoader()
        (document,) = loader.load(tmp_path)
        assert document.metadata["chunk_offsets"] == []
        assert loader.rejections == []

    def test_multiple_shards_sorted_and_merged(self, tmp_path):
        write_pair(tmp_path, [doc("bbb")], index=1)
        write_pair(tmp_path, [doc("aaa")], index=0)
        documents = list(OwiLoader().load(tmp_path))
        assert [d.document_id for d in documents] == ["aaa", "bbb"]

    def test_extra_text_rows_are_not_part_of_the_corpus(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")], extra_text_docs=[doc("zzz")])
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["aaa"]
        assert loader.rejections == []  # extras are not rejections

    def test_deterministic_across_runs(self, tmp_path):
        write_pair(tmp_path, [doc("aaa"), doc("bbb"), doc("ccc")])
        first = [(d.document_id, d.text) for d in OwiLoader().load(tmp_path)]
        second = [(d.document_id, d.text) for d in OwiLoader().load(tmp_path)]
        assert first == second

    def test_multiple_hive_partitions_pair_independently(self, tmp_path):
        cat = "year=2026/month=7/day=28/language=cat"
        write_pair(tmp_path, [doc("spa-doc")], partition=PARTITION)
        write_pair(tmp_path, [doc("cat-doc", language="cat")], partition=cat)
        documents = list(OwiLoader().load(tmp_path))
        # deterministic order: partitions sort as tuples, cat < spa
        assert [d.document_id for d in documents] == ["cat-doc", "spa-doc"]
        assert documents[0].metadata["language"] == "cat"
        assert documents[1].metadata["language"] == "spa"

    def test_whole_null_offsets_become_empty_list(self, tmp_path):
        write_pair(tmp_path, [doc("aaa", raw_offsets=None)])
        loader = OwiLoader()
        (document,) = loader.load(tmp_path)
        assert document.metadata["chunk_offsets"] == []
        assert loader.rejections == []


class TestRejections:
    def test_missing_text_row_rejected_with_reason(self, tmp_path, caplog):
        # "ghost" exists in records but has no row in the text file
        write_pair(tmp_path, [doc("aaa"), doc("ghost")], text_docs=[doc("aaa")])
        loader = OwiLoader()
        with caplog.at_level(logging.WARNING):
            documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["aaa"]
        (rejection,) = loader.rejections
        assert "ghost" in rejection.reason
        assert "no text row" in rejection.reason
        assert "rejected record" in caplog.text

    def test_empty_main_content_rejected(self, tmp_path):
        write_pair(tmp_path, [doc("aaa", main_content="   ")])
        loader = OwiLoader()
        assert list(loader.load(tmp_path)) == []
        (rejection,) = loader.rejections
        assert "empty main_content" in rejection.reason

    def test_missing_id_rejected(self, tmp_path):
        write_pair(tmp_path, [doc(""), doc("bbb")])
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["bbb"]
        (rejection,) = loader.rejections
        assert rejection.reason == "missing id"
        assert "#0" in rejection.source  # row-level traceability

    def test_duplicate_id_first_wins(self, tmp_path):
        write_pair(tmp_path, [doc("aaa", title="first")], index=0)
        write_pair(tmp_path, [doc("aaa", title="second")], index=1)
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert len(documents) == 1
        assert documents[0].metadata["title"] == "first"
        assert loader.duplicates_skipped == 1
        assert loader.rejections == []  # duplicates are counted, not rejected

    def test_counters_reset_between_load_calls(self, tmp_path):
        write_pair(tmp_path, [doc("aaa", main_content=None)])
        loader = OwiLoader()
        list(loader.load(tmp_path))
        list(loader.load(tmp_path))
        assert len(loader.rejections) == 1

    def test_whitespace_only_id_rejected(self, tmp_path):
        write_pair(tmp_path, [doc("   "), doc("bbb")])
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["bbb"]
        (rejection,) = loader.rejections
        assert rejection.reason == "missing id"

    def test_malformed_offsets_rejected(self, tmp_path):
        write_pair(
            tmp_path,
            [doc("aaa", raw_offsets=[{"start": None, "end": 5}]), doc("bbb")],
        )
        loader = OwiLoader()
        documents = list(loader.load(tmp_path))
        assert [d.document_id for d in documents] == ["bbb"]
        (rejection,) = loader.rejections
        assert "malformed chunk_offsets" in rejection.reason

    def test_duplicate_id_within_text_file_first_wins(self, tmp_path, caplog):
        write_pair(
            tmp_path,
            [doc("aaa")],
            text_docs=[doc("aaa", title="first"), doc("aaa", title="second")],
        )
        loader = OwiLoader()
        with caplog.at_level(logging.WARNING):
            (document,) = loader.load(tmp_path)
        assert document.metadata["title"] == "first"
        assert "duplicate id in text file" in caplog.text


class TestStructuralErrors:
    def test_missing_source_directory(self, tmp_path):
        with pytest.raises(LoaderError, match="not a directory"):
            OwiLoader().load(tmp_path / "nope")

    def test_no_records_files(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(LoaderError, match="no records parquet"):
            OwiLoader().load(tmp_path / "empty")

    def test_records_without_text_sibling_raises(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")], with_text_file=False)
        with pytest.raises(LoaderError, match="without a text sibling"):
            OwiLoader().load(tmp_path)

    def test_structural_error_raises_at_call_not_first_next(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")], with_text_file=False)
        loader = OwiLoader()
        with pytest.raises(LoaderError):
            loader.load(tmp_path)  # no iteration needed

    def test_embeddings_file_is_optional_and_ignored(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")], with_embeddings_file=False)
        documents = list(OwiLoader().load(tmp_path))
        assert [d.document_id for d in documents] == ["aaa"]

    def test_wrong_schema_raises_loader_error_at_call(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")])
        # overwrite the text file with one lacking required columns
        text_file = (
            tmp_path / "owi-dataset" / PARTITION / "metadata_0.parquet"
        )
        pq.write_table(pa.table({"id": pa.array(["aaa"], pa.string())}), text_file)
        with pytest.raises(LoaderError, match="missing required column"):
            OwiLoader().load(tmp_path)

    def test_conflicting_text_files_for_one_shard_raise(self, tmp_path):
        write_pair(tmp_path, [doc("aaa")])
        # an impostor text file with the same (partition, index) key in
        # the records dataset directory must not silently shadow the
        # real one
        impostor_dir = tmp_path / "owie-dataset" / PARTITION
        pq.write_table(
            pa.table({"id": pa.array(["zzz"], pa.string())}),
            impostor_dir / "metadata_0.parquet",
        )
        with pytest.raises(LoaderError, match="conflicting files"):
            OwiLoader().load(tmp_path)
