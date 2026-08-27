"""Unit tests for deterministic corpus sampling."""

import pytest

from rag_framework.loaders.sample import SampledLoader
from rag_framework.models import Document, Rejection, stable_bucket


class StubLoader:
    def __init__(self, ids):
        self.ids = list(ids)
        self.rejections = [Rejection(source="r", reason="bad")]
        self.duplicates_skipped = 2
        self.loads = 0

    def load(self, source):
        self.loads += 1
        for doc_id in self.ids:
            yield Document(document_id=doc_id, text="t", metadata={})


def kept_ids(percent, ids):
    loader = SampledLoader(StubLoader(ids), percent)
    return {d.document_id for d in loader.load("src")}, loader


IDS = [f"doc-{i}" for i in range(2000)]


class TestSampling:
    def test_full_percent_is_identity(self):
        kept, loader = kept_ids(100, IDS)
        assert kept == set(IDS)
        assert loader.documents_sampled_out == 0

    def test_deterministic_and_roughly_proportional(self):
        first, loader = kept_ids(25, IDS)
        second, _ = kept_ids(25, IDS)
        assert first == second
        assert 0.22 * len(IDS) < len(first) < 0.28 * len(IDS)
        assert loader.documents_sampled_out == len(IDS) - len(first)

    def test_subsets_are_nested(self):
        ten, _ = kept_ids(10, IDS)
        twenty_five, _ = kept_ids(25, IDS)
        fifty, _ = kept_ids(50, IDS)
        assert ten < twenty_five < fifty < set(IDS)

    def test_independent_of_input_order(self):
        forward, _ = kept_ids(30, IDS)
        backward, _ = kept_ids(30, list(reversed(IDS)))
        assert forward == backward

    def test_uses_stable_bucket(self):
        kept, _ = kept_ids(40, IDS)
        assert kept == {i for i in IDS if stable_bucket(i, 100) < 40}

    def test_delegates_inner_accounting_and_resets_per_load(self):
        inner = StubLoader(IDS[:10])
        loader = SampledLoader(inner, 50)
        list(loader.load("src"))
        first = loader.documents_sampled_out
        list(loader.load("src"))
        assert loader.documents_sampled_out == first  # reset, not accumulated
        assert loader.rejections is inner.rejections
        assert loader.duplicates_skipped == 2
        assert inner.loads == 2

    @pytest.mark.parametrize("bad", [0, 101, -5])
    def test_out_of_range_percent(self, bad):
        with pytest.raises(ValueError, match="between 1 and 100"):
            SampledLoader(StubLoader([]), bad)

    @pytest.mark.parametrize("bad", [50.0, True, "50"])
    def test_non_int_percent(self, bad):
        with pytest.raises(TypeError):
            SampledLoader(StubLoader([]), bad)


class TestStableBucket:
    def test_range_and_pins(self):
        assert stable_bucket("abc", 4) == 2 and stable_bucket("abc", 16) == 10
        assert all(0 <= stable_bucket(i, 7) < 7 for i in IDS[:100])

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            stable_bucket("", 4)
        with pytest.raises(ValueError):
            stable_bucket("x", 0)
        with pytest.raises(TypeError):
            stable_bucket("x", True)
