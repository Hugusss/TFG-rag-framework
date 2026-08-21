"""Tests for the plot script: result discovery and a full render."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[2]


def load_module():
    spec = importlib.util.spec_from_file_location("plot_results", REPO / "benchmarks" / "plot_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiscovery:
    def test_latest_picks_the_newest_timestamp(self, tmp_path):
        module = load_module()
        for stamp in ("20260101T000000Z", "20260301T000000Z", "20260201T000000Z"):
            (tmp_path / f"correctness-{stamp}.json").write_text("{}")
        (tmp_path / "correctness-extra-20260901T000000Z.json").write_text("{}")
        assert module.latest(tmp_path, "correctness").name == "correctness-extra-20260901T000000Z.json"
        (tmp_path / "correctness-extra-20260901T000000Z.json").unlink()
        assert module.latest(tmp_path, "correctness").name == "correctness-20260301T000000Z.json"

    def test_missing_source_is_an_error_not_a_placeholder(self, tmp_path):
        module = load_module()
        with pytest.raises(SystemExit, match="no scaling-workers"):
            module.latest(tmp_path, "scaling-workers")
        with pytest.raises(SystemExit):
            module.plot_latency_vs_workers(tmp_path, tmp_path)
        assert list(tmp_path.glob("*.png")) == []

    def test_stamp_names_source_and_commit(self):
        module = load_module()
        text = module.stamp({"manifest": {"git_commit": "abcdef0123456789"}}, Path("x.json"))
        assert text == "x.json @ abcdef012345"
        dirty = module.stamp({"manifest": {"git_commit": "abcdef0123456789-dirty"}}, Path("x.json"))
        assert dirty == "x.json @ abcdef012345-dirty"  # the flag must survive truncation


class TestRender:
    def test_every_required_plot_renders_from_committed_results(self, tmp_path):
        module = load_module()
        assert module.main(["--results", str(REPO / "results"), "--output", str(tmp_path)]) == 0
        names = sorted(p.name for p in tmp_path.glob("*.png"))
        assert names == [
            "01-ingestion-time-vs-chunks.png",
            "02-latency-vs-dataset-size.png",
            "03-latency-vs-workers.png",
            "04-latency-vs-partitions.png",
            "05-worker-time-distribution.png",
            "06-merge-time-vs-candidates.png",
            "07-overlap-vs-partitions.png",
            "08-quality-vs-chunk-size.png",
        ]
        assert all((tmp_path / n).stat().st_size > 10_000 for n in names)

    def test_only_selects_plots(self, tmp_path):
        module = load_module()
        assert module.main(["--results", str(REPO / "results"), "--output", str(tmp_path), "--only", "6"]) == 0
        assert [p.name for p in tmp_path.glob("*.png")] == ["06-merge-time-vs-candidates.png"]
