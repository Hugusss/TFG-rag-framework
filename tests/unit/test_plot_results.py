"""Tests for the plot script: result discovery and a full render."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[2]

# The two kinds with more than one committed campaign: the published figures
# describe the main corpus, so they pin its run (the other is the 1.36M one).
PINS = [
    "--pin", "scaling-workers=scaling-workers-20260821T143459.302145Z.json",
    "--pin", "correctness=correctness-20260821T141643.871410Z.json",
]


def load_module():
    spec = importlib.util.spec_from_file_location("plot_results", REPO / "benchmarks" / "plot_results.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestDiscovery:
    def test_one_candidate_needs_no_pin(self, tmp_path):
        module = load_module()
        (tmp_path / "correctness-20260301T000000Z.json").write_text("{}")
        assert module.latest(tmp_path, "correctness").name == "correctness-20260301T000000Z.json"

    def test_several_candidates_are_an_error_not_a_guess(self, tmp_path):
        module = load_module()
        for stamp in ("20260101T000000Z", "20260301T000000Z"):
            (tmp_path / f"correctness-{stamp}.json").write_text("{}")
        with pytest.raises(SystemExit, match="refusing to guess"):
            module.latest(tmp_path, "correctness")

    def test_pin_selects_the_run_and_a_missing_pin_is_an_error(self, tmp_path):
        module = load_module()
        for stamp in ("20260101T000000Z", "20260301T000000Z"):
            (tmp_path / f"correctness-{stamp}.json").write_text("{}")
        module.PINNED["correctness"] = "correctness-20260101T000000Z.json"
        try:
            assert module.latest(tmp_path, "correctness").name == "correctness-20260101T000000Z.json"
            module.PINNED["correctness"] = "correctness-does-not-exist.json"
            with pytest.raises(SystemExit, match="pinned correctness file not found"):
                module.latest(tmp_path, "correctness")
        finally:
            module.PINNED.clear()

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
        assert module.main(["--results", str(REPO / "results"), "--output", str(tmp_path), *PINS]) == 0
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
