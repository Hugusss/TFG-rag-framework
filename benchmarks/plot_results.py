"""Regenerate the eight result figures from saved raw results.

Every figure is drawn from a result file written by the metrics layer —
never from values typed here — and is stamped with the source file name
and the git commit recorded in its manifest. A missing source is an
error, never a placeholder figure.

Reads: results/*.json. Writes: figures/01-*.png .. figures/08-*.png.
``--no-source-stamp`` omits that footer for figures destined to a
document that cites its sources in prose; the stamp is on by default
because a figure that travels alone must carry its own provenance.
Requires the optional ``plots`` extra; the framework itself never
imports matplotlib (ADR-012).

Each figure takes the newest result file of its kind, so a later
campaign of the same kind would silently redraw it at another scale.
The eight published figures all describe the main corpus, and this is
the command that regenerates them:

    python benchmarks/plot_results.py \
        --pin scaling-workers=scaling-workers-20260821T143459.302145Z.json \
        --pin correctness=correctness-20260821T141643.871410Z.json

Without those pins the two files from the full-scale campaign win and
figures 3 and 7 come out at 1.36M vectors instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise SystemExit(
        "matplotlib is not installed: pip install -e '.[plots]'"
    ) from error
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEQUENTIAL = {"color": "#1f77b4", "marker": "s", "linestyle": "--"}
COLLECTIVE = {"color": "#d62728", "marker": "o", "linestyle": "-"}
SERIAL = {"color": "#7f7f7f", "marker": "D"}


# --- result discovery --------------------------------------------------------


# set by main(); maps a result kind to the exact file chosen with --pin
PINNED: dict[str, str] = {}


def latest(results: Path, kind: str) -> Path:
    """The one result file for ``kind``, or the one ``--pin`` selects.

    Picking the newest of several candidates would be a convenience with a
    trap in it: a later campaign of the same kind becomes the source without
    saying so, and the figure quietly changes what it means — same title,
    same axes, another corpus. So ambiguity is an error here, and ``--pin``
    is how a figure states which run it draws.
    """
    if kind in PINNED:
        path = results / PINNED[kind]
        if not path.is_file():
            raise SystemExit(f"pinned {kind} file not found: {path}")
        return path
    files = sorted(results.glob(f"{kind}-*.json"))
    if not files:
        raise SystemExit(f"no {kind}-*.json under {results}; run the benchmark first")
    if len(files) > 1:
        listing = "\n  ".join(f.name for f in files)
        raise SystemExit(
            f"{len(files)} {kind}-*.json files under {results}; refusing to guess.\n"
            f"  {listing}\n"
            f"Campaigns of the same kind can differ in corpus and scale, so picking "
            f"one silently would change what the figure means. Choose with "
            f"--pin {kind}=<file>."
        )
    return files[0]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# set by main(); when False, finish() draws no provenance footer
SOURCE_STAMP = True


def stamp(payload: dict, path: Path) -> str:
    commit = (payload.get("manifest") or {}).get("git_commit", "unknown")
    short = commit[:12] + ("-dirty" if commit.endswith("-dirty") else "")
    return f"{path.name} @ {short}"


def stamp_many(pairs: list[tuple[dict, Path]]) -> str:
    """One line per source file so long lists never run off the figure."""
    return "\n".join(stamp(payload, path) for payload, path in pairs)


def range_bars(summary: dict, scale: float = 1000.0):
    """Asymmetric error bar from the median down to min and up to p95."""
    median = summary["median"] * scale
    return median, [[median - summary["min"] * scale], [summary["p95"] * scale - median]]


def finish(fig, axis, title: str, source: str, output: Path, name: str) -> Path:
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if SOURCE_STAMP:
        fig.text(0.99, 0.01, source, ha="right", va="bottom", fontsize=6, color="#555555")
        fig.tight_layout(rect=(0, 0.02 + 0.025 * source.count("\n"), 1, 1))
    else:
        fig.tight_layout()
    path = output / name
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --- the eight plots ----------------------------------------------------------


def plot_ingestion_vs_chunks(results: Path, output: Path) -> Path:
    """Plot 1 — ingestion time versus number of chunks."""
    fig, axis = plt.subplots(figsize=(7, 4.5))
    dataset_path = latest(results, "scaling-dataset")
    dataset = load(dataset_path)
    rows = dataset["rows"]
    axis.plot(
        [r["chunks"] for r in rows], [r["index_construction_seconds"] for r in rows],
        label="dataset subsets, precomputed vectors (sequential)", **SEQUENTIAL,
    )
    sources = [(dataset, dataset_path)]
    live, precomputed = [], []
    for name in ("publisher", "small", "medium", "large"):
        path = latest(results, f"chunkexp-{name}")
        payload = load(path)
        report = payload["ingest_report"]
        hits = payload["ingest_details"].get("embedding_cache_hits") or 0
        point = (report["chunks_created"], report["total_time_seconds"], name, hits)
        (precomputed if payload["manifest"]["config"]["embedding"]["provider"] == "precomputed" else live).append(point)
        sources.append((payload, path))
    for points, label, style in (
        (precomputed, "chunk experiment, precomputed vectors", {"color": "#2ca02c", "marker": "^"}),
        (live, "chunk experiment, live CPU encoding", {"color": "#ff7f0e", "marker": "v"}),
    ):
        if points:
            axis.scatter([p[0] for p in points], [p[1] for p in points], label=label, s=60, zorder=3, **style)
            for chunks, seconds, name, hits in points:
                note = f"{name}" + (f" ({hits} cached)" if hits else "")
                axis.annotate(note, (chunks, seconds), textcoords="offset points", xytext=(6, 4), fontsize=8)
    axis.set_yscale("log")
    axis.set_xlabel("chunks ingested")
    axis.set_ylabel("ingestion time (s, log scale; setup + embed + index)")
    axis.legend(fontsize=8)
    return finish(fig, axis, "Ingestion time vs number of chunks", stamp_many(sources), output, "01-ingestion-time-vs-chunks.png")


def plot_latency_vs_dataset(results: Path, output: Path) -> Path:
    """Plot 2 — retrieval latency versus dataset size."""
    path = latest(results, "scaling-dataset")
    payload = load(path)
    rows = payload["rows"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for key, label, style in (
        ("retrieval_seconds", "retrieval total (embed + search), sequential", SEQUENTIAL),
        ("search_seconds", "search only, sequential", {**SEQUENTIAL, "marker": "o", "linestyle": ":"}),
    ):
        medians, lows, highs = [], [], []
        for r in rows:
            median, (low, high) = range_bars(r["latency"][key])
            medians.append(median); lows.append(low[0]); highs.append(high[0])
        axis.errorbar([r["vectors"] for r in rows], medians, yerr=[lows, highs], capsize=3, label=label, **style)
    axis.set_yscale("log")
    axis.set_xlabel("vectors in the index")
    axis.set_ylabel("latency (ms, log scale; median, bars = min..p95)")
    axis.legend(fontsize=8)
    return finish(fig, axis, f"Retrieval latency vs dataset size (k={payload['k']}, {payload['queries']} queries × {payload['repetitions']})", stamp(payload, path), output, "02-latency-vs-dataset-size.png")


def plot_latency_vs_workers(results: Path, output: Path) -> Path:
    """Plot 3 — retrieval latency versus number of workers."""
    path = latest(results, "scaling-workers")
    payload = load(path)
    rows = payload["rows"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    labels = [r["label"] for r in rows]
    positions = list(range(len(rows)))
    for key, label, style in (
        ("search_seconds", "fan-out wall time (collective)", COLLECTIVE),
        ("worker_max_seconds", "slowest worker", {**COLLECTIVE, "marker": "^", "linestyle": ":"}),
        ("worker_mean_seconds", "mean worker", {**COLLECTIVE, "marker": "v", "linestyle": "-."}),
    ):
        medians, lows, highs = [], [], []
        for r in rows:
            median, (low, high) = range_bars(r["timings"][key])
            medians.append(median); lows.append(low[0]); highs.append(high[0])
        axis.errorbar(positions, medians, yerr=[lows, highs], capsize=3, label=label, **style)
    serial = [i for i, r in enumerate(rows) if r["executor"] == "serial"]
    for i in serial:
        axis.axvspan(i - 0.3, i + 0.3, color=SERIAL["color"], alpha=0.15, label="serial executor (no threads)")
    axis.set_xticks(positions, labels)
    axis.set_xlabel("executor × workers")
    axis.set_ylabel("latency (ms; median, bars = min..p95)")
    partitions = rows[0]["partitions"]
    axis.legend(fontsize=8)
    return finish(fig, axis, f"Retrieval latency vs number of workers (P={partitions}, k={payload['k']})", stamp(payload, path), output, "03-latency-vs-workers.png")


def plot_latency_vs_partitions(results: Path, output: Path) -> Path:
    """Plot 4 — retrieval latency versus number of partitions."""
    path = latest(results, "scaling-partitions")
    payload = load(path)
    rows = payload["rows"]
    sequential = [r for r in rows if r["mode"] == "sequential"]
    collective = [r for r in rows if r["mode"] == "collective"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for key, label, style in (
        ("search_seconds", "fan-out wall time (collective, workers = P)", COLLECTIVE),
        ("worker_max_seconds", "slowest worker (collective)", {**COLLECTIVE, "marker": "^", "linestyle": ":"}),
    ):
        medians, lows, highs = [], [], []
        for r in collective:
            median, (low, high) = range_bars(r["timings"][key])
            medians.append(median); lows.append(low[0]); highs.append(high[0])
        axis.errorbar([r["partitions"] for r in collective], medians, yerr=[lows, highs], capsize=3, label=label, **style)
    if sequential:
        summary = sequential[0]["timings"]["search_seconds"]
        axis.axhline(summary["median"] * 1000, label="sequential, single collection (median)", **{k: v for k, v in SEQUENTIAL.items() if k != "marker"})
        axis.axhspan(summary["min"] * 1000, summary["p95"] * 1000, color=SEQUENTIAL["color"], alpha=0.12, label="sequential min..p95")
    axis.set_xscale("log", base=2)
    axis.set_xticks([r["partitions"] for r in collective], [str(r["partitions"]) for r in collective])
    axis.set_xlabel("partitions P (workers = P)")
    axis.set_ylabel("search latency (ms; median, bars = min..p95)")
    axis.legend(fontsize=8)
    return finish(fig, axis, f"Retrieval latency vs number of partitions (k={payload['k']})", stamp(payload, path), output, "04-latency-vs-partitions.png")


def plot_worker_time_distribution(results: Path, output: Path) -> Path:
    """Plot 5 — worker-time distribution and partition imbalance."""
    path = latest(results, "scaling-partitions")
    payload = load(path)
    collective = [r for r in payload["rows"] if r["mode"] == "collective" and "partition_mean_seconds" in r["timings"]]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    width = 0.8 / max(len(collective), 1)
    for i, r in enumerate(collective):
        means = r["timings"]["partition_mean_seconds"]
        xs = [int(pid) + i * width for pid in means]
        left.bar(xs, [1000 * v for v in means.values()], width=width, label=f"P={r['partitions']} (time imbalance {r['timings']['worker_time_imbalance']['max_over_mean']:.2f})")
        counts = r["partition_counts"]
        right.bar([j + i * width for j in range(len(counts))], counts, width=width, label=f"P={r['partitions']} (size imbalance {r['partition_size_imbalance']['max_over_mean']:.2f})")
    left.set_xlabel("partition index"); left.set_ylabel("mean worker search time (ms, over all queries)")
    left.set_title("Worker-time distribution per partition (collective)"); left.legend(fontsize=7); left.grid(True, alpha=0.3)
    right.set_xlabel("partition index"); right.set_ylabel("vectors in partition")
    right.set_title("Partition sizes (document-hash assignment)"); right.legend(fontsize=7); right.grid(True, alpha=0.3)
    if SOURCE_STAMP:
        fig.text(0.99, 0.01, stamp(payload, path), ha="right", va="bottom", fontsize=7, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
    else:
        fig.tight_layout()
    out = output / "05-worker-time-distribution.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    return out


def plot_merge_vs_candidates(results: Path, output: Path) -> Path:
    """Plot 6 — merge time versus number of candidates."""
    path = latest(results, "scaling-partitions")
    payload = load(path)
    collective = [r for r in payload["rows"] if r["mode"] == "collective"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    medians, lows, highs, candidates = [], [], [], []
    for r in collective:
        median, (low, high) = range_bars(r["timings"]["merge_seconds"])
        medians.append(median); lows.append(low[0]); highs.append(high[0])
        candidates.append(r["timings"]["candidates_returned"]["mean"])
    axis.errorbar(candidates, medians, yerr=[lows, highs], capsize=3, label="coordinator merge_top_k (collective)", **COLLECTIVE)
    for r, x, y in zip(collective, candidates, medians):
        axis.annotate(f"P={r['partitions']}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8)
    axis.set_xlabel("candidates merged (P × k)")
    axis.set_ylabel("merge time (ms; median, bars = min..p95)")
    axis.legend(fontsize=8)
    return finish(fig, axis, f"Merge time vs number of candidates (k={payload['k']})", stamp(payload, path), output, "06-merge-time-vs-candidates.png")


def plot_overlap_vs_partitions(results: Path, output: Path) -> Path:
    """Plot 7 — top-k overlap versus number of partitions."""
    path = latest(results, "correctness")
    payload = load(path)
    layouts = payload["layouts"]
    ps = [L["partitions"] for L in layouts]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(ps, [L["vs_baseline"]["overlap_at_k"] for L in layouts], label="collective vs sequential baseline (single HNSW)", **COLLECTIVE)
    axis.plot(ps, [L["vs_exact"]["overlap_at_k"] for L in layouts], label="collective vs exact brute-force reference", **{**COLLECTIVE, "marker": "^", "linestyle": ":"})
    base = payload["baseline_vs_exact"]["summary"]["overlap_at_k"]
    axis.axhline(base, label="sequential baseline vs exact reference", **{k: v for k, v in SEQUENTIAL.items() if k != "marker"})
    for L in layouts:
        axis.annotate(f"order {L['vs_exact']['identical_order']}/{payload['queries']}", (L["partitions"], L["vs_exact"]["overlap_at_k"]), textcoords="offset points", xytext=(6, 6), fontsize=7)
    axis.set_ylim(min(0.9, base - 0.02), 1.01)
    axis.set_xscale("log", base=2); axis.set_xticks(ps, [str(p) for p in ps])
    axis.set_xlabel("partitions P")
    axis.set_ylabel(f"mean top-{payload['k']} overlap (fraction of {payload['k']})")
    axis.legend(fontsize=8, loc="lower left")
    return finish(fig, axis, f"Top-k overlap vs number of partitions ({payload['queries']} queries)", stamp(payload, path), output, "07-overlap-vs-partitions.png")


def plot_quality_vs_chunk_size(results: Path, output: Path) -> Path:
    """Plot 8 — retrieval quality versus chunk size."""
    names = ("publisher", "small", "medium", "large")
    payloads, sources = [], []
    for name in names:
        path = latest(results, f"chunkexp-{name}")
        payloads.append(load(path)); sources.append((payloads[-1], path))
    fig, axis = plt.subplots(figsize=(7.5, 4.5))
    labels = []
    for payload in payloads:
        chunking = payload["manifest"]["config"]["chunking"]
        label = payload["configuration"]
        if chunking.get("target_tokens"):
            label += f"\n{chunking['target_tokens']}/{chunking['overlap_tokens']} words"
        else:
            label += "\nofficial chunks"
        labels.append(label)
    positions = list(range(len(payloads)))
    width = 0.25
    for offset, key, label in ((-width, "mean_recall_at_k", "Recall@k"), (0, "mean_reciprocal_rank", "MRR"), (width, "mean_precision_at_k", "Precision@k")):
        values = [p["quality"][key] for p in payloads]
        bars = axis.bar([x + offset for x in positions], values, width=width, label=label)
        for bar, value in zip(bars, values):
            axis.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom", fontsize=7)
    for x, payload in zip(positions, payloads):
        dropped = payload["ingest_details"]["documents_without_chunks"]
        axis.annotate(f"{payload['ingest_report']['chunks_created']} chunks\n{dropped} docs without chunks", (x, 0.02), ha="center", va="bottom", fontsize=7, color="#333333")
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylim(0, 1.18)
    axis.set_xlabel("chunking configuration")
    axis.set_ylabel(f"document-level quality (k={payloads[0]['k']}, {payloads[0]['quality']['answerable_queries']} answerable queries)")
    axis.legend(fontsize=8, loc="upper center", ncol=3)
    return finish(fig, axis, "Retrieval quality vs chunk size (sequential)", stamp_many(sources), output, "08-quality-vs-chunk-size.png")


PLOTS = {
    1: plot_ingestion_vs_chunks,
    2: plot_latency_vs_dataset,
    3: plot_latency_vs_workers,
    4: plot_latency_vs_partitions,
    5: plot_worker_time_distribution,
    6: plot_merge_vs_candidates,
    7: plot_overlap_vs_partitions,
    8: plot_quality_vs_chunk_size,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="./results")
    parser.add_argument("--output", default="./figures")
    parser.add_argument("--only", nargs="+", type=int, choices=sorted(PLOTS), help="plot numbers to regenerate (default: all)")
    parser.add_argument(
        "--no-source-stamp",
        action="store_true",
        help="omit the source-file footer (for figures embedded in a document)",
    )
    parser.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="KIND=FILE",
        help="pin a result kind to one file, e.g. --pin correctness=correctness-X.json "
        "(repeatable); without it each kind uses its newest file",
    )
    args = parser.parse_args(argv)
    global SOURCE_STAMP
    SOURCE_STAMP = not args.no_source_stamp
    for pin in args.pin:
        kind, _, name = pin.partition("=")
        if not kind or not name:
            raise SystemExit(f"--pin expects KIND=FILE, got {pin!r}")
        PINNED[kind] = name
    results, output = Path(args.results), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for number in args.only or sorted(PLOTS):
        path = PLOTS[number](results, output)
        print(f"[{number}] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
