#!/usr/bin/env python3
"""Measure real CLI latency, sampled process-tree RSS, coverage and warm reuse.

Cold means fresh puzzle caches, not empty OS page caches or fresh downloads.
Uses registry cases and compares generation strategies at the same bag budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from anagram_paths import BENCHMARK_DIR, PROJECT_DIR
from anagram_suite import case_by_id, cases_for, normal_user_case


def summarize(runs: list[dict]) -> list[dict]:
    groups = sorted({(row["case"], row["strategy"]) for row in runs})
    summary = []
    for case, strategy in groups:
        selected = [
            row for row in runs if (row["case"], row["strategy"]) == (case, strategy)
        ]
        cold = [row for row in selected if row["phase"] == "cold"]
        warm = [row for row in selected if row["phase"] == "warm"]
        if not cold or not warm:
            continue
        cold_median = statistics.median(row["seconds"] for row in cold)
        warm_median = statistics.median(row["seconds"] for row in warm)
        summary.append(
            {
                "case": case,
                "strategy": strategy,
                "samples": len(cold),
                "cold_median_seconds": cold_median,
                "warm_median_seconds": warm_median,
                "warm_speedup": cold_median / warm_median,
                "cold_peak_tree_rss_bytes": max(
                    row["peak_tree_rss_bytes"] for row in cold
                ),
                "warm_peak_tree_rss_bytes": max(
                    row["peak_tree_rss_bytes"] for row in warm
                ),
                "expected_bag_recall": sum(
                    row["expected_bag_generated"] for row in cold
                )
                / len(cold),
            }
        )
    return summary


def measure(command: list[str], timeout: float) -> tuple[dict, float, int]:
    try:
        import psutil
    except ImportError as exc:
        raise SystemExit(
            "Memory measurements require psutil: python -m pip install psutil==7.2.2"
        ) from exc

    peak = 0
    start = time.perf_counter()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        root = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:
                    tree = [root, *root.children(recursive=True)]
                except psutil.NoSuchProcess:
                    tree = []
                rss = 0
                for member in tree:
                    try:
                        rss += member.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                peak = max(peak, rss)
                if time.perf_counter() - start > timeout:
                    raise TimeoutError(f"CLI exceeded {timeout}s")
                time.sleep(0.02)
        finally:
            if process.poll() is None:
                try:
                    children = root.children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for member in reversed(children):
                    try:
                        member.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
        elapsed = time.perf_counter() - start
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read().decode("utf-8", errors="replace")
        if process.returncode:
            raise RuntimeError(
                stderr.read().decode("utf-8", errors="replace")[-3000:]
                or output[-3000:]
            )
    return json.loads(output), elapsed, peak


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Registry case id; default: performance suite",
    )
    parser.add_argument("--cap", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--output", type=Path, default=BENCHMARK_DIR / "user_performance.json"
    )
    args = parser.parse_args(argv)
    if (
        not math.isfinite(args.timeout)
        or min(args.cap, args.samples, args.timeout) <= 0
        or args.workers < 0
    ):
        parser.error(
            "cap, samples and timeout must be positive; workers must be nonnegative"
        )

    # Provision shared corpora once, outside measured cold-puzzle invocations.
    import anagram_generate as generator
    from anagram_rerank import DEFAULT_WORDNET_DIR, ensure_wordnet
    from anagram_user_lexicon import ensure_user_lexicon

    ensure_user_lexicon()
    ensure_wordnet(DEFAULT_WORDNET_DIR)
    generator.ensure_ngram_data(
        generator.DEFAULT_NGRAM_DIR, refresh=False, need_bigrams=True
    )
    cases = (
        [case_by_id(case_id) for case_id in args.case]
        if args.case
        else cases_for("performance")
    )
    report = {
        "platform": platform.platform(),
        "python": sys.version,
        "source_sha256": hashlib.sha256(
            b"".join(
                path.read_bytes() for path in sorted(PROJECT_DIR.glob("anagram_*.py"))
            )
        ).hexdigest(),
        "cold_definition": "fresh candidate, prepared, and ranking caches; runtime corpora and OS caches warm",
        "memory_definition": "peak sampled sum of root and descendant RSS every 20ms; shared pages can be counted more than once",
        "cap": args.cap,
        "workers": args.workers,
        "runs": [],
    }
    for case in cases:
        resolved = normal_user_case(case)
        options = [
            option
            for option in resolved.solver_args
            if option not in {"--verbose", "--rebuild", "--quick", "--exhaustive"}
        ]
        for strategy in ("prefix", "diverse"):
            for sample in range(args.samples):
                with tempfile.TemporaryDirectory(prefix="anagram_perf_") as tmp:
                    command = [
                        sys.executable,
                        str(PROJECT_DIR / "anagram_solver.py"),
                        resolved.target,
                        *options,
                        "--quick",
                        "--max-results",
                        str(args.cap),
                        "--workers",
                        str(args.workers),
                        "--search-strategy",
                        strategy,
                        "--work-root",
                        tmp,
                        "--json",
                    ]
                    cold_results = None
                    for phase in ("cold", "warm"):
                        value, elapsed, peak = measure(command, args.timeout)
                        if phase == "cold":
                            cold_results = value["results"]
                        elif value["results"] != cold_results:
                            raise RuntimeError(
                                f"Warm result drift: {resolved.id}/{strategy}"
                            )
                        expected_bag = sorted(
                            str(case.get("answer", "")).lower().split()
                        )
                        candidate_recall = False
                        # Use the real parser: pretty contractions normalize in
                        # exactly the same way as ordinary candidate ingestion.
                        from anagram_rerank_core import norm_token, parse_candidates

                        expected_bag = sorted(norm_token(word) for word in expected_bag)
                        for export in Path(tmp).rglob("candidates.txt"):
                            candidate_recall |= any(
                                sorted(row.words) == expected_bag
                                for row in parse_candidates(export)
                            )
                        rank = next(
                            (
                                row["rank"]
                                for row in value["results"]
                                if [norm_token(word) for word in row["phrase"].split()]
                                == [
                                    norm_token(word)
                                    for word in str(case.get("answer", "")).split()
                                ]
                            ),
                            None,
                        )
                        record = {
                            "case": resolved.id,
                            "strategy": strategy,
                            "sample": sample + 1,
                            "phase": phase,
                            "seconds": elapsed,
                            "peak_tree_rss_bytes": peak,
                            "expected_bag_generated": candidate_recall,
                            "displayed_answer_rank": rank,
                            "search": value["search"],
                        }
                        report["runs"].append(record)
                        print(json.dumps(record), flush=True)
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(report, indent=2) + "\n", encoding="utf-8"
                        )
    report["summary"] = summarize(report["runs"])
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nSummary:\n" + json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
