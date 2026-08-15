#!/usr/bin/env python3
"""Fail CI when the fast ordering benchmark materially regresses."""

from __future__ import annotations

import time

import anagram_benchmark as benchmark
import anagram_rerank as rerank

MIN_RECALL_1 = 0.35
MIN_RECALL_10 = 0.79
MIN_RECALL_50 = 0.95
MIN_MRR = 0.47


def metrics(results: list[benchmark.OrderResult]) -> dict[str, float]:
    exact = [result for result in results if result.exact_rank is not None]
    if not exact:
        raise RuntimeError("Ordering benchmark produced no exact-rank cases")
    n = len(exact)
    return {
        "recall1": sum(result.exact_rank <= 1 for result in exact) / n,
        "recall10": sum(result.exact_rank <= 10 for result in exact) / n,
        "recall50": sum(result.exact_rank <= 50 for result in exact) / n,
        "mrr": sum(1.0 / result.exact_rank for result in exact) / n,
    }


def main() -> int:
    cases = benchmark.load_cases(benchmark.DEFAULT_CASES, set())
    wn_dir = rerank.ensure_wordnet(rerank.DEFAULT_WORDNET_DIR)
    print(f"Loading WordNet from {wn_dir} ...")
    lex = rerank.WordNetLexicon.load(wn_dir)

    t0 = time.perf_counter()
    results = [benchmark.run_order_case(rerank, lex, case) for case in cases]
    benchmark.print_order_summary(results)
    observed = metrics(results)

    thresholds = {
        "recall1": MIN_RECALL_1,
        "recall10": MIN_RECALL_10,
        "recall50": MIN_RECALL_50,
        "mrr": MIN_MRR,
    }
    failures = [
        f"{name}={observed[name]:.3f} < {minimum:.3f}"
        for name, minimum in thresholds.items()
        if observed[name] + 1e-12 < minimum
    ]

    print("\n=== CI ORDERING GATE ===")
    for name in ("recall1", "recall10", "recall50", "mrr"):
        print(f"  {name:<9} {observed[name]:.3f}  minimum {thresholds[name]:.3f}")
    print(f"  wall time {time.perf_counter() - t0:.2f}s")

    if failures:
        print("\nOrdering regression gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOrdering regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
