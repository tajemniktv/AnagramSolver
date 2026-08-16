#!/usr/bin/env python3
"""Fail CI when the fast ordering benchmark materially regresses."""

from __future__ import annotations

import time

import anagram_benchmark as benchmark
import anagram_clause_validity as validity
import anagram_rerank as rerank

MIN_RECALL_1 = 0.35
MIN_RECALL_10 = 0.79
MIN_RECALL_50 = 0.95
MIN_MRR = 0.47
MIN_CROSS_BAG_MARGIN = 0.02

_CROSS_BAG_INTENDED = ("i", "am", "testing", "anagrams")
_CROSS_BAG_MALFORMED = (
    ("a", "am", "sitting", "managers"),
    ("an", "game", "starting", "aims"),
)
_TARGET_MAX_RANKS = {
    "actions_words": 10,
    # Four structurally symmetric parallel variants tie at the same objective;
    # do not turn their incidental permutation enumeration into a grammar rule.
    "united_stand": 4,
}


def main() -> int:
    cases = benchmark.load_cases(benchmark.DEFAULT_CASES, set())
    wn_dir = rerank.ensure_wordnet(rerank.DEFAULT_WORDNET_DIR)
    print(f"Loading WordNet from {wn_dir} ...")
    lex = rerank.WordNetLexicon.load(wn_dir)

    t0 = time.perf_counter()
    results = [benchmark.run_order_case(rerank, lex, case) for case in cases]
    benchmark.print_order_summary(results)
    observed = benchmark.compute_order_metrics(results)
    if not observed:
        raise RuntimeError("Ordering benchmark produced no exact-rank cases")

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

    intended_objective = validity.grammar_structure_objective(
        _CROSS_BAG_INTENDED,
        lex,
        rerank.local_grammar_raw,
        rerank.phrase_structure,
    )
    malformed_objectives = [
        (
            validity.grammar_structure_objective(
                words,
                lex,
                rerank.local_grammar_raw,
                rerank.phrase_structure,
            ),
            words,
        )
        for words in _CROSS_BAG_MALFORMED
    ]
    strongest_bad_score, strongest_bad_words = max(malformed_objectives)
    cross_bag_margin = intended_objective - strongest_bad_score
    if cross_bag_margin + 1e-12 < MIN_CROSS_BAG_MARGIN:
        failures.append(
            "cross-bag grammar margin "
            f"{cross_bag_margin:.3f} < {MIN_CROSS_BAG_MARGIN:.3f} "
            f"against {' '.join(strongest_bad_words)}"
        )

    result_by_id = {result.case_id: result for result in results}
    target_ranks: dict[str, int | None] = {}
    for case_id, maximum in _TARGET_MAX_RANKS.items():
        result = result_by_id.get(case_id)
        rank = None if result is None else result.exact_rank
        target_ranks[case_id] = rank
        if rank is None or rank > maximum:
            failures.append(f"{case_id} rank={rank!r} > {maximum}")

    print("\n=== CI ORDERING GATE ===")
    for name in ("recall1", "recall10", "recall50", "mrr"):
        print(f"  {name:<9} {observed[name]:.3f}  minimum {thresholds[name]:.3f}")
    print(
        "  cross-bag "
        f"{cross_bag_margin:.3f}  minimum margin {MIN_CROSS_BAG_MARGIN:.3f}"
    )
    for case_id, maximum in _TARGET_MAX_RANKS.items():
        print(f"  {case_id:<13} rank={target_ranks[case_id]!r}  maximum {maximum}")
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
