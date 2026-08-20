#!/usr/bin/env python3
"""Fail CI when the fast ordering benchmark materially regresses."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import cast

import anagram_benchmark as benchmark
import anagram_clause_validity as validity
import anagram_rerank as rerank
import anagram_rerank_core as core
from anagram_suite import ORDERING_GATE, case_by_id, load_cases

_GrammarFn = Callable[[Sequence[str], core.WordNetLexicon], float]
_StructureFn = Callable[
    [Sequence[str], core.WordNetLexicon], core.StructureResult
]


def main() -> int:
    cases = load_cases()
    wn_dir = rerank.ensure_wordnet(rerank.DEFAULT_WORDNET_DIR)
    print(f"Loading WordNet from {wn_dir} ...")
    lex = rerank.WordNetLexicon.load(wn_dir)

    # The facade annotations expose the optimized subclass, but both callbacks
    # implement the base WordNetLexicon protocol used by the shared objective.
    # Widen only the static callable view here; no runtime conversion occurs.
    local_grammar_raw = cast(_GrammarFn, rerank.local_grammar_raw)
    phrase_structure = cast(_StructureFn, rerank.phrase_structure)

    t0 = time.perf_counter()
    results = [benchmark.run_order_case(rerank, lex, case) for case in cases]
    benchmark.print_order_summary(results)
    observed = benchmark.compute_order_metrics(results)
    if not observed:
        raise RuntimeError("Ordering benchmark produced no exact-rank cases")

    thresholds = {
        "recall1": ORDERING_GATE.min_recall_1,
        "recall10": ORDERING_GATE.min_recall_10,
        "recall50": ORDERING_GATE.min_recall_50,
        "mrr": ORDERING_GATE.min_mrr,
    }
    failures = [
        f"{name}={observed[name]:.3f} < {minimum:.3f}"
        for name, minimum in thresholds.items()
        if observed[name] + 1e-12 < minimum
    ]

    intended_case = case_by_id(ORDERING_GATE.cross_bag_case_id)
    intended_words = benchmark.tokens(str(intended_case["answer"]))
    intended_objective = validity.grammar_structure_objective(
        intended_words,
        lex,
        local_grammar_raw,
        phrase_structure,
    )
    malformed_objectives = [
        (
            validity.grammar_structure_objective(
                words,
                lex,
                local_grammar_raw,
                phrase_structure,
            ),
            words,
        )
        for words in ORDERING_GATE.malformed_bags
    ]
    strongest_bad_score, strongest_bad_words = max(malformed_objectives)
    cross_bag_margin = intended_objective - strongest_bad_score
    if cross_bag_margin + 1e-12 < ORDERING_GATE.min_cross_bag_margin:
        failures.append(
            "cross-bag grammar margin "
            f"{cross_bag_margin:.3f} < {ORDERING_GATE.min_cross_bag_margin:.3f} "
            f"against {' '.join(strongest_bad_words)}"
        )

    result_by_id = {result.case_id: result for result in results}
    target_ranks: dict[str, int | None] = {}
    for case_id, maximum in ORDERING_GATE.target_max_ranks:
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
        f"{cross_bag_margin:.3f}  minimum margin "
        f"{ORDERING_GATE.min_cross_bag_margin:.3f}"
    )
    for case_id, maximum in ORDERING_GATE.target_max_ranks:
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
