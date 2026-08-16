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
_CONSTRUCTION_DIAGNOSTICS = (
    ("comparative", ("actions", "speak", "louder", "than", "words")),
    ("parallel", ("united", "we", "stand", "divided", "we", "fall")),
)


def _print_construction_diagnostic(
    name: str,
    words: tuple[str, ...],
    lex: rerank.WordNetLexicon,
) -> None:
    candidates, _evaluated = rerank.rank_orders(
        words,
        lex,
        order_mode="exact",
        top_k=720,
    )
    target = next(
        (candidate for candidate in candidates if candidate.order == words),
        None,
    )
    print(f"\n{name} construction diagnostic:")
    if target is None:
        print("  target was not retained")
    else:
        rank = candidates.index(target) + 1
        print(
            f"  target rank={rank}/{len(candidates)} "
            f"objective={target.objective:.4f} grammar={target.grammar_norm:.4f} "
            f"structure={target.structure_norm:.4f} valency={target.valency_norm:.4f} "
            f"coverage={target.syntax_coverage:.4f} kind={target.phrase_kind}"
        )
    for index, candidate in enumerate(candidates[:5], start=1):
        print(
            f"  #{index:<2} {' '.join(candidate.order):<42} "
            f"obj={candidate.objective:.4f} grammar={candidate.grammar_norm:.4f} "
            f"structure={candidate.structure_norm:.4f} kind={candidate.phrase_kind}"
        )


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

    print("\n=== CI ORDERING GATE ===")
    for name in ("recall1", "recall10", "recall50", "mrr"):
        print(f"  {name:<9} {observed[name]:.3f}  minimum {thresholds[name]:.3f}")
    print(
        "  cross-bag "
        f"{cross_bag_margin:.3f}  minimum margin {MIN_CROSS_BAG_MARGIN:.3f}"
    )
    print(f"  wall time {time.perf_counter() - t0:.2f}s")

    for name, words in _CONSTRUCTION_DIAGNOSTICS:
        _print_construction_diagnostic(name, words, lex)

    if failures:
        print("\nOrdering regression gate FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOrdering regression gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
