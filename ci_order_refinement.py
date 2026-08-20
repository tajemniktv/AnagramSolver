#!/usr/bin/env python3
"""Informational forced-beam A/B for bounded complete-order refinement."""

from __future__ import annotations

from collections.abc import Sequence

import anagram_benchmark as benchmark
import anagram_rerank as reranker
import anagram_rerank_core as core
from anagram_order_refinement import augment_seed_pool
from anagram_suite import load_cases


def _objective(order: tuple[str, ...], lex: reranker.WordNetLexicon) -> float:
    raw = reranker.local_grammar_raw(order, lex)
    grammar = core.grammar_normalize(raw)
    structure = reranker.phrase_structure(order, lex)
    return (
        0.38 * grammar
        + 0.44 * structure.norm
        + 0.12 * structure.valency
        + 0.06 * structure.coverage
    )


def _acceptable(case: dict[str, object], answer: str) -> set[tuple[str, ...]]:
    raw: object = case.get("acceptable_orders", [answer])
    values: Sequence[object] = raw if isinstance(raw, list) else [answer]
    return {
        benchmark.phrase_key(item)
        for item in values
        if isinstance(item, str)
    }


def _run_configuration(
    cases: list[dict[str, object]],
    lex: reranker.WordNetLexicon,
    *,
    seed_limit: int,
    max_window: int,
    max_rounds: int,
    max_evaluations_per_seed: int,
) -> tuple[int, int, int, int, int]:
    evaluated_cases = 0
    recovered = 0
    improved_seed_best = 0
    total_evaluations = 0
    total_added_orders = 0

    print(
        "\n--- "
        f"refine-first={seed_limit} window<={max_window} rounds={max_rounds} "
        f"eval/refined-seed<={max_evaluations_per_seed} ---"
    )
    for case in cases:
        answer = str(case.get("answer", ""))
        words = benchmark.tokens(answer)
        if not 5 <= len(words) <= 6:
            continue
        acceptable = _acceptable(case, answer)
        seeds, _ = reranker.rank_orders(
            words,
            lex,
            order_mode="beam",
            beam_width=64,
            exact_max_words=0,
            top_k=8,
        )
        if not seeds:
            continue
        evaluated_cases += 1

        seed_orders = tuple(candidate.order for candidate in seeds)
        seed_target = any(benchmark.phrase_key(order) in acceptable for order in seed_orders)
        seed_best_score = max(candidate.objective for candidate in seeds)

        augmented = augment_seed_pool(
            seed_orders,
            lambda order: _objective(order, lex),
            seed_limit=seed_limit,
            max_window=max_window,
            max_rounds=max_rounds,
            max_evaluations_per_seed=max_evaluations_per_seed,
        )
        total_evaluations += augmented.evaluated
        added = max(0, len(augmented.candidates) - len(set(seed_orders)))
        total_added_orders += added
        augmented_target = any(
            benchmark.phrase_key(result.order) in acceptable
            for result in augmented.candidates
        )
        augmented_best = max(
            (result.score for result in augmented.candidates),
            default=seed_best_score,
        )
        recovered_now = augmented_target and not seed_target
        improved_now = augmented_best > seed_best_score + 1e-12
        recovered += int(recovered_now)
        improved_seed_best += int(improved_now)
        if recovered_now or improved_now:
            print(
                f"{case.get('id', answer)!s:<24} "
                f"recover={int(recovered_now)} added={added:>2} "
                f"bestΔ={augmented_best - seed_best_score:+.4f}"
            )

    print(
        f"summary cases={evaluated_cases} recoveries={recovered} "
        f"better_best={improved_seed_best} added={total_added_orders} "
        f"scorer_evaluations={total_evaluations}"
    )
    return (
        evaluated_cases,
        recovered,
        improved_seed_best,
        total_added_orders,
        total_evaluations,
    )


def main() -> int:
    wordnet_dir = reranker.ensure_wordnet(reranker.DEFAULT_WORDNET_DIR)
    lex = reranker.WordNetLexicon.load(wordnet_dir)
    cases = load_cases()

    print("=== FORCED-BEAM K-OPT AUGMENTATION BUDGET A/B ===")
    configurations = (
        (1, 5, 2, 384),
        (2, 5, 2, 384),
        (4, 4, 2, 128),
    )
    results = [
        (
            config,
            _run_configuration(
                cases,
                lex,
                seed_limit=config[0],
                max_window=config[1],
                max_rounds=config[2],
                max_evaluations_per_seed=config[3],
            ),
        )
        for config in configurations
    ]

    print("\n=== K-OPT BUDGET SUMMARY ===")
    for config, result in results:
        _, recovered, improved, added, evaluated = result
        print(
            f"refine-first={config[0]} window<={config[1]} rounds={config[2]} "
            f"eval/refined-seed<={config[3]}: recoveries={recovered} "
            f"better_best={improved} added={added} scorer_evaluations={evaluated}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
