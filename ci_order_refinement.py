#!/usr/bin/env python3
"""Informational forced-beam A/B for bounded complete-order refinement."""

from __future__ import annotations

import json
from pathlib import Path

import anagram_benchmark as benchmark
import anagram_rerank as reranker
from anagram_order_refinement import augment_seed_pool

HERE = Path(__file__).resolve().parent


def _objective(order: tuple[str, ...], lex: reranker.WordNetLexicon) -> float:
    raw = reranker.local_grammar_raw(order, lex)
    grammar = reranker.grammar_normalize(raw)
    structure = reranker.phrase_structure(order, lex)
    return (
        0.38 * grammar
        + 0.44 * structure.norm
        + 0.12 * structure.valency
        + 0.06 * structure.coverage
    )


def _load_cases() -> list[dict[str, object]]:
    payload = json.loads((HERE / "anagram_benchmarks.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("invalid benchmark document")
    return [dict(item) for item in payload["cases"] if isinstance(item, dict)]


def main() -> int:
    wordnet_dir = reranker.ensure_wordnet(reranker.DEFAULT_WORDNET_DIR)
    lex = reranker.WordNetLexicon.load(wordnet_dir)
    evaluated_cases = 0
    recovered = 0
    improved_seed_best = 0
    total_extra_evaluations = 0
    total_added_orders = 0

    print("=== FORCED-BEAM K-OPT AUGMENTATION A/B ===")
    for case in _load_cases():
        answer = str(case.get("answer", ""))
        words = benchmark.tokens(answer)
        if not 5 <= len(words) <= 6:
            continue
        acceptable = {
            benchmark.phrase_key(str(value))
            for value in case.get("acceptable_orders", [answer])
            if isinstance(value, str)
        }
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
            seed_limit=8,
            max_window=5,
            max_rounds=3,
            max_evaluations_per_seed=384,
        )
        total_extra_evaluations += augmented.evaluated
        total_added_orders += max(0, len(augmented.candidates) - len(set(seed_orders)))
        augmented_target = any(
            benchmark.phrase_key(result.order) in acceptable
            for result in augmented.candidates
        )
        augmented_best = max(
            (result.score for result in augmented.candidates),
            default=seed_best_score,
        )
        recovered += int(augmented_target and not seed_target)
        improved_seed_best += int(augmented_best > seed_best_score + 1e-12)
        status = "RECOVER" if augmented_target and not seed_target else (
            "KEEP" if augmented_target else "MISS"
        )
        print(
            f"{status:7} {case.get('id', answer)!s:<24} "
            f"seed_target={int(seed_target)} augmented_target={int(augmented_target)} "
            f"added={max(0, len(augmented.candidates) - len(set(seed_orders))):>2} "
            f"bestΔ={augmented_best - seed_best_score:+.4f}"
        )

    print(f"cases:                   {evaluated_cases}")
    print(f"new target recoveries:   {recovered}")
    print(f"better full-score best:  {improved_seed_best}")
    print(f"added candidate orders:  {total_added_orders}")
    print(f"refinement evaluations:  {total_extra_evaluations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
