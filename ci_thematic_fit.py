#!/usr/bin/env python3
"""Informational A/B for generic corpus-derived thematic-fit evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import anagram_benchmark as benchmark
import anagram_rerank as reranker
from anagram_thematic_fit import score_thematic_fit

HERE = Path(__file__).resolve().parent


def _load_cases() -> list[dict[str, object]]:
    payload = json.loads((HERE / "anagram_benchmarks.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("invalid benchmark document")
    return [dict(item) for item in payload["cases"] if isinstance(item, dict)]


def _metrics(ranks: list[int]) -> tuple[float, float]:
    if not ranks:
        return 0.0, 0.0
    return (
        sum(rank == 1 for rank in ranks) / len(ranks),
        sum(1.0 / rank for rank in ranks) / len(ranks),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase-db", type=Path, required=True)
    parser.add_argument("--bonus", type=float, default=0.08)
    parser.add_argument("--order-candidates", type=int, default=56)
    args = parser.parse_args()
    if args.bonus < 0.0:
        parser.error("--bonus must be >= 0")

    wordnet_dir = reranker.ensure_wordnet(reranker.DEFAULT_WORDNET_DIR)
    lex = reranker.WordNetLexicon.load(wordnet_dir)
    phrase_index = cast(reranker.PhraseIndex, reranker.PhraseIndex.open(args.phrase_db))
    baseline_ranks: list[int] = []
    thematic_ranks: list[int] = []
    ordinary_baseline: list[int] = []
    ordinary_thematic: list[int] = []

    try:
        print("=== CORPUS-DERIVED THEMATIC-FIT A/B ===")
        for case in _load_cases():
            answer = str(case.get("answer", ""))
            words = benchmark.tokens(answer)
            if not 2 <= len(words) <= 6:
                continue
            raw_acceptable = case.get("acceptable_orders", [answer])
            if not isinstance(raw_acceptable, list):
                continue
            acceptable = {benchmark.phrase_key(str(value)) for value in raw_acceptable}
            candidates, _ = reranker.rank_orders(
                words,
                lex,
                order_mode="exact",
                beam_width=256,
                exact_max_words=6,
                top_k=args.order_candidates,
            )
            scored: list[tuple[float, float, tuple[str, ...], float]] = []
            for candidate in candidates:
                phrase_score, _ = phrase_index.score(candidate.order)
                baseline = candidate.objective + 0.10 * phrase_score
                thematic = score_thematic_fit(
                    candidate.order,
                    counts=phrase_index.counts,
                    is_verb=lambda word: lex.features(word).verb,
                    is_nominal=lambda word: lex.features(word).noun,
                )
                scored.append(
                    (
                        baseline,
                        baseline + args.bonus * thematic.score,
                        candidate.order,
                        thematic.score,
                    )
                )

            baseline_sorted = sorted(scored, key=lambda item: (-item[0], item[2]))
            thematic_sorted = sorted(scored, key=lambda item: (-item[1], -item[3], item[2]))
            baseline_hits = [
                index
                for index, item in enumerate(baseline_sorted, 1)
                if benchmark.phrase_key(item[2]) in acceptable
            ]
            thematic_hits = [
                index
                for index, item in enumerate(thematic_sorted, 1)
                if benchmark.phrase_key(item[2]) in acceptable
            ]
            if not baseline_hits or not thematic_hits:
                continue
            baseline_rank = min(baseline_hits)
            thematic_rank = min(thematic_hits)
            baseline_ranks.append(baseline_rank)
            thematic_ranks.append(thematic_rank)
            category = str(case.get("category", ""))
            if category.startswith("ordinary-"):
                ordinary_baseline.append(baseline_rank)
                ordinary_thematic.append(thematic_rank)

            case_id = str(case.get("id", answer))
            if baseline_rank != thematic_rank or case_id in {"dog_ball", "phone_charge"}:
                target_item = next(
                    (
                        item
                        for item in thematic_sorted
                        if benchmark.phrase_key(item[2]) in acceptable
                    ),
                    None,
                )
                target_fit = target_item[3] if target_item is not None else 0.0
                print(
                    f"{case.get('id', answer)!s:<24} "
                    f"baseline={baseline_rank:>2} thematic={thematic_rank:>2} "
                    f"best={' '.join(thematic_sorted[0][2])} "
                    f"best_fit={thematic_sorted[0][3]:.3f} target_fit={target_fit:.3f}"
                )
    finally:
        phrase_index.connection.close()

    base_r1, base_mrr = _metrics(baseline_ranks)
    fit_r1, fit_mrr = _metrics(thematic_ranks)
    ordinary_base_r1, ordinary_base_mrr = _metrics(ordinary_baseline)
    ordinary_fit_r1, ordinary_fit_mrr = _metrics(ordinary_thematic)
    print(f"all groups:      baseline R@1={base_r1:.3f} MRR={base_mrr:.3f}")
    print(f"all groups:      thematic R@1={fit_r1:.3f} MRR={fit_mrr:.3f}")
    print(f"all held-out Δ:  R@1={fit_r1 - base_r1:+.3f} MRR={fit_mrr - base_mrr:+.3f}")
    print(
        f"ordinary subset: baseline R@1={ordinary_base_r1:.3f} "
        f"MRR={ordinary_base_mrr:.3f}"
    )
    print(
        f"ordinary subset: thematic R@1={ordinary_fit_r1:.3f} "
        f"MRR={ordinary_fit_mrr:.3f}"
    )
    print(
        f"ordinary Δ:      R@1={ordinary_fit_r1 - ordinary_base_r1:+.3f} "
        f"MRR={ordinary_fit_mrr - ordinary_base_mrr:+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
