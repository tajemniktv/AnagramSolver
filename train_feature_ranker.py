"""Train/evaluate the explicit-feature order ranker on grouped benchmark bags.

The regression answers are labels, never scorer inputs. Cross-validation assigns
an entire unordered word bag to one fold, so no permutation of a held-out answer
can appear in its training set. Saving a model is optional and is deliberately
separate from the held-out report because all-group fitted metrics are not an
honest generalization estimate.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import cast

import anagram_benchmark as benchmark
import anagram_rerank as reranker
from anagram_feature_ranker import (
    FEATURE_NAMES,
    RankGroup,
    RankItem,
    cross_validate_pairwise_ranker,
    explicit_order_features,
    train_pairwise_ranker,
)
from anagram_suite import DEFAULT_CASES, load_cases


def _group_key(words: tuple[str, ...]) -> str:
    return " ".join(sorted(words))


def build_groups(
    cases: list[dict[str, object]],
    *,
    lex: reranker.WordNetLexicon,
    phrase_index: reranker.PhraseIndex | None,
    order_candidates: int,
    phrase_bonus_max: float,
) -> tuple[list[RankGroup], list[str]]:
    groups: list[RankGroup] = []
    skipped: list[str] = []

    for case in cases:
        answer = str(case["answer"])
        words = benchmark.tokens(answer)
        case_id = str(case.get("id", answer))
        if not 2 <= len(words) <= 6:
            skipped.append(f"{case_id}: word-count {len(words)} outside exact training range")
            continue

        raw_acceptable = case.get("acceptable_orders", [answer])
        if not isinstance(raw_acceptable, list):
            skipped.append(f"{case_id}: acceptable_orders is not a list")
            continue
        acceptable = {benchmark.phrase_key(str(value)) for value in raw_acceptable}
        candidates, _ = reranker.rank_orders(
            words,
            lex,
            order_mode="exact",
            beam_width=256,
            exact_max_words=6,
            top_k=order_candidates,
        )
        if not candidates:
            skipped.append(f"{case_id}: no retained orders")
            continue

        items: list[RankItem] = []
        target_retained = False
        for candidate in candidates:
            phrase_score = 0.0
            phrase_details: dict[str, float] = {}
            if phrase_index is not None:
                phrase_score, phrase_details = phrase_index.score(candidate.order)
            positive = benchmark.phrase_key(candidate.order) in acceptable
            target_retained = target_retained or positive
            baseline_score = candidate.objective + (
                phrase_bonus_max / 100.0
            ) * phrase_score
            features = explicit_order_features(
                grammar_norm=candidate.grammar_norm,
                structure_norm=candidate.structure_norm,
                valency_norm=candidate.valency_norm,
                syntax_coverage=candidate.syntax_coverage,
                objective=candidate.objective,
                phrase_score=phrase_score,
                phrase_details=phrase_details,
                word_count=len(words),
            )
            items.append(
                RankItem(
                    key=" ".join(candidate.order),
                    features=features,
                    positive=positive,
                    baseline_score=baseline_score,
                )
            )

        if not target_retained:
            skipped.append(f"{case_id}: acceptable order not retained in top-{order_candidates}")
            continue
        groups.append(RankGroup(key=_group_key(words), items=tuple(items)))

    return groups, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--phrase-db", type=Path)
    parser.add_argument("--order-candidates", type=int, default=56)
    parser.add_argument("--phrase-bonus-max", type=float, default=10.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.order_candidates < 2:
        parser.error("--order-candidates must be >= 2")
    if not math.isfinite(args.phrase_bonus_max) or args.phrase_bonus_max < 0.0:
        parser.error("--phrase-bonus-max must be finite and >= 0")
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0:
        parser.error("--learning-rate must be finite and > 0")
    if not math.isfinite(args.l2) or args.l2 < 0.0:
        parser.error("--l2 must be finite and >= 0")

    wordnet_dir = reranker.ensure_wordnet(reranker.DEFAULT_WORDNET_DIR)
    lex = reranker.WordNetLexicon.load(wordnet_dir)
    phrase_index: reranker.PhraseIndex | None = (
        cast(reranker.PhraseIndex, reranker.PhraseIndex.open(args.phrase_db))
        if args.phrase_db is not None
        else None
    )
    try:
        groups, skipped = build_groups(
            load_cases(args.cases, require_ids=False),
            lex=lex,
            phrase_index=phrase_index,
            order_candidates=args.order_candidates,
            phrase_bonus_max=args.phrase_bonus_max,
        )
        baseline, learned = cross_validate_pairwise_ranker(
            groups,
            folds=args.folds,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )

        print("=== EXPLICIT-FEATURE RANKER / GROUPED HELD-OUT CV ===")
        print(f"groups:     {len(groups)}")
        print(f"features:   {len(FEATURE_NAMES)}")
        print(f"folds:      {args.folds}")
        print(f"phrase DB:  {args.phrase_db if args.phrase_db else 'none'}")
        print(
            f"baseline:   R@1={baseline.recall1:.3f} MRR={baseline.mrr:.3f} "
            f"groups={baseline.groups}"
        )
        print(
            f"learned:    R@1={learned.recall1:.3f} MRR={learned.mrr:.3f} "
            f"groups={learned.groups}"
        )
        print(
            f"held-out Δ: R@1={learned.recall1 - baseline.recall1:+.3f} "
            f"MRR={learned.mrr - baseline.mrr:+.3f}"
        )
        if skipped:
            print("skipped:")
            for reason in skipped:
                print(f"  - {reason}")

        if args.output is not None:
            model = train_pairwise_ranker(
                groups,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                l2=args.l2,
            )
            model.save(args.output)
            print(
                f"saved all-group fit to {args.output} (deployment artifact only; "
                "held-out metrics above remain the evaluation)"
            )
    finally:
        if phrase_index is not None:
            phrase_index.connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
