from __future__ import annotations

import itertools
import unittest
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from unittest.mock import patch

import anagram_order_diversity as diversity


@dataclass(frozen=True, slots=True)
class Candidate:
    order: tuple[str, ...]
    objective: float
    phrase_kind: str


def reference_similarity(left: Candidate, right: Candidate) -> float:
    a = left.order
    b = right.order
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    left_pairs = Counter(pairwise(a))
    right_pairs = Counter(pairwise(b))
    overlap = sum((left_pairs & right_pairs).values())
    adjacency = overlap / max(sum(left_pairs.values()), sum(right_pairs.values()), 1)
    position = sum(x == y for x, y in zip(a, b)) / max(len(a), len(b))
    endpoints = 0.5 * float(a[0] == b[0]) + 0.5 * float(a[-1] == b[-1])
    same_kind = float(left.phrase_kind == right.phrase_kind)
    return max(
        0.0,
        min(
            1.0,
            0.55 * adjacency
            + 0.30 * position
            + 0.10 * endpoints
            + 0.05 * same_kind,
        ),
    )


def reference_select(
    candidates: tuple[Candidate, ...],
    top_k: int,
    *,
    quality_core: int = diversity.DEFAULT_QUALITY_CORE,
    diversity_strength: float = diversity.DEFAULT_DIVERSITY_STRENGTH,
) -> tuple[Candidate, ...]:
    limit = min(top_k, len(candidates))
    core_count = min(quality_core, limit)
    selected_indices = list(range(core_count))
    remaining = set(range(core_count, len(candidates)))

    while len(selected_indices) < limit and remaining:
        best_index = None
        best_key = None
        for index in remaining:
            candidate = candidates[index]
            max_similarity = max(
                reference_similarity(candidate, candidates[chosen])
                for chosen in selected_indices
            )
            utility = candidate.objective - diversity_strength * max_similarity
            key = (-utility, -candidate.objective, candidate.order)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        assert best_index is not None
        selected_indices.append(best_index)
        remaining.remove(best_index)

    selected_indices.sort()
    return tuple(candidates[index] for index in selected_indices)


class OptimizedOrderDiversityTests(unittest.TestCase):
    def _pool(self) -> tuple[Candidate, ...]:
        orders = list(itertools.islice(itertools.permutations("abcdef"), 72))
        return tuple(
            Candidate(
                order=tuple(order),
                objective=1.0 - index / 1000.0,
                phrase_kind=("clause" if index % 3 else "noun-phrase"),
            )
            for index, order in enumerate(orders)
        )

    def test_order_similarity_matches_reference_with_repeated_tokens(self) -> None:
        cases = (
            (
                Candidate(("the", "cat", "the", "sat"), 1.0, "clause"),
                Candidate(("the", "sat", "the", "cat"), 0.9, "clause"),
            ),
            (
                Candidate(("a",), 1.0, "fragment"),
                Candidate(("a",), 0.9, "fragment"),
            ),
            (
                Candidate((), 1.0, "fragment"),
                Candidate(("a",), 0.9, "fragment"),
            ),
        )
        for left, right in cases:
            with self.subTest(left=left.order, right=right.order):
                self.assertAlmostEqual(
                    diversity.order_similarity(left, right),
                    reference_similarity(left, right),
                    places=12,
                )

    def test_optimized_selector_matches_reference(self) -> None:
        pool = self._pool()
        self.assertEqual(
            diversity.select_diverse_orders(pool, 64),
            reference_select(pool, 64),
        )

    def test_selector_equivalence_with_repeated_tokens_and_ties(self) -> None:
        orders = sorted(set(itertools.permutations(("a", "a", "b", "c"))))
        pool = tuple(
            Candidate(
                order=tuple(order),
                objective=1.0 - (index // 2) * 0.01,
                phrase_kind=("clause" if index % 2 else "fragment"),
            )
            for index, order in enumerate(orders)
        )
        kwargs = {"quality_core": 3, "diversity_strength": 0.27}
        self.assertEqual(
            diversity.select_diverse_orders(pool, 8, **kwargs),
            reference_select(pool, 8, **kwargs),
        )

    def test_default_72_to_64_pool_uses_incremental_comparisons(self) -> None:
        pool = self._pool()
        calls = 0
        original = diversity._fingerprint_similarity

        def counted(left, right):
            nonlocal calls
            calls += 1
            return original(left, right)

        with patch.object(diversity, "_fingerprint_similarity", side_effect=counted):
            actual = diversity.select_diverse_orders(pool, 64)

        self.assertEqual(actual, reference_select(pool, 64))
        self.assertLessEqual(calls, 1400)

    def test_quality_core_only_path_avoids_similarity_work(self) -> None:
        pool = self._pool()
        with patch.object(
            diversity,
            "_fingerprint_similarity",
            side_effect=AssertionError("similarity should not run"),
        ):
            self.assertEqual(diversity.select_diverse_orders(pool, 48), pool[:48])


if __name__ == "__main__":
    unittest.main()
