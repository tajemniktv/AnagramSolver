from __future__ import annotations

import copy
import itertools
import pickle
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import anagram_order_diversity as diversity
import anagram_rerank as rerank
import anagram_solver as solver


@dataclass(frozen=True)
class _Candidate:
    order: tuple[str, ...]
    objective: float
    phrase_kind: str = "clause"


@dataclass(frozen=True, slots=True)
class _ReferenceCandidate:
    order: tuple[str, ...]
    objective: float
    phrase_kind: str


def _reference_similarity(
    left: _ReferenceCandidate,
    right: _ReferenceCandidate,
) -> float:
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


def _reference_select(
    candidates: tuple[_ReferenceCandidate, ...],
    top_k: int,
    *,
    quality_core: int = diversity.DEFAULT_QUALITY_CORE,
    diversity_strength: float = diversity.DEFAULT_DIVERSITY_STRENGTH,
) -> tuple[_ReferenceCandidate, ...]:
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
                _reference_similarity(candidate, candidates[chosen])
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


def _order_candidates(count: int = 64) -> tuple[rerank.OrderCandidate, ...]:
    return tuple(
        rerank.OrderCandidate(
            order=(f"w{i}", "a", "b"),
            grammar_raw=2.0,
            grammar_norm=0.8,
            structure_norm=0.8,
            valency_norm=0.8,
            syntax_coverage=0.8,
            phrase_kind="clause",
            objective=1.0 - i * 0.005,
        )
        for i in range(count)
    )


class OrderDiversityTests(unittest.TestCase):
    def tearDown(self) -> None:
        rerank._clear_order_side_tables()

    def test_direct_facade_default_initializes_worker_retention(self) -> None:
        self.assertEqual(
            rerank.impl._ORDER_CANDIDATE_COUNT,
            rerank.DEFAULT_ORDER_CANDIDATES,
        )

    def test_raw_pool_widens_only_after_quality_core(self) -> None:
        self.assertEqual(diversity.raw_pool_size(8), 8)
        self.assertEqual(diversity.raw_pool_size(16), 16)
        self.assertEqual(diversity.raw_pool_size(32), 32)
        self.assertEqual(diversity.raw_pool_size(48), 48)
        self.assertEqual(diversity.raw_pool_size(56), 64)
        self.assertEqual(diversity.raw_pool_size(64), 72)
        self.assertEqual(diversity.raw_pool_size(256), 256)

    def test_raw_pool_size_rejects_invalid_parameters(self) -> None:
        cases = (
            ((0,), {}),
            ((-1,), {}),
            ((8,), {"quality_core": 0}),
            ((8,), {"quality_core": -1}),
            ((8,), {"pool_extra": -1}),
            ((8,), {"max_pool": 0}),
            ((8,), {"max_pool": -1}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    diversity.raw_pool_size(*args, **kwargs)

    def test_small_k_preserves_historical_score_prefix_exactly(self) -> None:
        candidates = tuple(
            _Candidate((str(i),), 1.0 - i * 0.01)
            for i in range(56)
        )
        retained = diversity.select_diverse_orders(candidates, 48)
        self.assertEqual(retained, candidates[:48])

    def test_wider_retention_never_drops_score_top48(self) -> None:
        candidates = tuple(
            _Candidate((str(i), "x", "y"), 1.0 - i * 0.01)
            for i in range(64)
        )
        retained = diversity.select_diverse_orders(candidates, 56)
        self.assertEqual(len(retained), 56)
        for candidate in candidates[:48]:
            self.assertIn(candidate, retained)
        self.assertEqual(retained[0], candidates[0])

    def test_select_diverse_orders_rejects_invalid_parameters(self) -> None:
        candidates = (_Candidate(("a",), 1.0),)
        cases = (
            ((candidates, 0), {}),
            ((candidates, -1), {}),
            ((candidates, 1), {"quality_core": 0}),
            ((candidates, 1), {"quality_core": -1}),
            ((candidates, 1), {"diversity_strength": -0.01}),
            ((candidates, 1), {"diversity_strength": 1.01}),
        )
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    diversity.select_diverse_orders(*args, **kwargs)

    def test_select_diverse_orders_empty_candidates_returns_empty_tuple(self) -> None:
        self.assertEqual(diversity.select_diverse_orders((), 1), ())
        self.assertEqual(diversity.select_diverse_orders((), 56), ())

    def test_extra_slot_can_prefer_structural_novelty_over_near_duplicate(self) -> None:
        best = _Candidate(("a", "b", "c", "d"), 1.00)
        near = _Candidate(("a", "b", "d", "c"), 0.99)
        diverse = _Candidate(("d", "c", "b", "a"), 0.95)

        retained = diversity.select_diverse_orders(
            (best, near, diverse),
            2,
            quality_core=1,
        )

        self.assertEqual(retained, (best, diverse))
        self.assertGreater(
            diversity.order_similarity(best, near),
            diversity.order_similarity(best, diverse),
        )

    def test_order_similarity_identical_orders_is_one(self) -> None:
        left = _Candidate(("a", "b", "c"), 1.0, "clause")
        right = _Candidate(("a", "b", "c"), 0.5, "fragment")
        self.assertEqual(diversity.order_similarity(left, right), 1.0)

    def test_order_similarity_empty_vs_nonempty_is_zero(self) -> None:
        empty = _Candidate((), 1.0)
        nonempty = _Candidate(("a",), 1.0)
        self.assertEqual(diversity.order_similarity(empty, nonempty), 0.0)

    def test_repeated_token_adjacency_uses_multiset_overlap(self) -> None:
        self.assertAlmostEqual(
            diversity._adjacency_similarity(
                ("a", "a", "b"),
                ("a", "b", "b"),
            ),
            0.5,
        )

    def test_order_similarity_phrase_kind_contributes_for_distinct_orders(self) -> None:
        left = _Candidate(("a", "b", "c"), 1.0, "clause")
        same_kind = _Candidate(("a", "c", "b"), 0.9, "clause")
        different_kind = _Candidate(("a", "c", "b"), 0.9, "fragment")
        self.assertGreater(
            diversity.order_similarity(left, same_kind),
            diversity.order_similarity(left, different_kind),
        )

    def test_adjacency_similarity_short_sequences(self) -> None:
        self.assertEqual(diversity._adjacency_similarity(("a",), ("a",)), 1.0)
        self.assertEqual(diversity._adjacency_similarity(("a",), ("b",)), 0.0)
        self.assertEqual(
            diversity._adjacency_similarity(("a", "b"), ("a", "b")),
            1.0,
        )
        self.assertEqual(
            diversity._adjacency_similarity(("a", "b"), ("b", "a")),
            0.0,
        )

    def test_facade_requests_top64_raw_pool_for_default_top56(self) -> None:
        candidates = _order_candidates()
        with patch.object(
            rerank,
            "_BASE_RANK_ORDERS",
            return_value=(candidates, 99),
        ) as base:
            retained, evaluated = rerank.rank_orders(
                ("a", "b", "c"),
                object(),
                top_k=56,
            )

        self.assertEqual(evaluated, 99)
        self.assertEqual(len(retained), 56)
        self.assertEqual(retained[:48], candidates[:48])
        self.assertEqual(base.call_args.kwargs["top_k"], 64)

    def test_facade_small_k_forwards_score_prefix_width_unchanged(self) -> None:
        candidates = _order_candidates()
        with patch.object(
            rerank,
            "_BASE_RANK_ORDERS",
            return_value=(candidates, 17),
        ) as base:
            retained, evaluated = rerank.rank_orders(
                ("a", "b", "c"),
                object(),
                top_k=16,
            )

        self.assertEqual(evaluated, 17)
        self.assertEqual(retained, candidates[:16])
        self.assertEqual(base.call_args.kwargs["top_k"], 16)

    def test_parent_side_table_is_diversified_after_worker_collection(self) -> None:
        candidates = _order_candidates()
        table = rerank.impl._ORDER_CANDIDATES_BY_ROW_ID
        table[123] = candidates

        rerank._diversify_order_side_tables(56)

        self.assertEqual(len(table[123]), 56)
        self.assertEqual(table[123][:48], candidates[:48])

    def test_deep_analyze_widens_and_restores_worker_order_count(self) -> None:
        original_count = rerank.impl._ORDER_CANDIDATE_COUNT
        seen_counts: list[int] = []
        try:
            rerank.impl._ORDER_CANDIDATE_COUNT = 56

            def fake_base(*args: object, **kwargs: object) -> dict[str, float]:
                seen_counts.append(rerank.impl._ORDER_CANDIDATE_COUNT)
                return {"seconds": 0.0, "orders": 0.0, "candidates": 0.0}

            with patch.object(rerank, "_BASE_DEEP_ANALYZE", side_effect=fake_base):
                result = rerank.deep_analyze(
                    [],
                    set(),
                    object(),
                    wordnet_dir=Path("."),
                    backend="serial",
                    workers=0,
                    batch_size=1,
                    order_mode="auto",
                    beam_width=128,
                    exact_max_words=5,
                )

            self.assertEqual(result["candidates"], 0.0)
            self.assertEqual(seen_counts, [64])
            self.assertEqual(rerank.impl._ORDER_CANDIDATE_COUNT, 56)
        finally:
            rerank.impl._ORDER_CANDIDATE_COUNT = original_count

    def test_deep_analyze_restores_worker_order_count_on_failure(self) -> None:
        original_count = rerank.impl._ORDER_CANDIDATE_COUNT
        try:
            rerank.impl._ORDER_CANDIDATE_COUNT = 56
            with patch.object(
                rerank,
                "_BASE_DEEP_ANALYZE",
                side_effect=RuntimeError("boom"),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    rerank.deep_analyze(
                        [],
                        set(),
                        object(),
                        wordnet_dir=Path("."),
                        backend="serial",
                        workers=0,
                        batch_size=1,
                        order_mode="auto",
                        beam_width=128,
                        exact_max_words=5,
                    )
            self.assertEqual(rerank.impl._ORDER_CANDIDATE_COUNT, 56)
        finally:
            rerank.impl._ORDER_CANDIDATE_COUNT = original_count

    def test_user_solver_defaults_to_wider_retention_and_forwards_override(self) -> None:
        default_args = solver.build_parser().parse_args(["abcdef"])
        solver._validate_args(default_args)
        self.assertEqual(default_args.order_candidates, 56)

        override_args = solver.build_parser().parse_args(
            ["abcdef", "--order-candidates", "64"]
        )
        solver._validate_args(override_args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cmd = solver.build_reranker_command(
                override_args,
                root / "candidates.txt",
                root / "reranked.txt",
            )
        self.assertEqual(cmd[cmd.index("--order-candidates") + 1], "64")

    def test_user_solver_rejects_invalid_order_candidate_count(self) -> None:
        args = solver.build_parser().parse_args(
            ["abcdef", "--order-candidates", "0"]
        )
        with self.assertRaisesRegex(SystemExit, "--order-candidates"):
            solver._validate_args(args)


class OptimizedOrderDiversityTests(unittest.TestCase):
    @staticmethod
    def _pool() -> tuple[_ReferenceCandidate, ...]:
        orders = list(itertools.islice(itertools.permutations("abcdef"), 72))
        return tuple(
            _ReferenceCandidate(
                order=tuple(order),
                objective=1.0 - index / 1000.0,
                phrase_kind=("clause" if index % 3 else "noun-phrase"),
            )
            for index, order in enumerate(orders)
        )

    def test_order_similarity_matches_reference_with_repeated_tokens(self) -> None:
        cases = (
            (
                _ReferenceCandidate(("the", "cat", "the", "sat"), 1.0, "clause"),
                _ReferenceCandidate(("the", "sat", "the", "cat"), 0.9, "clause"),
            ),
            (
                _ReferenceCandidate(("a",), 1.0, "fragment"),
                _ReferenceCandidate(("a",), 0.9, "fragment"),
            ),
            (
                _ReferenceCandidate((), 1.0, "fragment"),
                _ReferenceCandidate(("a",), 0.9, "fragment"),
            ),
        )
        for left, right in cases:
            with self.subTest(left=left.order, right=right.order):
                self.assertAlmostEqual(
                    diversity.order_similarity(left, right),
                    _reference_similarity(left, right),
                    places=12,
                )

    def test_optimized_selector_matches_reference(self) -> None:
        pool = self._pool()
        self.assertEqual(
            diversity.select_diverse_orders(pool, 64),
            _reference_select(pool, 64),
        )

    def test_selector_equivalence_with_repeated_tokens_and_ties(self) -> None:
        orders = sorted(set(itertools.permutations(("a", "a", "b", "c"))))
        pool = tuple(
            _ReferenceCandidate(
                order=tuple(order),
                objective=1.0 - (index // 2) * 0.01,
                phrase_kind=("clause" if index % 2 else "fragment"),
            )
            for index, order in enumerate(orders)
        )
        kwargs = {"quality_core": 3, "diversity_strength": 0.27}
        self.assertEqual(
            diversity.select_diverse_orders(pool, 8, **kwargs),
            _reference_select(pool, 8, **kwargs),
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

        self.assertEqual(actual, _reference_select(pool, 64))
        self.assertLessEqual(calls, 1400)

    def test_quality_core_only_path_avoids_similarity_work(self) -> None:
        pool = self._pool()
        with patch.object(
            diversity,
            "_fingerprint_similarity",
            side_effect=AssertionError("similarity should not run"),
        ):
            self.assertEqual(diversity.select_diverse_orders(pool, 48), pool[:48])


class LazyOrderDiversityTests(unittest.TestCase):
    def tearDown(self) -> None:
        rerank._clear_order_side_tables()

    def test_widened_pool_defers_similarity_work_until_alternatives_are_consumed(self) -> None:
        candidates = _order_candidates()
        table = rerank.impl._ORDER_CANDIDATES_BY_ROW_ID
        table[123] = candidates

        with patch.object(
            diversity,
            "_fingerprint_similarity",
            wraps=diversity._fingerprint_similarity,
        ) as similarity:
            rerank._diversify_order_side_tables(56)
            retained = table[123]

            self.assertIsInstance(retained, tuple)
            self.assertFalse(getattr(retained, "is_materialized"))
            self.assertTrue(retained)
            self.assertEqual(retained[0], candidates[0])
            self.assertFalse(getattr(retained, "is_materialized"))
            self.assertEqual(similarity.call_count, 0)

            realized = tuple(retained)

        self.assertTrue(getattr(retained, "is_materialized"))
        self.assertEqual(len(realized), 56)
        self.assertEqual(realized[:48], candidates[:48])
        self.assertGreater(similarity.call_count, 0)

    def test_deferred_selection_matches_eager_reference_exactly(self) -> None:
        candidates = _order_candidates()
        deferred = diversity.select_diverse_orders(candidates, 56)
        expected = diversity._select_diverse_orders_eager(
            candidates,
            56,
            quality_core=diversity.DEFAULT_QUALITY_CORE,
            diversity_strength=diversity.DEFAULT_DIVERSITY_STRENGTH,
        )

        self.assertFalse(getattr(deferred, "is_materialized"))
        self.assertEqual(tuple(deferred), expected)
        self.assertTrue(getattr(deferred, "is_materialized"))

    def test_deferred_tuple_operations_never_expose_rejected_pool_entries(self) -> None:
        candidates = _order_candidates()
        deferred = diversity.select_diverse_orders(candidates, 56)
        expected = diversity._select_diverse_orders_eager(
            candidates,
            56,
            quality_core=diversity.DEFAULT_QUALITY_CORE,
            diversity_strength=diversity.DEFAULT_DIVERSITY_STRENGTH,
        )
        rejected = next(candidate for candidate in candidates if candidate not in expected)

        self.assertEqual(deferred.count(rejected), 0)
        with self.assertRaises(ValueError):
            deferred.index(rejected)
        self.assertEqual(deferred + (), expected)
        self.assertEqual(() + deferred, expected)
        self.assertEqual(deferred * 1, expected)
        self.assertEqual(1 * deferred, expected)
        self.assertEqual(deferred, expected)
        self.assertNotEqual(deferred, list(expected))
        self.assertFalse(deferred != expected)
        self.assertFalse(deferred < expected)
        self.assertTrue(deferred <= expected)
        self.assertFalse(deferred > expected)
        self.assertTrue(deferred >= expected)
        self.assertEqual(hash(deferred), hash(expected))

    def test_deferred_copy_and_pickle_materialize_to_plain_tuple(self) -> None:
        candidates = _order_candidates()
        deferred = diversity.select_diverse_orders(candidates, 56)
        expected = diversity._select_diverse_orders_eager(
            candidates,
            56,
            quality_core=diversity.DEFAULT_QUALITY_CORE,
            diversity_strength=diversity.DEFAULT_DIVERSITY_STRENGTH,
        )

        copied = copy.copy(deferred)
        restored = pickle.loads(pickle.dumps(deferred))

        self.assertIs(type(copied), tuple)
        self.assertIs(type(restored), tuple)
        self.assertEqual(copied, expected)
        self.assertEqual(restored, expected)

    def test_score_prefix_paths_remain_eager_plain_tuples(self) -> None:
        candidates = _order_candidates()
        retained = diversity.select_diverse_orders(candidates, 48)

        self.assertIs(type(retained), tuple)
        self.assertEqual(retained, candidates[:48])


if __name__ == "__main__":
    unittest.main()
