from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
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

    def _order_candidates(self, count: int = 64) -> tuple[rerank.OrderCandidate, ...]:
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

    def test_facade_requests_top64_raw_pool_for_default_top56(self) -> None:
        candidates = self._order_candidates()
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
        candidates = self._order_candidates()
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
        candidates = self._order_candidates()
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


if __name__ == "__main__":
    unittest.main()
