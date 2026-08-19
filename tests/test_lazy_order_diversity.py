from __future__ import annotations

import unittest
from unittest.mock import patch

import anagram_order_diversity as diversity
import anagram_rerank as rerank


class LazyOrderDiversityTests(unittest.TestCase):
    def tearDown(self) -> None:
        rerank._clear_order_side_tables()

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

    def test_widened_pool_defers_similarity_work_until_alternatives_are_consumed(self) -> None:
        candidates = self._order_candidates()
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
        candidates = self._order_candidates()
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

    def test_score_prefix_paths_remain_eager_plain_tuples(self) -> None:
        candidates = self._order_candidates()
        retained = diversity.select_diverse_orders(candidates, 48)

        self.assertIs(type(retained), tuple)
        self.assertEqual(retained, candidates[:48])


if __name__ == "__main__":
    unittest.main()
