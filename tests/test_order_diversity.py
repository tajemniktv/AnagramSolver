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

    def test_raw_pool_widens_only_after_quality_core(self) -> None:
        self.assertEqual(diversity.raw_pool_size(8), 8)
        self.assertEqual(diversity.raw_pool_size(16), 16)
        self.assertEqual(diversity.raw_pool_size(24), 48)
        self.assertEqual(diversity.raw_pool_size(32), 64)
        self.assertEqual(diversity.raw_pool_size(64), 128)

    def test_small_k_preserves_historical_score_prefix_exactly(self) -> None:
        candidates = tuple(
            _Candidate((str(i),), 1.0 - i * 0.01)
            for i in range(24)
        )
        retained = diversity.select_diverse_orders(candidates, 16)
        self.assertEqual(retained, candidates[:16])

    def test_wider_retention_never_drops_old_top16(self) -> None:
        candidates = tuple(
            _Candidate((str(i), "x", "y"), 1.0 - i * 0.01)
            for i in range(64)
        )
        retained = diversity.select_diverse_orders(candidates, 32)
        self.assertEqual(len(retained), 32)
        for candidate in candidates[:16]:
            self.assertIn(candidate, retained)
        self.assertEqual(retained[0], candidates[0])

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

    def test_facade_requests_top64_raw_pool_for_default_top32(self) -> None:
        candidates = tuple(
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
            for i in range(64)
        )

        with patch.object(rerank, "_BASE_RANK_ORDERS", return_value=(candidates, 99)) as base:
            retained, evaluated = rerank.rank_orders(("a", "b", "c"), object(), top_k=32)

        self.assertEqual(evaluated, 99)
        self.assertEqual(len(retained), 32)
        self.assertEqual(retained[:16], candidates[:16])
        self.assertEqual(base.call_args.kwargs["top_k"], 64)

    def test_parent_side_table_is_diversified_after_worker_collection(self) -> None:
        candidates = tuple(
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
            for i in range(64)
        )
        table = rerank.impl._ORDER_CANDIDATES_BY_ROW_ID
        table[123] = candidates

        rerank._diversify_order_side_tables(32)

        self.assertEqual(len(table[123]), 32)
        self.assertEqual(table[123][:16], candidates[:16])

    def test_user_solver_defaults_to_wider_retention_and_forwards_override(self) -> None:
        default_args = solver.build_parser().parse_args(["abcdef"])
        solver._validate_args(default_args)
        self.assertEqual(default_args.order_candidates, 32)

        override_args = solver.build_parser().parse_args(
            ["abcdef", "--order-candidates", "48"]
        )
        solver._validate_args(override_args)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cmd = solver.build_reranker_command(
                override_args,
                root / "candidates.txt",
                root / "reranked.txt",
            )
        self.assertEqual(cmd[cmd.index("--order-candidates") + 1], "48")

    def test_user_solver_rejects_invalid_order_candidate_count(self) -> None:
        args = solver.build_parser().parse_args(
            ["abcdef", "--order-candidates", "0"]
        )
        with self.assertRaisesRegex(SystemExit, "--order-candidates"):
            solver._validate_args(args)


if __name__ == "__main__":
    unittest.main()
