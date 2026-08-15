from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import anagram_benchmark as benchmark


class _FakeReranker:
    @staticmethod
    def rank_orders(words, _lex, **_kwargs):
        # Grammar winner first; intended phrase second.
        return (
            (
                SimpleNamespace(
                    order=("power", "is", "knowledge"),
                    objective=0.90,
                ),
                SimpleNamespace(
                    order=("knowledge", "is", "power"),
                    objective=0.88,
                ),
            ),
            6,
        )


class _NoTargetReranker:
    @staticmethod
    def rank_orders(words, _lex, **_kwargs):
        return (
            (
                SimpleNamespace(order=("power", "knowledge", "is"), objective=0.90),
                SimpleNamespace(order=("is", "power", "knowledge"), objective=0.80),
            ),
            6,
        )


class _TiedReranker:
    @staticmethod
    def rank_orders(words, _lex, **_kwargs):
        # Incoming rank_orders() tie order is authoritative. The intended phrase
        # deliberately comes second and has stronger phrase evidence.
        return (
            (
                SimpleNamespace(order=("power", "is", "knowledge"), objective=0.90),
                SimpleNamespace(order=("knowledge", "is", "power"), objective=0.90),
            ),
            6,
        )


class _PhraseIndex:
    def score(self, order):
        if tuple(order) == ("knowledge", "is", "power"):
            return 1.0, {}
        return 0.0, {}


class _NoEvidencePhraseIndex:
    def score(self, order):
        return 0.0, {}


class PhraseBenchmarkTests(unittest.TestCase):
    def test_phrase_evidence_can_flip_retained_order(self):
        result = benchmark.run_phrase_order_case(
            _FakeReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _PhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertTrue(result.target_retained)
        self.assertEqual(result.grammar_rank, 2)
        self.assertEqual(result.retained_rank, 1)
        self.assertEqual(result.grammar_best_order, "power is knowledge")
        self.assertEqual(result.best_order, "knowledge is power")

    def test_zero_phrase_evidence_preserves_retained_grammar_winner(self):
        result = benchmark.run_phrase_order_case(
            _FakeReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _NoEvidencePhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertEqual(result.grammar_rank, 2)
        self.assertEqual(result.retained_rank, 2)
        self.assertEqual(result.best_order, result.grammar_best_order)
        self.assertFalse(result.exact_best)

    def test_zero_bonus_preserves_incoming_order_for_objective_ties(self):
        result = benchmark.run_phrase_order_case(
            _TiedReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _PhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=0.0,
        )
        self.assertEqual(result.grammar_best_order, "power is knowledge")
        self.assertEqual(result.best_order, "power is knowledge")
        self.assertEqual(result.grammar_rank, 2)
        self.assertEqual(result.retained_rank, 2)
        self.assertFalse(result.exact_best)

    def test_no_acceptable_order_retained_counts_as_miss(self):
        result = benchmark.run_phrase_order_case(
            _NoTargetReranker,
            object(),
            {"id": "knowledge_power", "answer": "knowledge is power"},
            _PhraseIndex(),
            order_candidates=16,
            phrase_bonus_max=5.0,
        )
        self.assertFalse(result.target_retained)
        self.assertIsNone(result.grammar_rank)
        self.assertIsNone(result.retained_rank)
        metrics = benchmark.compute_phrase_order_metrics([result])
        self.assertEqual(metrics["retained_rate"], 0.0)
        self.assertEqual(metrics["recall1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_phrase_metrics_cover_retained_higher_and_dropped_targets(self):
        rows = [
            benchmark.PhraseOrderResult(
                "id1", "a b", "test", 1, 20, "a b", True, True, 1.0, 1, "a b"
            ),
            benchmark.PhraseOrderResult(
                "id2", "c d", "test", 5, 20, "d c", False, True, 0.0, 7, "d c"
            ),
            benchmark.PhraseOrderResult(
                "id3", "e f", "test", 20, 20, "f e", False, True, 0.0, 12, "f e"
            ),
            benchmark.PhraseOrderResult(
                "id4", "g h", "test", None, 20, "h g", False, False, 0.0, None, "h g"
            ),
        ]
        metrics = benchmark.compute_phrase_order_metrics(rows)
        self.assertEqual(metrics["retained"], 3)
        self.assertAlmostEqual(metrics["retained_rate"], 0.75)
        self.assertAlmostEqual(metrics["recall1"], 0.25)
        self.assertAlmostEqual(metrics["recall10"], 0.50)
        self.assertAlmostEqual(metrics["recall50"], 0.75)
        self.assertAlmostEqual(metrics["mrr"], (1.0 + 0.2 + 0.05) / 4.0)

        grammar = benchmark.compute_retained_grammar_metrics(rows)
        self.assertAlmostEqual(grammar["recall1"], 0.25)
        self.assertAlmostEqual(grammar["recall10"], 0.50)
        self.assertAlmostEqual(grammar["recall50"], 0.75)

    def test_empty_phrase_summary_does_not_crash(self):
        benchmark.print_phrase_order_summary(
            [], order_candidates=16, phrase_db=Path("empty.db")
        )

    def test_full_reranker_command_forwards_phrase_options(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power", "deep_per_group": 123},
            reranker=Path("anagram_rerank.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=Path("wikimedia_phrases.db"),
            phrase_bonus_max=4.5,
            order_candidates=24,
        )
        self.assertIn("--phrase-db", cmd)
        self.assertEqual(cmd[cmd.index("--phrase-db") + 1], "wikimedia_phrases.db")
        self.assertEqual(cmd[cmd.index("--phrase-bonus-max") + 1], "4.5")
        self.assertEqual(cmd[cmd.index("--order-candidates") + 1], "24")
        self.assertEqual(cmd[cmd.index("--deep-per-group") + 1], "123")

    def test_full_reranker_command_preserves_baseline_when_phrase_db_disabled(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power"},
            reranker=Path("anagram_rerank_core.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=None,
            phrase_bonus_max=5.0,
            order_candidates=16,
        )
        self.assertNotIn("--phrase-db", cmd)
        self.assertNotIn("--phrase-bonus-max", cmd)
        self.assertNotIn("--order-candidates", cmd)
        # Existing positive-bigram rescoring remains part of the baseline.
        self.assertIn("--phrase-rescore-top", cmd)


if __name__ == "__main__":
    unittest.main()
