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
                    grammar_norm=0.90,
                    structure_norm=0.90,
                    valency_norm=1.0,
                ),
                SimpleNamespace(
                    order=("knowledge", "is", "power"),
                    grammar_norm=0.88,
                    structure_norm=0.89,
                    valency_norm=1.0,
                ),
            ),
            6,
        )


class _PhraseIndex:
    def score(self, order):
        if tuple(order) == ("knowledge", "is", "power"):
            return 1.0, {}
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
        self.assertTrue(result.exact_best)
        self.assertEqual(result.retained_rank, 1)
        self.assertEqual(result.best_order, "knowledge is power")

    def test_missing_retained_target_counts_as_miss(self):
        result = benchmark.PhraseOrderResult(
            "x", "a b", "test", None, 2, "b a", False, False, 0.0
        )
        metrics = benchmark.compute_phrase_order_metrics([result])
        self.assertEqual(metrics["retained_rate"], 0.0)
        self.assertEqual(metrics["recall1"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

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

    def test_full_reranker_command_omits_phrase_db_when_disabled(self):
        cmd = benchmark.make_reranker_command(
            {"answer": "knowledge is power"},
            reranker=Path("anagram_rerank.py"),
            export=Path("candidates.txt"),
            output=Path("reranked.txt"),
            workers=4,
            phrase_db=None,
            phrase_bonus_max=5.0,
            order_candidates=16,
        )
        self.assertNotIn("--phrase-db", cmd)


if __name__ == "__main__":
    unittest.main()
