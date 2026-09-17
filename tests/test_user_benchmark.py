from __future__ import annotations

import unittest
from unittest.mock import patch

import benchmark_user_runs as benchmark


class UserBenchmarkTests(unittest.TestCase):
    def test_summary_uses_medians_and_keeps_strategies_separate(self):
        runs = []
        for strategy in ("prefix", "diverse"):
            for phase, times in (
                ("cold", (8.0, 100.0, 10.0)),
                ("warm", (1.0, 3.0, 2.0)),
            ):
                for value in times:
                    runs.append(
                        {
                            "case": "synthetic",
                            "strategy": strategy,
                            "phase": phase,
                            "seconds": value,
                            "peak_tree_rss_bytes": 1024,
                            "expected_bag_generated": strategy == "diverse",
                        }
                    )
        summary = benchmark.summarize(runs)
        self.assertEqual(len(summary), 2)
        for row in summary:
            self.assertEqual(row["cold_median_seconds"], 10.0)
            self.assertEqual(row["warm_median_seconds"], 2.0)
            self.assertEqual(row["warm_speedup"], 5.0)
            self.assertEqual(row["samples"], 3)
            self.assertEqual(
                row["expected_bag_recall"], float(row["strategy"] == "diverse")
            )

    def test_rejects_nonfinite_timeout_without_starting_work(self):
        with (
            patch.object(benchmark, "measure") as measure,
            patch("anagram_user_lexicon.ensure_user_lexicon") as lexicon,
            patch("anagram_rerank.ensure_wordnet") as wordnet,
            patch("anagram_generate.ensure_ngram_data") as ngrams,
        ):
            with self.assertRaises(SystemExit):
                benchmark.main(["--timeout", "nan"])
            for operation in (measure, lexicon, wordnet, ngrams):
                operation.assert_not_called()
