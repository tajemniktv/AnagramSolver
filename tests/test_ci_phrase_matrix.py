from __future__ import annotations

import unittest
from pathlib import Path

import ci_phrase_matrix as matrix


ORDER_BASELINE = """
Exact-order metrics (<=6 words):
  cases       45
  Recall@1    0.356
  Recall@10   0.800
  Recall@50   0.956
  MRR         0.482
  median rank 3
By category (exact-order cases):
Suite wall time: 0.99s
"""

ORDER_PHRASE = ORDER_BASELINE + """
=== PHRASE-AWARE FINAL ORDER A/B ===
Retained grammar metrics (same top-K):
  cases           45
  target retained 37/45 (0.822)
  Recall@1        0.267
  Recall@10       0.800
  Recall@50       0.822
  MRR             0.427
Phrase-aware retained-order metrics:
  cases           45
  target retained 37/45 (0.822)
  Recall@1        0.400
  Recall@10       0.822
  Recall@50       0.822
  MRR             0.537
A/B delta on identical retained candidates:
  Recall@1   +0.133
  Recall@10  +0.022
  Recall@50  +0.000
  MRR        +0.109
Suite wall time: 1.53s
"""

FULL_REPORT = """
Correct-word-bag ranking:
  BagRecall@1     0.750
  BagRecall@10    0.875
  BagRecall@50    0.875
  BagRecall@100   0.875
  BagMRR          0.781
End-to-end exact phrase surfaced:
  ExactRecall@1   0.625
  ExactRecall@10  0.625
  ExactRecall@50  0.625
  ExactRecall@100 0.625
  ExactMRR        0.625
NOTE: BAG rank answers a different question.
"""


class CorpusMatrixParserTests(unittest.TestCase):
    def test_current_phrase_report_parses_required_metrics(self):
        order = matrix.order_metrics(ORDER_PHRASE)
        full = matrix.full_metrics(FULL_REPORT)
        scenario = matrix.Scenario("Wiktionary", "wiktionary", Path("phrases.db"))
        matrix.validate_metrics(scenario, order, full)

        self.assertAlmostEqual(order["grammar_r1"], 0.356)
        self.assertAlmostEqual(order["retained_r1"], 0.267)
        self.assertAlmostEqual(order["phrase_r1"], 0.400)
        self.assertAlmostEqual(order["delta_r1"], 0.133)
        self.assertAlmostEqual(full["bag_r10"], 0.875)
        self.assertAlmostEqual(full["exact_r10"], 0.625)

    def test_baseline_allows_intentionally_absent_phrase_metrics(self):
        order = matrix.order_metrics(ORDER_BASELINE)
        full = matrix.full_metrics(FULL_REPORT)
        scenario = matrix.Scenario("Baseline", "baseline", None)
        matrix.validate_metrics(scenario, order, full)

        self.assertIsNone(order["phrase_r1"])
        self.assertIsNone(order["delta_r1"])

    def test_missing_phrase_heading_fails_validation(self):
        broken = ORDER_PHRASE.replace(
            "Phrase-aware retained-order metrics:",
            "Phrase-aware metrics:",
        )
        order = matrix.order_metrics(broken)
        full = matrix.full_metrics(FULL_REPORT)
        scenario = matrix.Scenario("Wiktionary", "wiktionary", Path("phrases.db"))

        with self.assertRaisesRegex(RuntimeError, "phrase_r1"):
            matrix.validate_metrics(scenario, order, full)

    def test_missing_full_metric_fails_validation(self):
        broken = FULL_REPORT.replace("  ExactMRR        0.625\n", "")
        order = matrix.order_metrics(ORDER_BASELINE)
        full = matrix.full_metrics(broken)
        scenario = matrix.Scenario("Baseline", "baseline", None)

        with self.assertRaisesRegex(RuntimeError, "exact_mrr"):
            matrix.validate_metrics(scenario, order, full)


if __name__ == "__main__":
    unittest.main()
