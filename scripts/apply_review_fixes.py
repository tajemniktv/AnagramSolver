#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace(path: str, old: str, new: str) -> None:
    p = ROOT / path
    text = p.read_text(encoding="utf-8")
    if old not in text:
        raise SystemExit(f"Expected text not found in {path}: {old!r}")
    p.write_text(text.replace(old, new), encoding="utf-8")


replace(
    "anagram_rerank_topk_impl.py",
    "    row.final = score_final(row)\n",
    "    row.final = core.score_final(row)\n",
)
replace(
    "anagram_rerank.py",
    'PREPARED_CACHE_SCHEMA = "topk-prepared-json-gzip-1"',
    'PREPARED_CACHE_SCHEMA = "topk-prepared-json-gzip-2"',
)
replace(
    "anagram_benchmark.py",
    'FINAL_RE = re.compile(r"FINAL rank:\\s+([\\d,]+)\\s+/\\s+([\\d,]+)")\nPRE_RE = re.compile(r"PRE rank:\\s+([\\d,]+)\\s+/\\s+([\\d,]+)")',
    'FINAL_RE = re.compile(r"^\\s*FINAL rank:\\s+([\\d,]+)\\s+/\\s+([\\d,]+)", re.MULTILINE)\nPRE_RE = re.compile(r"^\\s*PRE rank:\\s+([\\d,]+)\\s+/\\s+([\\d,]+)", re.MULTILINE)',
)

core = ROOT / "anagram_rerank_core.py"
text = core.read_text(encoding="utf-8")
text = text.replace("current reranker", "linguistic reranker")
text = text.replace("Current reranker", "Linguistic reranker")
core.write_text(text, encoding="utf-8")

(ROOT / "tests" / "test_review_fixes.py").write_text(
    '''from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_benchmark as benchmark
import anagram_rerank as rerank
import anagram_rerank_core as core


class ReviewFixRegressionTests(unittest.TestCase):
    @staticmethod
    def _row():
        row = rerank.Row(
            words=("a", "b", "c"),
            word_count=3,
            old_rank=1,
            old_pre=1.0,
            lex=0.8,
            fam=0.8,
            old_pair=0.5,
            hint=0.0,
            zavg=5.0,
            zmin=4.0,
            old_pcov=0.0,
            hints=(),
        )
        row.wn_coverage = 1.0
        row.pre_score = 70.0
        return row

    def test_deep_result_uses_defined_core_final_scorer(self) -> None:
        row = self._row()
        result = rerank.impl.DeepResult(
            row_index=0,
            grammar_raw=2.0,
            best_order=("a", "b", "c"),
            structure_norm=0.9,
            valency_norm=1.0,
            syntax_coverage=1.0,
            phrase_kind="clause",
            orders_evaluated=6,
            order_candidates=(),
        )
        with patch.object(core, "score_final", return_value=42.5) as scorer:
            rerank.impl._apply_deep_result([row], result)
        scorer.assert_called_once_with(row)
        self.assertEqual(row.final, 42.5)

    def test_pre_rank_regex_ignores_generator_pre_rank(self) -> None:
        output = """BENCHMARK: example\n  generator PRE rank: 1,749 / 5,696\n  PRE rank: 29 / 5,696  (87.42)\n  FINAL rank: 3 / 5,696 (90.27)\n"""
        pre = benchmark.PRE_RE.search(output)
        final = benchmark.FINAL_RE.search(output)
        self.assertIsNotNone(pre)
        self.assertIsNotNone(final)
        self.assertEqual(pre.groups(), ("29", "5,696"))
        self.assertEqual(final.groups(), ("3", "5,696"))

    def test_old_prepared_cache_schema_is_rejected_without_row_parsing(self) -> None:
        self.assertEqual(rerank.PREPARED_CACHE_SCHEMA, "topk-prepared-json-gzip-2")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old-cache.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump({"schema": "topk-prepared-json-gzip-1", "rows": []}, handle)
            self.assertIsNone(rerank.load_prepared_cache(path))


if __name__ == "__main__":
    unittest.main()
''',
    encoding="utf-8",
)
