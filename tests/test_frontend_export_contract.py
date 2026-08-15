from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anagram_generate as generator
import anagram_rerank_core as reranker


class FrontendExportContractTests(unittest.TestCase):
    def test_component_rich_generator_record_matches_reranker_parser(self) -> None:
        record = generator.Record(
            words=("anagram", "test"),
            word_count=2,
            matched_hints=(),
            lex_raw=1.0,
            avg_zipf=4.0,
            min_zipf=3.5,
            junk_penalty=0.0,
            family=("anagram", "test"),
            family_best_lex=1.0,
            pair_raw=0.0,
            pair_coverage=0.0,
            lex_pct=0.5,
            family_pct=0.5,
            pair_pct=0.5,
            pre_score=50.0,
        )
        line = generator.format_pre_record(1, record, show_components=True)
        self.assertIsNotNone(reranker.GENERATOR_LINE_RE.match(line))
        self.assertIsNone(
            reranker.GENERATOR_LINE_RE.match(
                generator.format_pre_record(1, record, show_components=False)
            )
        )

    def test_full_component_export_is_parseable(self) -> None:
        record = generator.Record(
            words=("anagram", "test"),
            word_count=2,
            matched_hints=(),
            lex_raw=1.0,
            avg_zipf=4.0,
            min_zipf=3.5,
            junk_penalty=0.0,
            family=("anagram", "test"),
            family_best_lex=1.0,
            pair_raw=0.0,
            pair_coverage=0.0,
            lex_pct=0.5,
            family_pct=0.5,
            pair_pct=0.5,
            pre_score=50.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "candidates.txt"
            generator.write_full_export(export, [record], show_components=True)
            rows = reranker.parse_candidates(export)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].words, ("anagram", "test"))
