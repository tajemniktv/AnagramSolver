from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anagram_generate as generator
import anagram_rerank_core as reranker


class FrontendExportContractTests(unittest.TestCase):
    def _record(self) -> generator.Record:
        return generator.Record(
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
            pair_coverage=0.25,
            lex_pct=0.5,
            family_pct=0.6,
            pair_pct=0.7,
            hint_info=0.2,
            pre_score=50.0,
        )

    def test_component_rich_generator_record_matches_reranker_parser(self) -> None:
        record = self._record()
        line = generator.format_pre_record(1, record, show_components=True)
        match = reranker.GENERATOR_LINE_RE.match(line)
        self.assertIsNotNone(match)
        assert match is not None
        groups = match.groupdict()
        self.assertEqual(float(groups["pre"]), record.pre_score)
        self.assertEqual(float(groups["lex"]), record.lex_pct)
        self.assertEqual(float(groups["fam"]), record.family_pct)
        self.assertEqual(float(groups["pair"]), record.pair_pct)
        self.assertEqual(float(groups["hint"]), record.hint_info)
        self.assertEqual(float(groups["zavg"]), record.avg_zipf)
        self.assertEqual(float(groups["zmin"]), record.min_zipf)
        self.assertEqual(float(groups["pcov"]), record.pair_coverage)
        self.assertEqual(groups["phrase"], "anagram test")
        self.assertEqual(groups["hints"], "-")

        self.assertIsNone(
            reranker.GENERATOR_LINE_RE.match(
                generator.format_pre_record(1, record, show_components=False)
            )
        )

    def test_full_component_export_is_parseable(self) -> None:
        record = self._record()
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "candidates.txt"
            generator.write_full_export(export, [record], show_components=True)
            rows = reranker.parse_candidates(export)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].words, ("anagram", "test"))

    def test_compact_export_produces_no_reranker_rows(self) -> None:
        record = self._record()
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "candidates.txt"
            generator.write_full_export(export, [record], show_components=False)
            rows = reranker.parse_candidates(export)
        self.assertEqual(rows, [])
