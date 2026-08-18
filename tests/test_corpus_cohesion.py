from __future__ import annotations

import sqlite3
import unittest

import anagram_performance as perf
from anagram_corpus_cohesion import (
    CohesionResult,
    blend_phrase_cohesion,
    score_corpus_cohesion,
)


def _lookup(mapping: dict[str, int]):
    def counts(phrases: tuple[str, ...]) -> dict[str, int]:
        return {phrase: mapping[phrase] for phrase in phrases if phrase in mapping}

    return counts


def _connection(rows: tuple[tuple[str, int, int], ...]) -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE ngrams (text TEXT PRIMARY KEY, n INTEGER NOT NULL, count INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO ngrams(text, n, count) VALUES (?, ?, ?)",
        rows,
    )
    return connection


def _cohesion(score: float) -> CohesionResult:
    return CohesionResult(score, 0.0, 0.0, 0, 0.0, 0.0, ())


class CorpusCohesionTests(unittest.TestCase):
    def test_blend_is_positive_only_and_bounded(self) -> None:
        self.assertEqual(blend_phrase_cohesion(0.7, _cohesion(0.1)), 0.7)
        self.assertGreater(blend_phrase_cohesion(0.4, _cohesion(0.9)), 0.4)
        self.assertLessEqual(blend_phrase_cohesion(0.4, _cohesion(1.0)), 1.0)
        for score in (0.0, 0.5, 1.0):
            with self.subTest(score=score):
                self.assertEqual(blend_phrase_cohesion(1.0, _cohesion(score)), 1.0)

    def test_whole_phrase_beats_fragmented_explanation(self) -> None:
        words = ("better", "late", "than", "never")
        whole = score_corpus_cohesion(
            words,
            counts=_lookup({"better late than never": 80}),
            max_n=4,
        )
        split = score_corpus_cohesion(
            words,
            counts=_lookup({"better late": 800, "than never": 800}),
            max_n=4,
        )

        self.assertEqual(tuple(span.text for span in whole.spans), ("better late than never",))
        self.assertEqual(whole.coverage, 1.0)
        self.assertEqual(whole.longest_fraction, 1.0)
        self.assertEqual(whole.segments, 1)
        self.assertEqual(split.coverage, 1.0)
        self.assertEqual(split.segments, 2)
        self.assertGreater(split.splice_penalty, 0.0)
        self.assertGreater(whole.score, split.score)

    def test_two_strong_chunks_can_cover_unseen_full_phrase(self) -> None:
        result = score_corpus_cohesion(
            ("better", "late", "than", "never"),
            counts=_lookup({"better late": 1200, "than never": 900}),
            max_n=4,
        )

        self.assertEqual(
            tuple(span.text for span in result.spans),
            ("better late", "than never"),
        )
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.longest_fraction, 0.5)
        self.assertEqual(result.segments, 2)
        self.assertGreater(result.score, 0.6)

    def test_missing_corpus_evidence_is_neutral(self) -> None:
        result = score_corpus_cohesion(
            ("novel", "unseen", "phrase"),
            counts=_lookup({}),
            max_n=3,
        )

        self.assertEqual(result.score, 0.0)
        self.assertEqual(result.coverage, 0.0)
        self.assertEqual(result.segments, 0)
        self.assertEqual(result.spans, ())

    def test_short_orders_and_max_n_limits_are_neutral(self) -> None:
        single = score_corpus_cohesion(
            ("lonely",),
            counts=_lookup({"lonely": 50}),
            max_n=3,
        )
        limited = score_corpus_cohesion(
            ("short", "phrase", "input"),
            counts=_lookup({"short phrase": 50, "phrase input": 50}),
            max_n=1,
        )
        for result in (single, limited):
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.coverage, 0.0)
            self.assertEqual(result.segments, 0)
            self.assertEqual(result.spans, ())

    def test_nonpositive_counts_and_empty_tokens_are_neutral(self) -> None:
        nonpositive = score_corpus_cohesion(
            ("neutral", "cohesion"),
            counts=_lookup({"neutral cohesion": 0}),
            max_n=2,
        )
        negative = score_corpus_cohesion(
            ("negative", "cohesion"),
            counts=_lookup({"negative cohesion": -5}),
            max_n=2,
        )
        malformed = score_corpus_cohesion(
            ("foo", "", "bar"),
            counts=_lookup({"foo bar": 1000}),
            max_n=3,
        )
        for result in (nonpositive, negative, malformed):
            self.assertEqual(result.score, 0.0)
            self.assertEqual(result.spans, ())

    def test_tie_breaking_is_deterministic(self) -> None:
        words = ("a", "b", "c")
        first = score_corpus_cohesion(
            words,
            counts=_lookup({"a b": 10, "b c": 10}),
            max_n=2,
        )
        second = score_corpus_cohesion(
            words,
            counts=_lookup({"b c": 10, "a b": 10}),
            max_n=2,
        )
        self.assertEqual(first, second)
        self.assertEqual(tuple(span.text for span in first.spans), ("b c",))

    def test_fast_phrase_index_exposes_and_blends_cohesion(self) -> None:
        baseline_connection = _connection(
            (
                ("better late", 2, 1200),
                ("than never", 2, 900),
            )
        )
        fast_connection = _connection(
            (
                ("better late", 2, 1200),
                ("than never", 2, 900),
            )
        )
        try:
            baseline = perf._ORIGINAL_PHRASE_INDEX(baseline_connection, 4)
            fast = perf.FastPhraseIndex(fast_connection, 4)
            words = ("better", "late", "than", "never")

            baseline_score, _ = baseline.score(words)
            score, details = fast.score(words)

            self.assertGreater(score, baseline_score)
            self.assertGreater(details["cohesion"], 0.6)
            self.assertEqual(details["cohesion_coverage"], 1.0)
            self.assertEqual(details["cohesion_segments"], 2.0)
            self.assertGreater(details["cohesion_splice_penalty"], 0.0)
        finally:
            baseline_connection.close()
            fast_connection.close()


if __name__ == "__main__":
    unittest.main()
