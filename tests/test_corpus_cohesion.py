from __future__ import annotations

import sqlite3
import unittest

import anagram_performance as perf
from anagram_corpus_cohesion import score_corpus_cohesion


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


class CorpusCohesionTests(unittest.TestCase):
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
