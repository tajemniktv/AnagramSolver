from __future__ import annotations

import sqlite3
import unittest

import anagram_performance as perf
import anagram_rerank_core as core


def _phrase_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE ngrams (text TEXT PRIMARY KEY, n INTEGER NOT NULL, count INTEGER NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO ngrams(text, n, count) VALUES (?, ?, ?)",
        (
            ("actions speak louder than words", 5, 20),
            ("actions speak", 2, 100),
            ("speak louder", 2, 80),
            ("louder than", 2, 75),
            ("than words", 2, 90),
            ("actions speak louder", 3, 30),
            ("speak louder than", 3, 25),
            ("louder than words", 3, 35),
        ),
    )
    return connection


class PerformanceHotPathTests(unittest.TestCase):
    def test_fast_norm_token_preserves_core_behavior(self) -> None:
        for text in ("hello", "HELLO", "café", "don't", "", "two words"):
            with self.subTest(text=text):
                self.assertEqual(perf.fast_norm_token(text), perf._ORIGINAL_NORM_TOKEN(text))

    def test_cached_function_class_matches_original(self) -> None:
        for word in ("the", "than", "we", "is", "never", "dog", "unknownword"):
            with self.subTest(word=word):
                self.assertEqual(
                    perf.cached_function_class(word),
                    perf._ORIGINAL_FUNCTION_CLASS(word),
                )

    def test_fast_wordnet_frames_are_cached_per_surface_form(self) -> None:
        lex = perf.FastWordNetLexicon(
            nouns=set(),
            verbs={"run"},
            adjs=set(),
            advs=set(),
            noun_exc={},
            verb_exc={},
            verb_frames={"run": frozenset({1, 8})},
        )
        self.assertEqual(lex.frames_for("run"), frozenset({1, 8}))
        self.assertEqual(lex.frames_for("RUN"), frozenset({1, 8}))
        self.assertEqual(lex._frames_cache, {"run": frozenset({1, 8})})

    def test_fast_phrase_index_matches_original_and_reuses_counts(self) -> None:
        baseline_connection = _phrase_connection()
        fast_connection = _phrase_connection()
        try:
            baseline = perf._ORIGINAL_PHRASE_INDEX(baseline_connection, 5)
            fast = perf.FastPhraseIndex(fast_connection, 5)
            words = ("actions", "speak", "louder", "than", "words")

            expected = baseline.score(words)
            select_statements: list[str] = []
            fast_connection.set_trace_callback(
                lambda statement: (
                    select_statements.append(statement)
                    if statement.startswith("SELECT text, count")
                    else None
                )
            )
            actual = fast.score(words)
            queries_after_first = len(select_statements)
            repeated = fast.score(words)

            self.assertEqual(actual, expected)
            self.assertEqual(repeated, expected)
            self.assertGreater(queries_after_first, 0)
            self.assertEqual(len(select_statements), queries_after_first)
            self.assertIn("actions speak louder than words", fast._count_cache)
            self.assertIn("actions speak louder than", fast._count_cache)
        finally:
            baseline_connection.close()
            fast_connection.close()

    def test_install_is_idempotent(self) -> None:
        perf.install_performance_hooks()
        first_class = core.WordNetLexicon
        first_phrase_index = core.PhraseIndex
        first_norm = core.norm_token
        first_function_class = core.function_class
        perf.install_performance_hooks()
        self.assertIs(core.WordNetLexicon, first_class)
        self.assertIs(core.PhraseIndex, first_phrase_index)
        self.assertIs(core.norm_token, first_norm)
        self.assertIs(core.function_class, first_function_class)


if __name__ == "__main__":
    unittest.main()
