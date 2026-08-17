from __future__ import annotations

import importlib
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
            ("zero gram", 2, 0),
            ("negative gram", 2, -3),
        ),
    )
    return connection


class PerformanceHotPathTests(unittest.TestCase):
    def tearDown(self) -> None:
        perf.clear_performance_caches()

    def test_fast_norm_token_preserves_core_behavior(self) -> None:
        for text in (
            "hello",
            "HELLO",
            "café",
            "don't",
            "",
            "two words",
            "hello123",
            "snake_case",
            "!!!",
            "a" * 10_000,
        ):
            with self.subTest(text=text[:80]):
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
            verbs={"run", "walk"},
            adjs=set(),
            advs=set(),
            noun_exc={},
            verb_exc={},
            verb_frames={"run": frozenset({1, 8})},
        )
        self.assertEqual(lex.frames_for("run"), frozenset({1, 8}))
        self.assertEqual(lex.frames_for("RUN"), frozenset({1, 8}))
        self.assertEqual(lex.frames_for("walk"), frozenset())
        self.assertEqual(lex.frames_for("WALK"), frozenset())
        self.assertEqual(
            lex._frames_cache,
            {
                "run": frozenset({1, 8}),
                "walk": frozenset(),
            },
        )

    def test_fast_wordnet_frame_cache_is_bounded_lru(self) -> None:
        lex = perf.FastWordNetLexicon(
            nouns=set(),
            verbs={"run", "walk", "jump"},
            adjs=set(),
            advs=set(),
            noun_exc={},
            verb_exc={},
            verb_frames={"run": frozenset({1})},
            _frames_cache_limit=2,
        )
        lex.frames_for("run")
        lex.frames_for("walk")
        lex.frames_for("run")
        lex.frames_for("jump")
        self.assertEqual(tuple(lex._frames_cache), ("run", "jump"))
        self.assertLessEqual(len(lex._frames_cache), lex._frames_cache_limit)

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

    def test_fast_phrase_index_preserves_nonpositive_rows_and_misses(self) -> None:
        baseline_connection = _phrase_connection()
        fast_connection = _phrase_connection()
        try:
            baseline = perf._ORIGINAL_PHRASE_INDEX(baseline_connection, 5)
            fast = perf.FastPhraseIndex(fast_connection, 5)
            phrases = ("zero gram", "negative gram", "missing gram")
            expected = baseline.counts(phrases)
            actual = fast.counts(phrases)

            self.assertEqual(actual, expected)
            self.assertEqual(actual, {"zero gram": 0, "negative gram": -3})
            self.assertIsNone(fast._count_cache["missing gram"])
            self.assertEqual(fast.counts(phrases), expected)
        finally:
            baseline_connection.close()
            fast_connection.close()

    def test_fast_phrase_index_count_cache_is_bounded(self) -> None:
        connection = _phrase_connection()
        try:
            fast = perf.FastPhraseIndex(connection, 5, _count_cache_limit=3)
            self.assertEqual(
                fast.counts(("missing one", "missing two", "missing three", "missing four")),
                {},
            )
            self.assertEqual(
                tuple(fast._count_cache),
                ("missing two", "missing three", "missing four"),
            )

            self.assertEqual(fast.counts(("missing two",)), {})
            fast.counts(("missing five",))
            self.assertEqual(
                tuple(fast._count_cache),
                ("missing four", "missing two", "missing five"),
            )
            self.assertLessEqual(len(fast._count_cache), fast._count_cache_limit)
        finally:
            connection.close()

    def test_performance_hooks_install_nest_and_restore(self) -> None:
        before = (
            core.norm_token,
            core.function_class,
            core.WordNetLexicon,
            core.PhraseIndex,
        )

        with perf.performance_hooks():
            self.assertIs(core.norm_token, perf.fast_norm_token)
            self.assertIs(core.function_class, perf.cached_function_class)
            self.assertIs(core.WordNetLexicon, perf.FastWordNetLexicon)
            self.assertIs(core.PhraseIndex, perf.FastPhraseIndex)
            with perf.performance_hooks():
                self.assertIs(core.norm_token, perf.fast_norm_token)
                self.assertIs(core.WordNetLexicon, perf.FastWordNetLexicon)

        self.assertEqual(
            (
                core.norm_token,
                core.function_class,
                core.WordNetLexicon,
                core.PhraseIndex,
            ),
            before,
        )

    def test_reranker_import_and_reload_preserve_core_bindings(self) -> None:
        before = (
            core.norm_token,
            core.function_class,
            core.WordNetLexicon,
            core.PhraseIndex,
        )
        rerank = importlib.import_module("anagram_rerank")
        self.assertEqual(
            (core.norm_token, core.function_class, core.WordNetLexicon, core.PhraseIndex),
            before,
        )
        importlib.reload(rerank)
        self.assertEqual(
            (core.norm_token, core.function_class, core.WordNetLexicon, core.PhraseIndex),
            before,
        )


if __name__ == "__main__":
    unittest.main()
