from __future__ import annotations

import unittest

import anagram_performance as perf
import anagram_rerank_core as core


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

    def test_install_is_idempotent(self) -> None:
        perf.install_performance_hooks()
        first_class = core.WordNetLexicon
        first_norm = core.norm_token
        first_function_class = core.function_class
        perf.install_performance_hooks()
        self.assertIs(core.WordNetLexicon, first_class)
        self.assertIs(core.norm_token, first_norm)
        self.assertIs(core.function_class, first_function_class)


if __name__ == "__main__":
    unittest.main()
