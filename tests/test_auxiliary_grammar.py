from __future__ import annotations

import unittest
from unittest.mock import patch

import anagram_auxiliary_grammar as auxiliary
import anagram_rerank as rerank
import anagram_rerank_core as core


class _FakeLexicon:
    def __init__(self) -> None:
        self._features = {
            "anagrams": core.Features(noun=True, noun_plural=True, recognized=True),
            "systems": core.Features(noun=True, noun_plural=True, recognized=True),
            "books": core.Features(noun=True, noun_plural=True, recognized=True),
            "ball": core.Features(noun=True, noun_singular=True, recognized=True),
            "testing": core.Features(verb=True, verb_ing=True, recognized=True),
            "reading": core.Features(verb=True, verb_ing=True, recognized=True),
            "playing": core.Features(verb=True, verb_ing=True, recognized=True),
            "making": core.Features(verb=True, verb_ing=True, recognized=True),
            "tested": core.Features(verb=True, verb_past=True, recognized=True),
            "read": core.Features(verb=True, verb_base=True, recognized=True),
            "test": core.Features(verb=True, verb_base=True, recognized=True),
        }

    def features(self, word: str) -> core.Features:
        return self._features.get(word, core.Features(recognized=core.function_class(word) is not None))

    def allows_object(self, word: str) -> bool | None:
        return word in {"testing", "reading", "playing", "making", "read", "test"}

    def allows_intransitive(self, word: str) -> bool | None:
        if word in {"testing", "reading", "playing", "making", "read", "test"}:
            return False
        return None

    def allows_pp(self, word: str) -> bool | None:
        del word
        return None

    def allows_predicative(self, word: str) -> bool | None:
        del word
        return None

    def allows_object_predicative(self, word: str) -> bool | None:
        del word
        return None

    def allows_infinitive_or_gerund(self, word: str) -> bool | None:
        del word
        return None

    def allows_clausal(self, word: str) -> bool | None:
        del word
        return None


class AuxiliaryGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lex = _FakeLexicon()

    def test_progressive_clause_consumes_subject_aux_verb_and_object(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("i", "am", "testing", "anagrams"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.agreement, 1.0)
        self.assertGreaterEqual(result.valency, 0.99)
        self.assertGreater(result.norm, 0.95)

    def test_bad_be_agreement_is_strongly_capped(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("they", "is", "testing", "anagrams"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertLessEqual(result.agreement, 0.10)
        self.assertLessEqual(result.norm, 0.42)

    def test_progressive_passive_chain_is_recognized(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("anagrams", "are", "being", "tested"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-progressive-passive")
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.agreement, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_perfect_progressive_chain_is_recognized(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("we", "have", "been", "testing", "systems"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-perfect-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_modal_progressive_chain_is_recognized(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("we", "will", "be", "testing", "systems"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-modal-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_simple_passive_is_complete_without_object(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("the", "ball", "was", "tested"), self.lex
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-passive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreaterEqual(result.valency, 0.95)

    def test_auxiliary_pair_bonus_is_morphology_specific(self) -> None:
        self.assertGreater(auxiliary.auxiliary_pair_bonus("am", "testing", self.lex), 0.0)
        self.assertGreater(auxiliary.auxiliary_pair_bonus("have", "tested", self.lex), 0.0)
        self.assertEqual(auxiliary.auxiliary_pair_bonus("am", "anagrams", self.lex), 0.0)

    def test_table_wrapper_adds_bonus_without_changing_endpoints(self) -> None:
        words = ("i", "am", "testing", "anagrams")

        def base_tables(_words, _lex):
            size = len(_words)
            pair = tuple(tuple(0.0 for _ in range(size)) for _ in range(size))
            return pair, tuple(0.1 for _ in range(size)), tuple(0.2 for _ in range(size))

        pair, starts, ends = auxiliary.order_local_tables_with_auxiliaries(
            words, self.lex, base_tables
        )

        self.assertGreater(pair[1][2], 0.0)
        self.assertEqual(pair[0][1], 0.0)
        self.assertEqual(starts, (0.1, 0.1, 0.1, 0.1))
        self.assertEqual(ends, (0.2, 0.2, 0.2, 0.2))

    def test_reranker_installs_auxiliary_hooks_for_ranker_and_workers(self) -> None:
        rerank._install_auxiliary_scoring()
        self.assertIs(rerank.impl.phrase_structure, rerank.phrase_structure)
        self.assertIs(rerank.impl._order_local_tables, rerank._order_local_tables)
        self.assertIs(rerank.impl.local_grammar_raw, rerank.local_grammar_raw)

        with patch.object(rerank, "_BASE_WORKER_INIT") as base_init:
            rerank._worker_init_with_auxiliary_scoring("wn", "auto", 128, 5, 56)

        base_init.assert_called_once_with("wn", "auto", 128, 5, 56)
        self.assertIs(rerank.impl.phrase_structure, rerank.phrase_structure)


if __name__ == "__main__":
    unittest.main()
