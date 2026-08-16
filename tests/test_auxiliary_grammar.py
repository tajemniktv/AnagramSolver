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
            "researchers": core.Features(noun=True, noun_plural=True, recognized=True),
            "ball": core.Features(noun=True, noun_singular=True, recognized=True),
            "testing": core.Features(verb=True, verb_ing=True, recognized=True),
            "reading": core.Features(verb=True, verb_ing=True, recognized=True),
            "playing": core.Features(verb=True, verb_ing=True, recognized=True),
            "making": core.Features(verb=True, verb_ing=True, recognized=True),
            "tested": core.Features(verb=True, verb_past=True, recognized=True),
            "read": core.Features(verb=True, verb_base=True, recognized=True),
            "test": core.Features(verb=True, verb_base=True, recognized=True),
            "recently": core.Features(adv=True, recognized=True),
        }

    def features(self, word: str) -> core.Features:
        return self._features.get(
            word,
            core.Features(recognized=core.function_class(word) is not None),
        )

    def allows_object(self, word: str) -> bool | None:
        if word in {
            "testing",
            "reading",
            "playing",
            "making",
            "tested",
            "read",
            "test",
        }:
            return True
        return None

    def allows_intransitive(self, word: str) -> bool | None:
        if word in {
            "testing",
            "reading",
            "playing",
            "making",
            "tested",
            "read",
            "test",
        }:
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

    def _structure(self, words: tuple[str, ...]) -> core.StructureResult:
        result = auxiliary.auxiliary_structure(words, self.lex)
        self.assertIsNotNone(result)
        assert result is not None
        return result

    def test_progressive_clause_consumes_subject_aux_verb_and_object(self) -> None:
        result = self._structure(("i", "am", "testing", "anagrams"))

        self.assertEqual(result.kind, "aux-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.agreement, 1.0)
        self.assertGreaterEqual(result.valency, 0.99)
        self.assertGreater(result.norm, 0.95)

    def test_negative_progressive_variants_keep_full_coverage(self) -> None:
        explicit_not = self._structure(
            ("i", "am", "not", "testing", "anagrams")
        )
        contracted = self._structure(("they", "arent", "testing", "anagrams"))

        self.assertEqual(explicit_not.kind, "aux-progressive")
        self.assertEqual(explicit_not.coverage, 1.0)
        self.assertEqual(explicit_not.agreement, 1.0)
        self.assertGreater(explicit_not.norm, 0.90)
        self.assertEqual(contracted.kind, "aux-progressive")
        self.assertEqual(contracted.coverage, 1.0)
        self.assertEqual(contracted.agreement, 1.0)
        self.assertGreater(contracted.norm, 0.90)

    def test_have_agreement_positive_and_negative_variants(self) -> None:
        she_has = self._structure(("she", "has", "tested", "anagrams"))
        they_have = self._structure(("they", "have", "tested", "anagrams"))
        she_hasnt = self._structure(("she", "hasnt", "tested", "anagrams"))

        for result in (she_has, they_have, she_hasnt):
            self.assertEqual(result.kind, "aux-perfect")
            self.assertEqual(result.coverage, 1.0)
            self.assertGreater(result.agreement, 0.90)
            self.assertGreater(result.norm, 0.90)

    def test_be_agreement_good_and_mixed_forms_are_bounded(self) -> None:
        good = (
            ("i", "non3sg", "am"),
            ("they", "non3sg", "are"),
            ("she", "3sg", "is"),
            ("we", "non3sg", "were"),
        )
        mixed = (
            ("they", "non3sg", "was"),
            ("she", "3sg", "were"),
        )

        for subject, number, verb in good:
            with self.subTest(subject=subject, verb=verb):
                self.assertGreater(
                    auxiliary.auxiliary_agreement(subject, number, verb, self.lex),
                    0.90,
                )
        for subject, number, verb in mixed:
            with self.subTest(subject=subject, verb=verb):
                score = auxiliary.auxiliary_agreement(subject, number, verb, self.lex)
                self.assertGreater(score, 0.10)
                self.assertLess(score, 0.20)

    def test_bad_be_agreement_is_strongly_capped(self) -> None:
        result = self._structure(("they", "is", "testing", "anagrams"))

        self.assertLessEqual(result.agreement, 0.10)
        self.assertLessEqual(result.norm, 0.42)

    def test_progressive_passive_chain_is_recognized(self) -> None:
        result = self._structure(("anagrams", "are", "being", "tested"))

        self.assertEqual(result.kind, "aux-progressive-passive")
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.agreement, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_passive_by_phrase_and_adverb_tail_are_consumed_exactly(self) -> None:
        by_agent = self._structure(
            ("anagrams", "were", "tested", "by", "researchers")
        )
        adverb = self._structure(("anagrams", "were", "tested", "recently"))
        by_agent_adverb = self._structure(
            ("anagrams", "were", "tested", "by", "researchers", "recently")
        )
        trailing_junk = self._structure(
            ("anagrams", "were", "tested", "by", "researchers", "systems")
        )

        for result in (by_agent, adverb, by_agent_adverb):
            self.assertEqual(result.kind, "aux-passive")
            self.assertEqual(result.coverage, 1.0)
            self.assertGreaterEqual(result.valency, 0.92)
        self.assertLess(trailing_junk.coverage, 1.0)
        self.assertAlmostEqual(trailing_junk.coverage, 5 / 6)

    def test_perfect_progressive_chain_is_recognized(self) -> None:
        result = self._structure(("we", "have", "been", "testing", "systems"))

        self.assertEqual(result.kind, "aux-perfect-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_modal_progressive_chain_is_recognized(self) -> None:
        result = self._structure(("we", "will", "be", "testing", "systems"))

        self.assertEqual(result.kind, "aux-modal-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_modal_perfect_and_passive_chains_are_recognized(self) -> None:
        perfect = self._structure(("we", "will", "have", "tested", "systems"))
        perfect_progressive = self._structure(
            ("we", "could", "have", "been", "testing", "systems")
        )
        perfect_passive = self._structure(
            ("systems", "could", "have", "been", "tested")
        )

        self.assertEqual(perfect.kind, "aux-modal-perfect")
        self.assertEqual(perfect_progressive.kind, "aux-modal-perfect-progressive")
        self.assertEqual(perfect_passive.kind, "aux-modal-perfect-passive")
        for result in (perfect, perfect_progressive, perfect_passive):
            self.assertEqual(result.coverage, 1.0)
            self.assertGreater(result.norm, 0.88)

    def test_do_support_variants_are_recognized(self) -> None:
        cases = (
            ("they", "do", "test", "systems"),
            ("they", "did", "test", "systems"),
            ("they", "dont", "test", "systems"),
            ("she", "doesnt", "test", "systems"),
        )
        for words in cases:
            with self.subTest(words=words):
                result = self._structure(words)
                self.assertEqual(result.kind, "aux-do-support")
                self.assertEqual(result.coverage, 1.0)
                self.assertGreater(result.norm, 0.80)

    def test_nonfinite_be_forms_do_not_start_finite_clause_parses(self) -> None:
        for words in (
            ("the", "ball", "being", "tested"),
            ("the", "ball", "been", "tested"),
            ("the", "ball", "be", "tested"),
        ):
            with self.subTest(words=words):
                self.assertIsNone(auxiliary.auxiliary_structure(words, self.lex))

    def test_simple_passive_is_complete_without_object(self) -> None:
        result = self._structure(("the", "ball", "was", "tested"))

        self.assertEqual(result.kind, "aux-passive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreaterEqual(result.valency, 0.95)

    def test_auxiliary_pair_bonus_is_morphology_specific(self) -> None:
        self.assertGreater(
            auxiliary.auxiliary_pair_bonus("am", "testing", self.lex), 0.0
        )
        self.assertGreater(
            auxiliary.auxiliary_pair_bonus("have", "tested", self.lex), 0.0
        )
        self.assertGreater(
            auxiliary.auxiliary_pair_bonus("have", "been", self.lex), 0.0
        )
        self.assertEqual(
            auxiliary.auxiliary_pair_bonus("have", "testing", self.lex), 0.0
        )
        self.assertGreater(
            auxiliary.auxiliary_pair_bonus("is", "tested", self.lex), 0.0
        )
        self.assertEqual(
            auxiliary.auxiliary_pair_bonus("testing", "anagrams", self.lex), 0.0
        )
        self.assertEqual(
            auxiliary.auxiliary_pair_bonus("am", "anagrams", self.lex), 0.0
        )

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

    def test_local_grammar_raw_with_auxiliaries_end_to_end(self) -> None:
        def base_tables(_words, _lex):
            size = len(_words)
            pair = tuple(tuple(0.0 for _ in range(size)) for _ in range(size))
            return pair, tuple(0.0 for _ in range(size)), tuple(0.0 for _ in range(size))

        score_with_aux = auxiliary.local_grammar_raw_with_auxiliaries(
            ("i", "am", "testing", "anagrams"), self.lex, base_tables
        )
        score_without_aux = auxiliary.local_grammar_raw_with_auxiliaries(
            ("i", "test", "anagrams"), self.lex, base_tables
        )

        self.assertGreater(score_with_aux, 0.0)
        self.assertEqual(score_without_aux, 0.0)

    def test_install_auxiliary_scoring_is_idempotent(self) -> None:
        rerank._install_auxiliary_scoring()
        first = (
            rerank.impl.phrase_structure,
            rerank.impl._order_local_tables,
            rerank.impl.local_grammar_raw,
        )
        rerank._install_auxiliary_scoring()

        self.assertEqual(
            first,
            (
                rerank.impl.phrase_structure,
                rerank.impl._order_local_tables,
                rerank.impl.local_grammar_raw,
            ),
        )
        self.assertIs(rerank.impl.phrase_structure, rerank.phrase_structure)
        self.assertIs(rerank.impl._order_local_tables, rerank._order_local_tables)
        self.assertIs(rerank.impl.local_grammar_raw, rerank.local_grammar_raw)
        self.assertIs(
            rerank.impl._worker_init, rerank._worker_init_with_auxiliary_scoring
        )

    def test_worker_init_reinstalls_auxiliary_hooks(self) -> None:
        def reset_hooks(*_args) -> None:
            rerank.impl.phrase_structure = rerank._BASE_PHRASE_STRUCTURE
            rerank.impl._order_local_tables = rerank._BASE_ORDER_LOCAL_TABLES
            rerank.impl.local_grammar_raw = core.local_grammar_raw

        with patch.object(rerank, "_BASE_WORKER_INIT", side_effect=reset_hooks) as base_init:
            rerank._worker_init_with_auxiliary_scoring("wn", "auto", 128, 5, 56)

        base_init.assert_called_once_with("wn", "auto", 128, 5, 56)
        self.assertIs(rerank.impl.phrase_structure, rerank.phrase_structure)
        self.assertIs(rerank.impl._order_local_tables, rerank._order_local_tables)
        self.assertIs(rerank.impl.local_grammar_raw, rerank.local_grammar_raw)


if __name__ == "__main__":
    unittest.main()
