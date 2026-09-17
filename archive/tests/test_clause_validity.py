from __future__ import annotations

import unittest

import anagram_clause_validity as validity
import anagram_auxiliary_grammar as auxiliary
import anagram_rerank as rerank
import anagram_rerank_core as core


class _FakeLexicon:
    def __init__(self) -> None:
        self._features = {
            "a": core.Features(noun=True, recognized=True),
            "an": core.Features(recognized=True),
            "anagrams": core.Features(noun=True, noun_plural=True, recognized=True),
            "aims": core.Features(
                noun=True,
                verb=True,
                noun_plural=True,
                verb_3sg=True,
                recognized=True,
            ),
            "cat": core.Features(noun=True, noun_singular=True, recognized=True),
            "cats": core.Features(noun=True, noun_plural=True, recognized=True),
            "dog": core.Features(noun=True, noun_singular=True, recognized=True),
            "game": core.Features(
                noun=True,
                verb=True,
                noun_singular=True,
                verb_base=True,
                recognized=True,
            ),
            "hour": core.Features(noun=True, noun_singular=True, recognized=True),
            "lining": core.Features(noun=True, verb=True, verb_ing=True, recognized=True),
            "managers": core.Features(noun=True, noun_plural=True, recognized=True),
            "runs": core.Features(verb=True, verb_3sg=True, recognized=True),
            "silver": core.Features(noun=True, adj=True, recognized=True),
            "sitting": core.Features(
                noun=True,
                verb=True,
                verb_ing=True,
                recognized=True,
            ),
            "starting": core.Features(
                noun=True,
                verb=True,
                verb_ing=True,
                recognized=True,
            ),
            "test": core.Features(verb=True, verb_base=True, recognized=True),
            "testing": core.Features(verb=True, verb_ing=True, recognized=True),
            "university": core.Features(noun=True, noun_singular=True, recognized=True),
        }

    def features(self, word: str) -> core.Features:
        return self._features.get(
            word,
            core.Features(recognized=core.function_class(word) is not None),
        )

    def allows_object(self, word: str) -> bool | None:
        if word in {"test", "testing", "starting"}:
            return True
        if word == "sitting":
            return False
        return None

    def allows_intransitive(self, word: str) -> bool | None:
        if word in {"runs", "aims"}:
            return True
        if word in {"test", "testing", "starting"}:
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


class ClauseValidityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lex = _FakeLexicon()

    def test_function_word_noun_sense_cannot_head_auxiliary_subject(self) -> None:
        self.assertFalse(validity.valid_subject_head("a", self.lex))
        self.assertIsNone(
            auxiliary.auxiliary_structure(
                ("a", "am", "sitting", "managers"),
                self.lex,
            )
        )

    def test_valid_pronoun_auxiliary_subject_is_preserved(self) -> None:
        result = auxiliary.auxiliary_structure(
            ("i", "am", "testing", "anagrams"),
            self.lex,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "aux-progressive")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.norm, 0.90)

    def test_bare_ing_does_not_claim_full_finite_clause_coverage(self) -> None:
        words = ("an", "game", "starting", "aims")
        base = core.phrase_structure(words, self.lex)
        adjusted = validity.adjust_base_clause_structure(words, self.lex, base)

        self.assertEqual(base.kind, "clause")
        self.assertGreater(base.coverage, 0.90)
        self.assertLess(adjusted.coverage, base.coverage)
        self.assertLess(adjusted.norm, base.norm)

    def test_aux_with_invalid_subject_is_demoted_to_fragment(self) -> None:
        base = core.StructureResult(0.90, 1.0, 1.0, 1.0, "clause", 3.6)
        adjusted = validity.adjust_base_clause_structure(
            ("a", "am", "testing", "anagrams"),
            self.lex,
            base,
        )

        self.assertEqual(adjusted.kind, "fragment")
        self.assertLess(adjusted.coverage, base.coverage)
        self.assertLess(adjusted.norm, base.norm)
        self.assertLess(adjusted.valency, base.valency)
        self.assertLess(adjusted.agreement, base.agreement)

    def test_nominal_function_subject_requires_compatible_auxiliary(self) -> None:
        base = core.StructureResult(0.90, 1.0, 1.0, 1.0, "copula", 3.6)
        valid = validity.adjust_base_clause_structure(
            ("one", "is", "enough"),
            self.lex,
            base,
        )
        invalid = validity.adjust_base_clause_structure(
            ("one", "am", "enough"),
            self.lex,
            base,
        )

        self.assertEqual(valid, base)
        self.assertEqual(invalid.kind, "fragment")
        self.assertLess(invalid.norm, base.norm)
        self.assertLess(invalid.coverage, base.coverage)

    def test_subjectless_do_imperative_clause_is_preserved(self) -> None:
        base = core.StructureResult(0.80, 0.90, 1.0, 0.60, "clause", 3.2)
        for words in (
            ("do", "test", "anagrams"),
            ("dont", "test", "anagrams"),
        ):
            with self.subTest(words=words):
                adjusted = validity.adjust_base_clause_structure(words, self.lex, base)
                self.assertEqual(adjusted, base)

    def test_subjectless_do_imperative_cannot_hide_later_bad_auxiliary(self) -> None:
        base = core.StructureResult(0.80, 0.90, 1.0, 0.60, "clause", 3.2)
        adjusted = validity.adjust_base_clause_structure(
            ("do", "test", "am", "managers"),
            self.lex,
            base,
        )

        self.assertEqual(adjusted.kind, "fragment")
        self.assertLess(adjusted.norm, base.norm)
        self.assertLess(adjusted.coverage, base.coverage)

    def test_normal_finite_clause_is_not_demoted(self) -> None:
        words = ("the", "dog", "runs")
        base = core.phrase_structure(words, self.lex)
        adjusted = validity.adjust_base_clause_structure(words, self.lex, base)

        self.assertEqual(base.kind, "clause")
        self.assertEqual(adjusted, base)

    def test_indefinite_article_mismatch_is_bounded_and_handles_common_exceptions(self) -> None:
        self.assertTrue(validity.indefinite_article_mismatch("an", "game"))
        self.assertTrue(validity.indefinite_article_mismatch("a", "apple"))
        self.assertFalse(validity.indefinite_article_mismatch("a", "game"))
        self.assertFalse(validity.indefinite_article_mismatch("an", "hour"))
        self.assertFalse(validity.indefinite_article_mismatch("a", "university"))

    def test_pair_adjustments_stay_narrow_and_preserve_noun_compounds(self) -> None:
        self.assertLess(validity.pair_validity_adjustment("a", "am", self.lex), 0.0)
        self.assertLess(validity.pair_validity_adjustment("an", "game", self.lex), 0.0)
        self.assertEqual(validity.pair_validity_adjustment("a", "game", self.lex), 0.0)
        self.assertEqual(validity.pair_validity_adjustment("one", "is", self.lex), 0.0)
        self.assertLess(validity.pair_validity_adjustment("one", "am", self.lex), 0.0)
        self.assertEqual(
            validity.pair_validity_adjustment("silver", "lining", self.lex),
            0.0,
        )

    def test_search_and_realized_local_scores_apply_pair_validity_once(self) -> None:
        words = ("a", "am")
        pair, starts, ends = rerank._order_local_tables(words, self.lex)
        realized = rerank.local_grammar_raw(words, self.lex)
        expected = starts[0] + pair[0][1] + ends[1]

        self.assertAlmostEqual(realized, expected)

    def test_surface_penalty_reduces_mismatched_article_structure(self) -> None:
        result = core.StructureResult(0.90, 1.0, 1.0, 1.0, "clause", 3.6)
        adjusted = validity.apply_surface_structure_penalties(
            ("an", "game", "runs"),
            self.lex,
            result,
        )

        self.assertLess(adjusted.norm, result.norm)
        self.assertEqual(adjusted.coverage, result.coverage)

    def test_surface_penalty_reduces_determiner_aux_structure(self) -> None:
        result = core.StructureResult(0.90, 1.0, 1.0, 1.0, "clause", 3.6)
        adjusted = validity.apply_surface_structure_penalties(
            ("the", "am", "sitting"),
            self.lex,
            result,
        )

        self.assertFalse(validity.indefinite_article_mismatch("the", "am"))
        self.assertLess(adjusted.norm, result.norm)
        self.assertEqual(adjusted.coverage, result.coverage)
        self.assertEqual(adjusted.valency, result.valency)
        self.assertEqual(adjusted.agreement, result.agreement)

    def test_nominal_function_subject_avoids_determiner_aux_surface_penalty(self) -> None:
        result = core.StructureResult(0.90, 1.0, 1.0, 1.0, "clause", 3.6)
        valid = validity.apply_surface_structure_penalties(
            ("one", "is", "enough"),
            self.lex,
            result,
        )
        invalid = validity.apply_surface_structure_penalties(
            ("one", "am", "enough"),
            self.lex,
            result,
        )

        self.assertEqual(valid, result)
        self.assertLess(invalid.norm, result.norm)

    def test_user_failure_cross_bag_structural_objective_prefers_intended_phrase(self) -> None:
        def objective(words: tuple[str, ...]) -> float:
            return validity.grammar_structure_objective(
                words,
                self.lex,
                rerank.local_grammar_raw,
                rerank.phrase_structure,
            )

        intended = objective(("i", "am", "testing", "anagrams"))
        malformed_aux = objective(("a", "am", "sitting", "managers"))
        malformed_ing = objective(("an", "game", "starting", "aims"))

        self.assertGreater(intended, malformed_aux)
        self.assertGreater(intended, malformed_ing)


if __name__ == "__main__":
    unittest.main()
