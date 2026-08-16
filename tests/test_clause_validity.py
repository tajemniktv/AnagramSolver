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
            "testing": core.Features(verb=True, verb_ing=True, recognized=True),
            "university": core.Features(noun=True, noun_singular=True, recognized=True),
        }

    def features(self, word: str) -> core.Features:
        return self._features.get(
            word,
            core.Features(recognized=core.function_class(word) is not None),
        )

    def allows_object(self, word: str) -> bool | None:
        if word in {"testing", "starting"}:
            return True
        if word == "sitting":
            return False
        return None

    def allows_intransitive(self, word: str) -> bool | None:
        if word in {"runs", "aims"}:
            return True
        if word in {"testing", "starting"}:
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
        self.assertEqual(
            validity.pair_validity_adjustment("silver", "lining", self.lex),
            0.0,
        )

    def test_surface_penalty_reduces_mismatched_article_structure(self) -> None:
        result = core.StructureResult(0.90, 1.0, 1.0, 1.0, "clause", 3.6)
        adjusted = validity.apply_surface_structure_penalties(
            ("an", "game", "runs"),
            self.lex,
            result,
        )

        self.assertLess(adjusted.norm, result.norm)
        self.assertEqual(adjusted.coverage, result.coverage)

    def test_user_failure_cross_bag_structural_objective_prefers_intended_phrase(self) -> None:
        def objective(words: tuple[str, ...]) -> float:
            raw = rerank.local_grammar_raw(words, self.lex)
            grammar = core.grammar_normalize(raw)
            structure = rerank.phrase_structure(words, self.lex)
            return (
                0.38 * grammar
                + 0.44 * structure.norm
                + 0.12 * structure.valency
                + 0.06 * structure.coverage
            )

        intended = objective(("i", "am", "testing", "anagrams"))
        malformed_aux = objective(("a", "am", "sitting", "managers"))
        malformed_ing = objective(("an", "game", "starting", "aims"))

        self.assertGreater(intended, malformed_aux)
        self.assertGreater(intended, malformed_ing)


if __name__ == "__main__":
    unittest.main()
