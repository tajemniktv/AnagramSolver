from __future__ import annotations

import unittest

import anagram_clause_validity as validity
import anagram_rerank_core as core


class Lexicon:
    def features(self, word):
        if word in {"professional", "young", "light", "red"}:
            return core.Features(noun=True, adj=True, noun_singular=True)
        if word in {"repaired", "faded", "ran"}:
            return core.Features(verb=True, verb_past=True)
        if word in {"engine", "mechanic", "engines", "car"}:
            return core.Features(noun=True, noun_singular=True)
        return core.Features()


class NounPhraseHeadTests(unittest.TestCase):
    def test_finite_be_collision_is_not_a_nominal_complement(self):
        result = core.StructureResult(1.0, 1.0, 1.0, 1.0, "clause", 4.0)
        penalized = validity.apply_surface_structure_penalties(
            ["cat", "is", "was"], Lexicon(), result
        )
        self.assertLess(penalized.norm, 0.8)
        self.assertLess(validity.pair_validity_adjustment("is", "was", Lexicon()), 0.0)
        for left, right in (("has", "been"), ("is", "being"), ("had", "had")):
            self.assertEqual(
                validity.pair_validity_adjustment(left, right, Lexicon()), 0.0
            )

    def test_ambiguous_head_is_not_discarded_as_an_adjective(self):
        lex = Lexicon()
        for phrase, end in (
            ("a professional", 1),
            ("the young", 1),
            ("the light faded", 1),
            ("the red car", 2),
        ):
            with self.subTest(phrase=phrase):
                span = core._np_span_starting_at(phrase.split(), 0, lex)
                self.assertIsNotNone(span)
                self.assertEqual(span[0], end)

    def test_determiner_led_participle_is_an_attributive_modifier(self):
        words = "the repaired engine".split()
        self.assertEqual(core._np_span_starting_at(words, 0, Lexicon())[0], 2)
        self.assertEqual(core._np_span_ending_at(words, 2, Lexicon())[0], 0)

    def test_finite_predicate_does_not_become_a_modifier(self):
        words = "the mechanic repaired engines".split()
        self.assertEqual(core._np_span_starting_at(words, 0, Lexicon())[0], 1)
        self.assertEqual(core._np_span_ending_at(words, 3, Lexicon())[0], 3)
        self.assertIsNone(core._np_span_starting_at("the ran engine".split(), 0, Lexicon()))
