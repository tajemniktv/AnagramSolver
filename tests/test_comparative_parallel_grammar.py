from __future__ import annotations

import unittest

import anagram_auxiliary_grammar as grammar
import anagram_comparative_grammar as comparatives
import anagram_rerank_core as core


class _FakeLexicon:
    def __init__(self) -> None:
        self._features = {
            "actions": core.Features(noun=True, noun_plural=True, recognized=True),
            "words": core.Features(noun=True, noun_plural=True, recognized=True),
            "speak": core.Features(verb=True, verb_base=True, recognized=True),
            # Match the real WordNet gap that motivated the active comparative
            # morphology layer: the base is lexical, while the inflected surface
            # form itself need not appear in index.adj/index.adv.
            "loud": core.Features(adj=True, adv=True, recognized=True),
            "nice": core.Features(adj=True, recognized=True),
            "happy": core.Features(adj=True, recognized=True),
            "big": core.Features(adj=True, recognized=True),
            # True negatives: a surface adjective with no recovered adj/adv base,
            # and an -er form whose apparent base is verb-only.
            "silver": core.Features(adj=True, recognized=True),
            "work": core.Features(verb=True, verb_base=True, recognized=True),
            # Homographic surfaces. Each has a legitimate regular comparative
            # analysis, but the surface is independently indexed under another
            # POS, so local evidence must stay weak rather than binary.
            "closer": core.Features(noun=True, recognized=True),
            "close": core.Features(adj=True, adv=True, verb=True, recognized=True),
            "number": core.Features(noun=True, recognized=True),
            "numb": core.Features(adj=True, recognized=True),
            "buffer": core.Features(noun=True, verb=True, recognized=True),
            "buff": core.Features(adj=True, noun=True, verb=True, recognized=True),
            "elder": core.Features(adj=True, recognized=True),
            "farther": core.Features(adj=True, adv=True, recognized=True),
            "further": core.Features(adj=True, adv=True, recognized=True),
            "run": core.Features(verb=True, verb_base=True, recognized=True),
            "today": core.Features(adv=True, recognized=True),
            "ever": core.Features(adv=True, recognized=True),
            "united": core.Features(
                verb=True,
                verb_past=True,
                adj=True,
                recognized=True,
            ),
            "divided": core.Features(
                verb=True,
                verb_past=True,
                adj=True,
                recognized=True,
            ),
            "called": core.Features(
                verb=True,
                verb_past=True,
                recognized=True,
            ),
            "stand": core.Features(verb=True, verb_base=True, recognized=True),
            "fall": core.Features(verb=True, verb_base=True, recognized=True),
            "standing": core.Features(verb=True, verb_ing=True, recognized=True),
            "falling": core.Features(verb=True, verb_ing=True, recognized=True),
            "old": core.Features(adj=True, recognized=True),
            "young": core.Features(adj=True, recognized=True),
        }

    def features(self, word: str) -> core.Features:
        return self._features.get(
            word,
            core.Features(recognized=core.function_class(word) is not None),
        )

    def allows_intransitive(self, word: str) -> bool | None:
        if word in {"stand", "fall"}:
            return True
        if word == "speak":
            return False
        return None

    def allows_object(self, word: str) -> bool | None:
        del word
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


class ComparativeParallelGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lex = _FakeLexicon()

    def test_regular_comparative_evidence_recovers_lexical_bases(self) -> None:
        for word, base in (
            ("louder", "loud"),
            ("nicer", "nice"),
            ("happier", "happy"),
            ("bigger", "big"),
        ):
            with self.subTest(word=word):
                evidence = comparatives.comparative_evidence(word, self.lex)
                self.assertTrue(evidence.present)
                self.assertEqual(evidence.base, base)
                self.assertGreaterEqual(evidence.confidence, 0.85)

    def test_false_er_friends_have_no_comparative_evidence(self) -> None:
        for word in ("silver", "worker", "her", "far"):
            with self.subTest(word=word):
                evidence = comparatives.comparative_evidence(word, self.lex)
                self.assertFalse(evidence.present)
                self.assertEqual(
                    grammar.construction_pair_bonus(word, "than", self.lex),
                    0.0,
                )

    def test_homographic_er_surfaces_keep_weak_comparative_evidence(self) -> None:
        strong_bonus = grammar.construction_pair_bonus("louder", "than", self.lex)
        for word, base in (
            ("closer", "close"),
            ("number", "numb"),
            ("buffer", "buff"),
        ):
            with self.subTest(word=word):
                evidence = comparatives.comparative_evidence(word, self.lex)
                self.assertTrue(evidence.present)
                self.assertEqual(evidence.base, base)
                self.assertEqual(evidence.source, "regular-ambiguous")
                self.assertLess(evidence.confidence, 0.50)
                local_bonus = grammar.construction_pair_bonus(word, "than", self.lex)
                self.assertGreater(local_bonus, 0.0)
                self.assertLess(local_bonus, 1.0)
                self.assertLess(local_bonus, strong_bonus)

    def test_curated_irregular_er_comparatives_remain_strong(self) -> None:
        for word in ("elder", "farther", "further"):
            with self.subTest(word=word):
                evidence = comparatives.comparative_evidence(word, self.lex)
                self.assertTrue(evidence.present)
                self.assertGreater(evidence.confidence, 0.90)
                self.assertGreater(
                    grammar.construction_pair_bonus(word, "than", self.lex),
                    1.0,
                )

    def test_complete_span_can_contextualize_ambiguous_comparative(self) -> None:
        result = comparatives.comparative_span_starting_at(
            ("closer", "than", "words"),
            0,
            self.lex,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0], 2)
        self.assertGreater(result[1], 0.85)

    def test_lexical_comparative_spans_remain_supported(self) -> None:
        for words in (
            ("better", "than", "words"),
            ("more", "words", "than", "actions"),
        ):
            with self.subTest(words=words):
                self.assertIsNotNone(
                    comparatives.comparative_span_starting_at(words, 0, self.lex)
                )

    def test_elliptical_adjective_and_adverb_complements_remain_supported(self) -> None:
        for complement in ("old", "today", "ever"):
            with self.subTest(complement=complement):
                self.assertIsNotNone(
                    comparatives.comparative_span_starting_at(
                        ("better", "than", complement),
                        0,
                        self.lex,
                    )
                )
                self.assertGreater(
                    grammar.construction_pair_bonus("than", complement, self.lex),
                    0.0,
                )

    def test_comparative_clause_consumes_complete_tail(self) -> None:
        result = grammar.comparative_clause_structure(
            ("actions", "speak", "louder", "than", "words"),
            self.lex,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "comparative-clause")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.agreement, 0.90)
        self.assertGreater(result.norm, 0.90)

    def test_comparative_clause_rejects_scrambled_tail(self) -> None:
        self.assertIsNone(
            grammar.comparative_clause_structure(
                ("actions", "speak", "than", "words", "louder"),
                self.lex,
            )
        )

    def test_comparative_clause_rejects_plain_adjective_before_than(self) -> None:
        self.assertIsNone(
            grammar.comparative_clause_structure(
                ("actions", "speak", "old", "than", "words"),
                self.lex,
            )
        )

    def test_comparative_clause_rejects_non_tail_comparative_span(self) -> None:
        self.assertIsNone(
            grammar.comparative_clause_structure(
                ("actions", "speak", "louder", "than", "words", "today"),
                self.lex,
            )
        )

    def test_comparative_clause_rejects_non_nominal_than_complement(self) -> None:
        self.assertIsNone(
            grammar.comparative_clause_structure(
                ("actions", "speak", "louder", "than", "run"),
                self.lex,
            )
        )

    def test_comparative_pair_evidence_is_directional(self) -> None:
        self.assertGreater(
            grammar.construction_pair_bonus("louder", "than", self.lex),
            1.0,
        )
        self.assertGreater(
            grammar.construction_pair_bonus("than", "words", self.lex),
            0.0,
        )
        self.assertEqual(
            grammar.construction_pair_bonus("than", "louder", self.lex),
            0.0,
        )

    def test_parallel_clause_recognizes_repeated_subject_symmetry(self) -> None:
        result = grammar.parallel_clause_structure(
            ("united", "we", "stand", "divided", "we", "fall"),
            self.lex,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "parallel-clause")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.agreement, 0.90)
        self.assertGreater(result.valency, 0.90)
        self.assertGreater(result.norm, 0.95)

    def test_parallel_clause_accepts_adjective_fronted_predicates(self) -> None:
        result = grammar.parallel_clause_structure(
            ("old", "we", "stand", "young", "we", "fall"),
            self.lex,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.kind, "parallel-clause")
        self.assertEqual(result.coverage, 1.0)
        self.assertGreater(result.agreement, 0.90)
        self.assertGreater(result.valency, 0.90)
        self.assertGreater(result.norm, 0.94)

    def test_parallel_clause_rejects_scrambled_topology(self) -> None:
        self.assertIsNone(
            grammar.parallel_clause_structure(
                ("we", "fall", "divided", "united", "stand", "we"),
                self.lex,
            )
        )

    def test_parallel_clause_rejects_mismatched_subjects(self) -> None:
        self.assertIsNone(
            grammar.parallel_clause_structure(
                ("united", "we", "stand", "divided", "they", "fall"),
                self.lex,
            )
        )

    def test_parallel_clause_rejects_non_finite_verbs(self) -> None:
        self.assertIsNone(
            grammar.parallel_clause_structure(
                ("united", "we", "standing", "divided", "we", "falling"),
                self.lex,
            )
        )

    def test_participial_subject_bonus_stays_narrow(self) -> None:
        self.assertGreater(
            grammar.construction_pair_bonus("united", "we", self.lex),
            0.0,
        )
        self.assertEqual(
            grammar.construction_pair_bonus("old", "we", self.lex),
            0.0,
        )
        self.assertEqual(
            grammar.construction_pair_bonus("called", "we", self.lex),
            0.0,
        )

    def test_structure_wrapper_prefers_new_full_constructions(self) -> None:
        weak = core.StructureResult(0.20, 0.50, 0.25, 0.50, "fragment", 0.80)

        def base_structure(_words, _lex):
            return weak

        comparative = grammar.phrase_structure_with_auxiliaries(
            ("actions", "speak", "louder", "than", "words"),
            self.lex,
            base_structure,
        )
        parallel = grammar.phrase_structure_with_auxiliaries(
            ("united", "we", "stand", "divided", "we", "fall"),
            self.lex,
            base_structure,
        )
        self.assertEqual(comparative.kind, "comparative-clause")
        self.assertEqual(parallel.kind, "parallel-clause")

    def test_search_and_realized_scores_share_construction_bonuses(self) -> None:
        words = ("actions", "speak", "louder", "than", "words")

        def base_tables(_words, _lex):
            size = len(_words)
            pair = tuple(tuple(0.0 for _ in range(size)) for _ in range(size))
            zeros = tuple(0.0 for _ in range(size))
            return pair, zeros, zeros

        pair, _starts, _ends = grammar.order_local_tables_with_auxiliaries(
            words,
            self.lex,
            base_tables,
        )
        realized = grammar.local_grammar_raw_with_auxiliaries(
            words,
            self.lex,
            base_tables,
        )
        expected = sum(pair[i][i + 1] for i in range(len(words) - 1)) / (
            len(words) - 1
        )
        self.assertAlmostEqual(realized, expected)


if __name__ == "__main__":
    unittest.main()
