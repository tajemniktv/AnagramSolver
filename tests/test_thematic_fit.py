from __future__ import annotations

import unittest

from anagram_thematic_fit import score_thematic_fit


def _lookup(mapping: dict[str, int]):
    def counts(phrases: tuple[str, ...]) -> dict[str, int]:
        return {phrase: mapping[phrase] for phrase in phrases if phrase in mapping}

    return counts


class ThematicFitTests(unittest.TestCase):
    def test_directional_corpus_evidence_prefers_dog_chasing_ball(self) -> None:
        counts = _lookup(
            {
                "dog chased": 1800,
                "the dog chased": 1400,
                "chased the ball": 1600,
                "chased ball": 300,
                "ball chased": 12,
                "the ball chased": 9,
                "chased the dog": 25,
                "chased dog": 8,
            }
        )
        is_verb = lambda word: word == "chased"
        is_nominal = lambda word: word in {"dog", "ball"}

        good = score_thematic_fit(
            ("the", "dog", "chased", "the", "ball"),
            counts=counts,
            is_verb=is_verb,
            is_nominal=is_nominal,
        )
        bad = score_thematic_fit(
            ("the", "ball", "chased", "the", "dog"),
            counts=counts,
            is_verb=is_verb,
            is_nominal=is_nominal,
        )

        self.assertGreater(good.score, bad.score)
        self.assertGreater(good.subject_strength, bad.subject_strength)
        self.assertGreater(good.object_strength, bad.object_strength)
        self.assertEqual(good.verb_coverage, 1.0)

    def test_directional_corpus_evidence_prefers_phone_needs_charge(self) -> None:
        counts = _lookup(
            {
                "phone needs": 900,
                "my phone needs": 700,
                "needs a charge": 1300,
                "charge needs": 4,
                "my charge needs": 2,
                "needs a phone": 35,
            }
        )
        is_verb = lambda word: word == "needs"
        is_nominal = lambda word: word in {"phone", "charge"}

        good = score_thematic_fit(
            ("my", "phone", "needs", "a", "charge"),
            counts=counts,
            is_verb=is_verb,
            is_nominal=is_nominal,
        )
        bad = score_thematic_fit(
            ("my", "charge", "needs", "a", "phone"),
            counts=counts,
            is_verb=is_verb,
            is_nominal=is_nominal,
        )
        self.assertGreater(good.score, bad.score)

    def test_absent_or_nonverbal_evidence_is_neutral(self) -> None:
        missing = score_thematic_fit(
            ("novel", "words", "combine"),
            counts=_lookup({}),
            is_verb=lambda word: word == "combine",
            is_nominal=lambda word: word in {"novel", "words"},
        )
        no_verb = score_thematic_fit(
            ("quiet", "blue", "room"),
            counts=_lookup({"blue room": 100}),
            is_verb=lambda _word: False,
            is_nominal=lambda word: word == "room",
        )

        self.assertEqual(missing.score, 0.0)
        self.assertEqual(missing.verbs_with_evidence, 0)
        self.assertEqual(no_verb.score, 0.0)
        self.assertEqual(no_verb.spans, ())


if __name__ == "__main__":
    unittest.main()
