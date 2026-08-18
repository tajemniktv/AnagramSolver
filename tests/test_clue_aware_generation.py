from __future__ import annotations

import unittest

import anagram_generate as generator


def sig(a=0, b=0, c=0):
    return (a, b, c) + (0,) * 23


class ClueAwareGenerationTests(unittest.TestCase):
    def _candidates(self) -> list[generator.Candidate]:
        return [
            generator.Candidate("a", sig(1), 1, 5.0),
            generator.Candidate("b", sig(0, 1), 1, 4.9),
            generator.Candidate("ab", sig(1, 1), 2, 4.8),
        ]

    def test_cap_counts_clue_accepted_results_not_irrelevant_exact_sets(self) -> None:
        stats = generator.SearchStats()
        actual = list(
            generator.solve(
                sig(2, 2),
                self._candidates(),
                2,
                4,
                1,
                True,
                clue_words={"a"},
                hint_mode="any",
                stats=stats,
            )
        )

        # Historical uncued search would find ("ab", "ab") first. Because that
        # branch can no longer include clue word "a", clue-aware search prunes it
        # before the terminal leaf and spends the one-result budget on the first
        # clue-valid bag instead.
        self.assertEqual(actual, [("a", "b", "ab")])
        self.assertEqual(stats.accepted, 1)
        self.assertEqual(stats.exact_examined, 1)

    def test_exactly_one_counts_distinct_clue_words(self) -> None:
        actual = list(
            generator.solve(
                sig(2, 2),
                self._candidates(),
                2,
                4,
                0,
                True,
                clue_words={"a", "ab"},
                hint_mode="exactly-one",
            )
        )

        # Repeating "ab" still matches one distinct clue. The middle bag has
        # both "a" and "ab" and is rejected; the final bag only has "a".
        self.assertEqual(actual, [("ab", "ab"), ("a", "a", "b", "b")])

    def test_required_words_can_satisfy_clue_before_residual_search(self) -> None:
        actual = list(
            generator.solve(
                sig(2, 2),
                self._candidates(),
                2,
                4,
                1,
                True,
                clue_words={"required"},
                initial_clue_words={"required"},
            )
        )
        self.assertEqual(actual, [("ab", "ab")])

    def test_uncued_bounded_prefix_remains_unchanged(self) -> None:
        expected = [("ab", "ab"), ("a", "b", "ab")]
        self.assertEqual(
            list(generator.solve(sig(2, 2), self._candidates(), 2, 4, 2, True)),
            expected,
        )

    def test_invalid_hint_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            list(
                generator.solve(
                    sig(2, 2),
                    self._candidates(),
                    2,
                    4,
                    0,
                    True,
                    clue_words={"a"},
                    hint_mode="bogus",
                )
            )


if __name__ == "__main__":
    unittest.main()
