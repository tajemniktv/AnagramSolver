from __future__ import annotations

import unittest

import anagram_generate as generator


def sig(a=0, b=0, c=0):
    return (a, b, c) + (0,) * 23


class SearchOptimizationTests(unittest.TestCase):
    def test_historical_exhaustive_order_is_preserved(self) -> None:
        candidates = [
            generator.Candidate("ab", sig(1, 1), 2, 5.0),
            generator.Candidate("ac", sig(1, 0, 1), 2, 4.9),
            generator.Candidate("bc", sig(0, 1, 1), 2, 4.8),
            generator.Candidate("abc", sig(1, 1, 1), 3, 4.7),
        ]
        remaining = sig(2, 2, 2)
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 3, 0, True)),
            [("abc", "abc"), ("ab", "ac", "bc")],
        )
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 3, 0, False)),
            [("ab", "ac", "bc")],
        )

    def test_historical_bounded_prefix_is_preserved(self) -> None:
        candidates = [
            generator.Candidate("a", sig(1), 1, 5.0),
            generator.Candidate("b", sig(0, 1), 1, 4.9),
            generator.Candidate("ab", sig(1, 1), 2, 4.8),
        ]
        remaining = sig(2, 2)
        expected = [("ab", "ab"), ("a", "b", "ab"), ("a", "a", "b", "b")]
        self.assertEqual(
            list(generator.solve(remaining, candidates, 2, 4, 0, True)),
            expected,
        )
        for limit in (1, 2, 3):
            with self.subTest(limit=limit):
                self.assertEqual(
                    list(generator.solve(remaining, candidates, 2, 4, limit, True)),
                    expected[:limit],
                )
