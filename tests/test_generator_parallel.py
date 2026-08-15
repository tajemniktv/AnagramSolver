from __future__ import annotations

import unittest
from unittest.mock import patch

import anagram_search_parallel as search


def signature(a: int = 0, b: int = 0, z: int = 0) -> tuple[int, ...]:
    values = [0] * 26
    values[0] = a
    values[1] = b
    values[25] = z
    return tuple(values)


class ParallelSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.candidates = [
            search.SearchCandidate("a", signature(a=1), 1),
            search.SearchCandidate("b", signature(b=1), 1),
            search.SearchCandidate("ab", signature(a=1, b=1), 2),
        ]
        # Non-fitting candidates force the public path to exercise the process
        # pool without changing the tiny reference problem's solutions.
        self.candidates.extend(
            search.SearchCandidate(f"z{index}", signature(z=5 + index), 1)
            for index in range(61)
        )
        self.remaining = signature(a=2, b=2)

    def solve(self, *, workers: int, limit: int = 0, hints=()) -> list[tuple[str, ...]]:
        return list(
            search.solve_parallel(
                self.remaining,
                self.candidates,
                2,
                4,
                limit,
                allow_repeat=True,
                workers=workers,
                required_any=hints,
            )
        )

    def test_parallel_matches_single_worker_order_and_results(self) -> None:
        self.assertEqual(self.solve(workers=2), self.solve(workers=1))

    def test_parallel_preserves_bounded_prefix(self) -> None:
        self.assertEqual(
            self.solve(workers=2, limit=2),
            self.solve(workers=1, limit=2),
        )

    def test_hint_pruning_keeps_exact_matching_population(self) -> None:
        expected = [("ab", "ab"), ("a", "b", "ab")]
        self.assertEqual(self.solve(workers=1, hints={"ab"}), expected)
        self.assertEqual(self.solve(workers=2, hints={"ab"}), expected)

    def test_auto_worker_count_is_bounded(self) -> None:
        with patch.object(search.os, "cpu_count", return_value=16):
            self.assertEqual(search.resolve_worker_count(0), 8)
        with patch.object(search.os, "cpu_count", return_value=1):
            self.assertEqual(search.resolve_worker_count(0), 1)
        self.assertEqual(search.resolve_worker_count(3), 3)
