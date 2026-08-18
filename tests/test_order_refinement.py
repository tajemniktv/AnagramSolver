from __future__ import annotations

import unittest

from anagram_order_refinement import refine_order, refine_seed_pool, window_neighbors


class OrderRefinementTests(unittest.TestCase):
    def test_window_neighbors_are_unique_and_preserve_bag(self) -> None:
        seed = ("a", "b", "c", "d", "e")
        neighbors = tuple(window_neighbors(seed, min_window=3, max_window=4))
        self.assertEqual(len(neighbors), len(set(neighbors)))
        self.assertTrue(neighbors)
        self.assertTrue(all(sorted(order) == sorted(seed) for order in neighbors))
        self.assertNotIn(seed, neighbors)

    def test_refinement_escapes_seed_with_full_order_scorer(self) -> None:
        target = ("the", "quick", "brown", "fox", "jumps", "today")
        seed = ("the", "fox", "brown", "quick", "jumps", "today")

        def scorer(order: tuple[str, ...]) -> float:
            return sum(left == right for left, right in zip(order, target)) / len(target)

        result = refine_order(seed, scorer, max_window=5, max_evaluations=800)
        self.assertEqual(result.order, target)
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.improved)
        self.assertGreater(result.evaluated, 1)

    def test_budget_and_no_degradation_are_enforced(self) -> None:
        seed = ("a", "b", "c", "d", "e", "f")
        calls = 0

        def scorer(order: tuple[str, ...]) -> float:
            nonlocal calls
            calls += 1
            return 1.0 if order == seed else 0.0

        result = refine_order(seed, scorer, max_evaluations=17, max_rounds=5)
        self.assertEqual(result.order, seed)
        self.assertFalse(result.improved)
        self.assertLessEqual(result.evaluated, 17)
        self.assertEqual(calls, result.evaluated)

    def test_seed_pool_is_deterministic_and_deduplicated(self) -> None:
        target = ("a", "b", "c", "d")

        def scorer(order: tuple[str, ...]) -> float:
            return sum(left == right for left, right in zip(order, target))

        seeds = (
            ("a", "c", "b", "d"),
            ("a", "b", "d", "c"),
            ("a", "c", "b", "d"),
        )
        first = refine_seed_pool(seeds, scorer, max_window=4)
        second = refine_seed_pool(seeds, scorer, max_window=4)
        self.assertEqual(first, second)
        self.assertEqual(first[0].order, target)
        self.assertEqual(len({result.order for result in first}), len(first))


if __name__ == "__main__":
    unittest.main()
