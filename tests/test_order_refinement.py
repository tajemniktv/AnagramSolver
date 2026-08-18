from __future__ import annotations

import unittest

from anagram_order_refinement import (
    augment_seed_pool,
    refine_order,
    refine_seed_pool,
    window_neighbors,
)


class OrderRefinementTests(unittest.TestCase):
    def test_window_neighbors_are_unique_preserve_bag_and_deterministic(self) -> None:
        self.assertEqual(tuple(window_neighbors(("a",), min_window=3, max_window=5)), ())
        self.assertEqual(tuple(window_neighbors(("a", "b"), min_window=3, max_window=5)), ())

        seed = ("a", "b", "c", "d", "e")
        neighbors = tuple(window_neighbors(seed, min_window=3, max_window=4))
        repeated = tuple(window_neighbors(seed, min_window=3, max_window=4))
        self.assertEqual(neighbors, repeated)
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

    def test_refinement_uses_lexical_tie_break_only_among_improvements(self) -> None:
        seed = ("c", "b", "a")
        neighbors = tuple(window_neighbors(seed, min_window=3, max_window=3))
        expected = min(neighbors)

        def scorer(order: tuple[str, ...]) -> float:
            return 0.0 if order == seed else 1.0

        result = refine_order(
            seed,
            scorer,
            min_window=3,
            max_window=3,
            max_rounds=1,
            epsilon=1e-9,
        )
        self.assertEqual(result.order, expected)
        self.assertEqual(result.score, 1.0)

    def test_parameter_guards_and_initial_score_path(self) -> None:
        seed = ("a", "b", "c")
        scorer_calls = 0

        def scorer(_order: tuple[str, ...]) -> float:
            nonlocal scorer_calls
            scorer_calls += 1
            return 0.5

        invalid_kwargs = (
            {"min_window": 1},
            {"min_window": 3, "max_window": 2},
            {"max_rounds": -1},
            {"max_evaluations": 0},
            {"epsilon": -1e-6},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    refine_order(seed, scorer, **kwargs)

        initial = refine_order(seed, scorer, initial_score=0.5, max_rounds=0)
        self.assertEqual(initial.evaluated, 0)
        self.assertEqual(scorer_calls, 0)
        self.assertEqual(initial.order, seed)
        self.assertEqual(initial.score, 0.5)

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

    def test_augmentation_keeps_all_seeds_beyond_refinement_limit(self) -> None:
        target = ("a", "b", "c", "d")
        seeds = (
            ("a", "c", "b", "d"),
            ("d", "c", "b", "a"),
            ("b", "a", "d", "c"),
        )

        def scorer(order: tuple[str, ...]) -> float:
            return sum(left == right for left, right in zip(order, target))

        result = augment_seed_pool(
            seeds,
            scorer,
            seed_limit=1,
            max_window=4,
        )
        orders = {candidate.order for candidate in result.candidates}
        self.assertTrue(set(seeds).issubset(orders))
        self.assertIn(target, orders)
        self.assertGreater(result.evaluated, len(seeds))
        self.assertGreaterEqual(result.improved_seeds, 1)

    def test_augmentation_budget_includes_seed_score(self) -> None:
        seeds = (
            ("a", "b", "c"),
            ("b", "a", "c"),
            ("c", "b", "a"),
        )
        calls = 0

        def scorer(order: tuple[str, ...]) -> float:
            nonlocal calls
            calls += 1
            return float(order == seeds[0])

        result = augment_seed_pool(
            seeds,
            scorer,
            seed_limit=1,
            max_evaluations_per_seed=1,
        )
        self.assertEqual(calls, len(seeds))
        self.assertEqual(result.evaluated, len(seeds))
        self.assertEqual(result.improved_seeds, 0)
        self.assertEqual(
            {candidate.order for candidate in result.candidates},
            set(seeds),
        )

        with self.assertRaisesRegex(ValueError, "max_evaluations_per_seed"):
            augment_seed_pool(seeds, scorer, max_evaluations_per_seed=0)


if __name__ == "__main__":
    unittest.main()
