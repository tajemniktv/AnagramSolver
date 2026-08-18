from __future__ import annotations

import unittest

import anagram_generate as generator
import anagram_user_search as user_search


def sig(a: int = 0, b: int = 0) -> tuple[int, ...]:
    return (a, b) + (0,) * 24


class QualityGuidedGenerationTests(unittest.TestCase):
    def test_quality_search_beats_first_dfs_exact_bag(self) -> None:
        candidates = [
            generator.Candidate("high-a", sig(1, 0), 1, 6.0),
            generator.Candidate("good-ab", sig(1, 1), 2, 5.0),
            generator.Candidate("good-b", sig(0, 1), 1, 5.0),
            generator.Candidate("low-bb", sig(0, 2), 2, 1.0),
        ]
        remaining = sig(1, 2)

        historical = list(
            generator.solve(remaining, candidates, 2, 2, 1, True)
        )
        guided = list(
            user_search.quality_guided_bounded_solve(
                remaining,
                candidates,
                2,
                2,
                1,
                True,
            )
        )

        self.assertEqual(historical, [("high-a", "low-bb")])
        self.assertEqual(guided, [("good-ab", "good-b")])

    def test_multi_view_selection_preserves_collocation_candidate(self) -> None:
        lexical = [
            (10.0, 0, ("common-1",)),
            (9.0, -1, ("common-2",)),
            (8.0, -2, ("common-3",)),
            (7.0, -3, ("common-4",)),
        ]
        pair = [
            (1.0, -0.1, 4.0, -4, ("collocated",)),
            (0.5, -0.2, 9.0, -1, ("common-2",)),
        ]
        anchors = {99: [(3.0, -5, ("rare-anchor",))]}

        selected = user_search._select_multi_view(lexical, pair, anchors, 4)

        self.assertIn(("collocated",), selected)
        self.assertIn(("rare-anchor",), selected)
        self.assertEqual(len(selected), 4)

    def test_view_quotas_leave_room_for_diversity(self) -> None:
        lexical, pair = user_search._view_quotas(10_000)
        self.assertEqual(lexical, 5_500)
        self.assertEqual(pair, 3_500)
        self.assertLess(lexical + pair, 10_000)

    def test_pair_priority_prefers_observed_connectivity(self) -> None:
        candidates = [
            generator.Candidate("alpha", sig(1, 0), 5, 5.0),
            generator.Candidate("beta", sig(0, 1), 4, 5.0),
            generator.Candidate("gamma", sig(0, 1), 5, 5.0),
        ]
        unigrams = generator.UnigramModel(
            counts={"alpha": 1000, "beta": 1000, "gamma": 1000},
            total=3000,
        )
        model = generator.BigramModel(
            unigrams,
            {("alpha", "beta"): 500},
        )

        observed = user_search._pair_priority((0, 1), candidates, model)
        unseen = user_search._pair_priority((0, 2), candidates, model)

        self.assertGreater(observed, unseen)

    def test_balanced_budget_keeps_multiple_word_count_buckets(self) -> None:
        candidates = [
            generator.Candidate("ab", sig(1, 1), 2, 5.0),
            generator.Candidate("a", sig(1, 0), 1, 4.0),
            generator.Candidate("b", sig(0, 1), 1, 4.0),
        ]
        guided = list(
            user_search.quality_guided_bounded_solve(
                sig(2, 2),
                candidates,
                2,
                3,
                2,
                True,
            )
        )

        self.assertEqual(guided, [("ab", "ab"), ("ab", "a", "b")])

    def test_sparse_bucket_does_not_force_expensive_rescan_of_other_bucket(self) -> None:
        candidates = [
            generator.Candidate("ab", sig(1, 1), 2, 5.0),
            generator.Candidate("aa", sig(2, 0), 2, 4.5),
            generator.Candidate("bb", sig(0, 2), 2, 4.5),
        ]
        guided = list(
            user_search.quality_guided_bounded_solve(
                sig(2, 2),
                candidates,
                2,
                3,
                2,
                True,
            )
        )

        self.assertEqual(guided, [("aa", "bb")])

    def test_search_stats_report_exact_closures_not_just_retained_results(self) -> None:
        candidates = [
            generator.Candidate("ab", sig(1, 1), 2, 5.0),
            generator.Candidate("aa", sig(2, 0), 2, 4.5),
            generator.Candidate("bb", sig(0, 2), 2, 4.5),
        ]
        stats = generator.SearchStats()
        guided = list(
            user_search.quality_guided_bounded_solve(
                sig(2, 2),
                candidates,
                2,
                2,
                1,
                True,
                stats=stats,
            )
        )

        self.assertEqual(guided, [("aa", "bb")])
        self.assertEqual(stats.accepted, 1)
        self.assertGreaterEqual(stats.exact_examined, 2)

    def test_no_repeat_is_respected(self) -> None:
        candidates = [generator.Candidate("ab", sig(1, 1), 2, 5.0)]
        self.assertEqual(
            list(
                user_search.quality_guided_bounded_solve(
                    sig(2, 2),
                    candidates,
                    2,
                    2,
                    1,
                    False,
                )
            ),
            [],
        )

    def test_hinted_search_delegates_to_clue_aware_fallback(self) -> None:
        calls: list[dict[str, object]] = []

        def fallback(*args, **kwargs):
            calls.append(kwargs)
            yield ("legacy",)

        wrapped = user_search.make_quality_guided_solve(fallback)
        result = list(
            wrapped(
                sig(1),
                [generator.Candidate("a", sig(1), 1, 5.0)],
                1,
                1,
                1,
                True,
                clue_words={"hint"},
            )
        )

        self.assertEqual(result, [("legacy",)])
        self.assertEqual(calls[0]["clue_words"], {"hint"})

    def test_exhaustive_and_frequencyless_searches_delegate(self) -> None:
        calls = 0

        def fallback(*args, **kwargs):
            nonlocal calls
            calls += 1
            yield ("legacy",)

        wrapped = user_search.make_quality_guided_solve(fallback)
        positive = [generator.Candidate("a", sig(1), 1, 5.0)]
        zero = [generator.Candidate("a", sig(1), 1, 0.0)]

        self.assertEqual(
            list(wrapped(sig(1), positive, 1, 1, 0, True)),
            [("legacy",)],
        )
        self.assertEqual(
            list(wrapped(sig(1), zero, 1, 1, 1, True)),
            [("legacy",)],
        )
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
