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

        # One result slot is reserved per requested bucket. The 3-word bucket is
        # empty, and bounded normal search deliberately does not rerun the 2-word
        # beam merely to fill the global cap.
        self.assertEqual(guided, [("ab", "ab")])

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

        self.assertEqual(guided, [("ab", "ab")])
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
