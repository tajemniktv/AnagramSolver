from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_generate as generator
import anagram_solver as solver


class SearchFeedbackTests(unittest.TestCase):
    def test_custom_budget_is_validated_and_part_of_cache_identity(self):
        parser = solver.build_parser()
        small = parser.parse_args(["abc", "--max-results", "2"])
        larger = parser.parse_args(["abc", "--max-results", "3"])
        solver._validate_args(small)
        solver._validate_args(larger)
        self.assertEqual(solver._generation_cap(small), 2)
        self.assertNotEqual(solver._run_key(small), solver._run_key(larger))
        for options in (("--max-results", "0"), ("--max-results", "2", "--exhaustive")):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                solver._validate_args(parser.parse_args(["abc", *options]))

    @staticmethod
    def candidates():
        return [
            generator.Candidate(w, generator.counts(w), len(w), 5.0)
            for w in ("a", "b", "ab")
        ]

    def test_cap_equal_to_total_is_not_truncation(self):
        for strategy in ("prefix", "diverse"):
            for cap, truncated in ((2, True), (3, False), (0, False)):
                stats = generator.SearchStats()
                bags = list(
                    generator.search_solutions(
                        generator.counts("aabb"),
                        self.candidates(),
                        2,
                        4,
                        cap,
                        True,
                        stats=stats,
                        strategy=strategy,
                    )
                )
                self.assertEqual(len(bags), cap or 3)
                self.assertEqual(stats.truncated, truncated)
                self.assertEqual(stats.accepted, len(bags))

    def test_diverse_clues_do_not_spend_budget_on_duplicate_bags(self):
        stats = generator.SearchStats()
        bags = list(
            generator.search_solutions(
                generator.counts("aabb"),
                self.candidates(),
                2,
                4,
                3,
                True,
                clue_words={"a", "ab"},
                stats=stats,
                strategy="diverse",
            )
        )
        self.assertEqual(len(bags), 3)
        self.assertEqual(len(set(bags)), 3)
        self.assertFalse(stats.truncated)

    def test_diverse_search_reaches_later_word_counts_with_same_budget(self):
        # Both two-word bags precede the three-word solution historically.
        candidates = [
            generator.Candidate(w, generator.counts(w), len(w), 5.0)
            for w in ("ab", "ba", "a", "b")
        ]

        def run(strategy):
            return list(
                generator.search_solutions(
                    generator.counts("aabb"),
                    candidates,
                    2,
                    3,
                    2,
                    True,
                    strategy=strategy,
                )
            )

        self.assertEqual([len(bag) for bag in run("prefix")], [2, 2])
        self.assertEqual([len(bag) for bag in run("diverse")], [2, 3])

    def test_exhaustive_preserves_order_and_solution_set(self):
        expected = list(
            generator.solve(generator.counts("aabb"), self.candidates(), 2, 4, 0, True)
        )
        actual = list(
            generator.search_solutions(
                generator.counts("aabb"),
                self.candidates(),
                2,
                4,
                0,
                True,
                strategy="diverse",
            )
        )
        self.assertEqual(actual, expected)

    def test_no_matches_skips_ranking_and_returns_structured_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            run.mkdir()
            generator.write_full_export(
                run / "candidates.txt",
                [],
                True,
                search={
                    "generated": 0,
                    "truncated": False,
                    "status": "no_matches_under_constraints",
                },
            )
            output = io.StringIO()
            with (
                patch.object(solver, "ensure_user_lexicon"),
                patch.object(solver, "_run_key", return_value="run"),
                patch.object(solver, "_run") as child,
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(solver.main(["zzzz", "--work-root", tmp, "--json"]), 0)
            child.assert_not_called()
            value = json.loads(output.getvalue())
            self.assertEqual(value["results"], [])
            self.assertEqual(value["search"]["deep_analyzed"], 0)
