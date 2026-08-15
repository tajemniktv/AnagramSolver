from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_solver as solver


class UserCliTests(unittest.TestCase):
    def _args(self, *extra: str) -> argparse.Namespace:
        args = solver.build_parser().parse_args(["abcdef", *extra])
        solver._validate_args(args)
        return args

    def _target_args(self, target: str, *extra: str) -> argparse.Namespace:
        args = solver.build_parser().parse_args([target, *extra])
        solver._validate_args(args)
        return args

    def _assert_validation_error(self, *extra: str) -> str:
        args = solver.build_parser().parse_args(["abcdef", *extra])
        with self.assertRaises(SystemExit) as caught:
            solver._validate_args(args)
        return str(caught.exception)

    def test_default_generation_is_balanced_and_bounded(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertNotIn("--all-results", cmd)
        self.assertEqual(
            cmd[cmd.index("--max-results") + 1],
            str(solver.BALANCED_MAX_RESULTS),
        )
        self.assertEqual(cmd[cmd.index("--min-zipf") + 1], "2.7")
        self.assertEqual(solver._generation_mode(args), "balanced")
        self.assertEqual(solver._generation_cap(args), solver.BALANCED_MAX_RESULTS)

    def test_generator_export_requests_reranker_components(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertIn("--show-components", cmd)

    def test_quick_generation_uses_smaller_cap(self) -> None:
        args = self._args("--quick")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertNotIn("--all-results", cmd)
        self.assertEqual(
            cmd[cmd.index("--max-results") + 1],
            str(solver.QUICK_MAX_RESULTS),
        )
        self.assertLess(solver.QUICK_MAX_RESULTS, solver.BALANCED_MAX_RESULTS)
        self.assertEqual(solver._generation_mode(args), "quick")
        self.assertEqual(solver._generation_cap(args), solver.QUICK_MAX_RESULTS)

    def test_exhaustive_generation_is_explicit(self) -> None:
        args = self._args("--exhaustive")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertIn("--all-results", cmd)
        self.assertNotIn("--max-results", cmd)
        self.assertIsNone(solver._generation_cap(args))
        self.assertEqual(solver._generation_mode(args), "exhaustive")

    def test_quick_and_exhaustive_are_mutually_exclusive(self) -> None:
        with self.assertRaises(SystemExit):
            solver.build_parser().parse_args(["abcdef", "--quick", "--exhaustive"])

    def test_hints_excludes_require_and_word_count_are_forwarded(self) -> None:
        args = self._args(
            "--words", "4",
            "--hint", "dont,phone",
            "--exclude", "hit",
            "--require", "hips",
        )
        self.assertEqual((args.min_words, args.max_words), (4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertEqual(cmd[cmd.index("--contains-any") + 1], "dont,phone")
        self.assertEqual(cmd[cmd.index("--exclude") + 1], "hit")
        self.assertEqual(cmd[cmd.index("--require") + 1], "hips")

    def test_reranker_profile_workers_and_phrase_db_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phrase_db = root / "phrases.sqlite"
            phrase_db.touch()
            candidates = root / "candidates.txt"
            output = root / "reranked.txt"

            normal = self._args("--workers", "8", "--phrase-db", str(phrase_db))
            normal_cmd = solver.build_reranker_command(normal, candidates, output)
            self.assertEqual(normal_cmd[normal_cmd.index("--deep-per-group") + 1], "5000")
            self.assertEqual(normal_cmd[normal_cmd.index("--workers") + 1], "8")
            self.assertEqual(
                normal_cmd[normal_cmd.index("--phrase-db") + 1],
                str(phrase_db.resolve()),
            )

            quick = self._args("--quick")
            quick_cmd = solver.build_reranker_command(quick, candidates, output)
            self.assertEqual(quick_cmd[quick_cmd.index("--deep-per-group") + 1], "2000")
            self.assertNotIn("--phrase-db", quick_cmd)

    def test_invalid_cli_values_are_rejected(self) -> None:
        cases = [
            (("--words", "0"), "--words"),
            (("--min-words", "4", "--max-words", "3"), "--min-words/--max-words"),
            (("--min-word-len", "0"), "--min-word-len"),
            (("--workers", "-1"), "--workers"),
            (("--top", "0"), "--top"),
            (("--min-zipf", "-0.1"), "--min-zipf"),
            (("--phrase-db", "definitely-not-a-real-phrase-db.sqlite"), "--phrase-db"),
            (("--json", "--verbose"), "--json"),
        ]
        for argv, expected in cases:
            with self.subTest(argv=argv):
                self.assertIn(expected, self._assert_validation_error(*argv))

    def test_parse_results_handles_limits_separators_and_malformed_lines(self) -> None:
        text = """\
=== 3-WORD RERANKED CANDIDATES (3 total; 3 deep-analyzed) ===
      1. FINAL= 88.00 PRE= 70.00 GRAM=0.900 STRUCT=0.900 VAL=0.90 COV=1.00 KIND=clause      WN=1.00 LEX=0.900 FAM=0.900 HINT=0.000 COLL=0.20 PHRASE=0.40 PBONUS=2.00 knowledge is power  [CANON=is knowledge power; OLDPRE=1.00; OLDRANK=1]
not a reranker result
      2. FINAL= 70.00 PRE= 60.00 GRAM=0.800 STRUCT=0.800 VAL=0.80 COV=1.00 KIND=clause      WN=1.00 LEX=0.800 FAM=0.800 HINT=0.000 COLL=0.10 PHRASE=0.10 PBONUS=0.50 power is knowledge  [CANON=is knowledge power; OLDPRE=1.00; OLDRANK=2]
      3. FINAL= 60.00 PRE= 50.00 GRAM=0.700 STRUCT=0.700 VAL=0.70 COV=1.00 KIND=fragment    WN=1.00 LEX=0.700 FAM=0.700 HINT=0.000 COLL=0.05 PHRASE=0.05 PBONUS=0.25 knowledge power is  [CANON=is knowledge power; OLDPRE=1.00; OLDRANK=3]
--- NOT DEEP-ANALYZED; PRE ONLY ---
      4. PRE= 55.00 GPOT=0.500 WN=1.00 LEX=0.500 FAM=0.500 HINT=0.000 ignored pre only

=== 4-WORD RERANKED CANDIDATES (1 total; 1 deep-analyzed) ===
      1. FINAL= 91.25 PRE= 72.00 GRAM=0.920 STRUCT=0.950 VAL=0.90 COV=1.00 KIND=clause      WN=1.00 LEX=0.900 FAM=0.900 HINT=0.500 COLL=0.30 PHRASE=0.75 PBONUS=3.75 these hips dont lie  [CANON=dont hips lie these; OLDPRE=1.00; OLDRANK=1]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reranked.txt"
            path.write_text(text, encoding="utf-8")
            results = solver.parse_results(path, top=2)

        self.assertEqual(
            [(r.word_count, r.rank, r.phrase) for r in results],
            [
                (3, 1, "knowledge is power"),
                (3, 2, "power is knowledge"),
                (4, 1, "these hips dont lie"),
            ],
        )

    def test_run_key_normalizes_equivalent_target_text(self) -> None:
        spaced = self._target_args("Tóm Marvolo Riddle")
        compact = self._target_args("tom-marvolo-riddle")
        letters = self._target_args("TOMMARVOLORIDDLE")
        self.assertEqual(solver._run_key(spaced), solver._run_key(compact))
        self.assertEqual(solver._run_key(spaced), solver._run_key(letters))

    def test_run_key_changes_for_generation_constraints(self) -> None:
        base = self._args()
        base_key = solver._run_key(base)
        variants = [
            self._args("--hint", "dont"),
            self._args("--quick"),
            self._args("--exhaustive"),
            self._args("--min-zipf", "1.5"),
            self._args("--min-word-len", "3"),
            self._args("--min-words", "3"),
            self._args("--max-words", "5"),
            self._args("--exclude", "dont"),
            self._args("--require", "hips"),
        ]
        for variant in variants:
            with self.subTest(variant=vars(variant)):
                self.assertNotEqual(base_key, solver._run_key(variant))

    def test_run_key_ignores_non_generation_options(self) -> None:
        base = self._args()
        base_key = solver._run_key(base)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phrase_db = root / "phrases.sqlite"
            phrase_db.touch()
            variants = [
                self._args("--top", "25"),
                self._args("--workers", "4"),
                self._args("--phrase-db", str(phrase_db)),
                self._args("--json"),
                self._args("--work-root", str(root / "other")),
            ]
            for variant in variants:
                with self.subTest(variant=vars(variant)):
                    self.assertEqual(base_key, solver._run_key(variant))

    def test_verbose_subprocess_inherits_terminal_streams(self) -> None:
        completed = __import__("subprocess").CompletedProcess(["python"], 0)
        with patch.object(solver.subprocess, "run", return_value=completed) as run:
            solver._run(["python", "child.py"], verbose=True)
        _, kwargs = run.call_args
        self.assertTrue(kwargs["text"])
        self.assertNotIn("capture_output", kwargs)

    def test_candidate_cache_is_published_only_after_success(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            candidates = Path(tmp) / "candidates.txt"
            candidates.write_text("old-cache\n", encoding="utf-8")

            def fail_run(cmd: list[str], *, verbose: bool):
                export = Path(cmd[cmd.index("--export") + 1])
                export.write_text("partial\n", encoding="utf-8")
                raise SystemExit("generator failed")

            with patch.object(solver, "_run", side_effect=fail_run):
                with self.assertRaisesRegex(SystemExit, "generator failed"):
                    solver._generate_candidates(args, candidates)

            self.assertEqual(candidates.read_text(encoding="utf-8"), "old-cache\n")
            self.assertEqual(list(candidates.parent.glob(".candidates.txt.*.tmp")), [])

    def test_successful_generation_atomically_replaces_candidate_cache(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            candidates = Path(tmp) / "candidates.txt"
            candidates.write_text("old-cache\n", encoding="utf-8")

            def succeed_run(cmd: list[str], *, verbose: bool):
                export = Path(cmd[cmd.index("--export") + 1])
                export.write_text("fresh-cache\n", encoding="utf-8")
                return __import__("subprocess").CompletedProcess(cmd, 0)

            with patch.object(solver, "_run", side_effect=succeed_run):
                solver._generate_candidates(args, candidates)

            self.assertEqual(candidates.read_text(encoding="utf-8"), "fresh-cache\n")
            self.assertEqual(list(candidates.parent.glob(".candidates.txt.*.tmp")), [])
