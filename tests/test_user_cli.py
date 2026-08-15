from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import anagram_solver as solver


class UserCliTests(unittest.TestCase):
    def _args(self, *extra: str) -> argparse.Namespace:
        args = solver.build_parser().parse_args(["abcdef", *extra])
        solver._validate_args(args)
        return args

    def test_default_generation_is_exhaustive(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertIn("--all-results", cmd)
        self.assertNotIn("--max-results", cmd)
        self.assertEqual(cmd[cmd.index("--min-zipf") + 1], "2.7")

    def test_quick_generation_is_explicitly_capped(self) -> None:
        args = self._args("--quick")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        self.assertNotIn("--all-results", cmd)
        self.assertEqual(cmd[cmd.index("--max-results") + 1], "100000")

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

    def test_parse_results_returns_top_per_word_count(self) -> None:
        text = """\
=== 3-WORD RERANKED CANDIDATES (2 total; 2 deep-analyzed) ===
      1. FINAL= 88.00 PRE= 70.00 GRAM=0.900 STRUCT=0.900 VAL=0.90 COV=1.00 KIND=clause      WN=1.00 LEX=0.900 FAM=0.900 HINT=0.000 COLL=0.20 PHRASE=0.40 PBONUS=2.00 knowledge is power  [CANON=is knowledge power; OLDPRE=1.00; OLDRANK=1]
      2. FINAL= 70.00 PRE= 60.00 GRAM=0.800 STRUCT=0.800 VAL=0.80 COV=1.00 KIND=clause      WN=1.00 LEX=0.800 FAM=0.800 HINT=0.000 COLL=0.10 PHRASE=0.10 PBONUS=0.50 power is knowledge  [CANON=is knowledge power; OLDPRE=1.00; OLDRANK=2]

=== 4-WORD RERANKED CANDIDATES (1 total; 1 deep-analyzed) ===
      1. FINAL= 91.25 PRE= 72.00 GRAM=0.920 STRUCT=0.950 VAL=0.90 COV=1.00 KIND=clause      WN=1.00 LEX=0.900 FAM=0.900 HINT=0.500 COLL=0.30 PHRASE=0.75 PBONUS=3.75 these hips dont lie  [CANON=dont hips lie these; OLDPRE=1.00; OLDRANK=1]
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reranked.txt"
            path.write_text(text, encoding="utf-8")
            results = solver.parse_results(path, top=1)

        self.assertEqual(
            [(r.word_count, r.rank, r.phrase) for r in results],
            [
                (3, 1, "knowledge is power"),
                (4, 1, "these hips dont lie"),
            ],
        )

    def test_run_key_changes_when_generation_constraints_change(self) -> None:
        plain = self._args()
        hinted = self._args("--hint", "dont")
        self.assertNotEqual(solver._run_key(plain), solver._run_key(hinted))


if __name__ == "__main__":
    unittest.main()
