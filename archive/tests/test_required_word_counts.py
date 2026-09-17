from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anagram_solver as solver


class RequiredWordCountTests(unittest.TestCase):
    def _args(self, target: str, *extra: str):
        args = solver.build_parser().parse_args([target, *extra])
        solver._validate_args(args)
        return args

    def _generator_limits(self, target: str, *extra: str) -> tuple[int, int]:
        args = self._args(target, *extra)
        with tempfile.TemporaryDirectory() as tmp:
            cmd = solver.build_generator_command(args, Path(tmp) / "candidates.txt")
        return (
            int(cmd[cmd.index("--min-words") + 1]),
            int(cmd[cmd.index("--max-words") + 1]),
        )

    def test_exact_word_count_is_total_including_required_word(self) -> None:
        self.assertEqual(
            self._generator_limits(
                "ODITIHNSLSHEEEPT", "--words", "4", "--require", "dont"
            ),
            (3, 3),
        )
        self.assertEqual(
            self._generator_limits(
                "AHCWSOPSIO", "--words", "2", "--require", "picasso"
            ),
            (1, 1),
        )

    def test_required_phrase_consumes_multiple_total_word_slots(self) -> None:
        self.assertEqual(
            self._generator_limits(
                "alphabetagamma", "--words", "4", "--require", "alpha beta"
            ),
            (2, 2),
        )

    def test_word_range_is_total_including_required_words(self) -> None:
        self.assertEqual(
            self._generator_limits(
                "alphabetagamma",
                "--min-words", "3",
                "--max-words", "5",
                "--require", "alpha",
            ),
            (2, 4),
        )

    def test_required_words_cannot_exceed_total_maximum(self) -> None:
        args = solver.build_parser().parse_args(
            ["alphabetagamma", "--words", "1", "--require", "alpha beta"]
        )
        with self.assertRaisesRegex(SystemExit, "exceeding --max-words"):
            solver._validate_args(args)

    def test_zero_residual_ranked_runs_are_rejected_explicitly(self) -> None:
        for argv in (
            ["alpha", "--words", "1", "--require", "alpha"],
            ["alpha", "--words", "2", "--require", "alpha"],
        ):
            with self.subTest(argv=argv):
                args = solver.build_parser().parse_args(argv)
                with self.assertRaisesRegex(SystemExit, "zero-residual answers"):
                    solver._validate_args(args)

        args = solver.build_parser().parse_args(
            ["alphabeta", "--words", "1", "--require", "alpha"]
        )
        with self.assertRaisesRegex(SystemExit, "no residual word slots"):
            solver._validate_args(args)


if __name__ == "__main__":
    unittest.main()
