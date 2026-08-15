from __future__ import annotations

import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_solver as solver


class UserCliCacheSafetyTests(unittest.TestCase):
    def _args(self, *extra: str):
        args = solver.build_parser().parse_args(["ODITIHNSLSHEEEPT", *extra])
        solver._validate_args(args)
        return args

    def test_run_key_normalizes_word_constraints(self) -> None:
        lower = self._args(
            "--hint", "dont,phone",
            "--exclude", "lois",
            "--require", "hips",
        )
        equivalent = self._args(
            "--hint", "DONT,Phône,dont",
            "--exclude", "LOIS",
            "--require", "HIPS",
        )
        self.assertEqual(solver._run_key(lower), solver._run_key(equivalent))

    def test_same_key_generations_use_independent_temporary_exports(self) -> None:
        args = self._args()
        with tempfile.TemporaryDirectory() as tmp:
            candidates = Path(tmp) / "candidates.txt"
            barrier = threading.Barrier(2)
            export_paths: list[Path] = []
            errors: list[BaseException] = []
            export_lock = threading.Lock()

            def fake_run(cmd, *, verbose: bool):
                export = Path(cmd[cmd.index("--export") + 1])
                with export_lock:
                    export_paths.append(export)
                export.write_text(threading.current_thread().name, encoding="utf-8")
                barrier.wait(timeout=5)
                return subprocess.CompletedProcess(cmd, 0)

            def generate() -> None:
                try:
                    solver._generate_candidates(args, candidates)
                except BaseException as exc:  # capture failures from worker threads
                    errors.append(exc)

            with patch.object(solver, "_run", side_effect=fake_run):
                threads = [
                    threading.Thread(target=generate, name=f"generator-{index}")
                    for index in range(2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertTrue(all(not thread.is_alive() for thread in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(export_paths), 2)
            self.assertNotEqual(export_paths[0], export_paths[1])
            self.assertTrue(candidates.is_file())
            self.assertTrue(all(not path.exists() for path in export_paths))


if __name__ == "__main__":
    unittest.main()
