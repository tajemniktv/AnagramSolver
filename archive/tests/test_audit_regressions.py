from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_generate as generator
import anagram_solver as solver
from anagram_user_lexicon import UserLexicon


class AuditRegressionTests(unittest.TestCase):
    def test_invalid_frontend_input_does_not_provision_data(self):
        for argv in (["123!"], ["abc", "--min-zipf", "nan"],
                     ["abc", "--min-zipf", "inf"]):
            with self.subTest(argv=argv), patch.object(solver, "ensure_user_lexicon") as provision:
                with self.assertRaises(SystemExit):
                    solver.main(argv)
                provision.assert_not_called()

    def test_generator_rejects_unsatisfiable_clues_and_required_exclusions(self):
        cases = [
            ["ab", "--contains-any", "xyz"],
            ["ab", "--contains-any", "a", "--exclude", "a"],
            ["ab", "--require", "a", "--exclude", "a"],
            ["ab", "--require", "a", "--exclude-regex", "^a$"],
            ["ab", "--require", "a", "--forbid-chars", "a"],
            ["ab", "--min-zipf", "nan"],
        ]
        for argv in cases:
            with (self.subTest(argv=argv), patch.object(sys, "argv", ["generator", *argv]),
                  contextlib.redirect_stderr(io.StringIO()),
                  patch.object(generator, "get_dictionary") as provision):
                with self.assertRaises(SystemExit):
                    generator.main()
                provision.assert_not_called()

    def test_ranking_reads_private_export_and_preserves_cache_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run = root / "run"
                run.mkdir()
                generator.write_full_export(run / "candidates.txt", [], True,
                    search={"generated": 1, "truncated": False})
                shared = run / "reranked.txt"
                shared.write_text("previous", encoding="utf-8")
                exports = []

                def fake_run(cmd, *, verbose):
                    export = Path(cmd[cmd.index("--export") + 1])
                    exports.append(export)
                    export.write_text("own results", encoding="utf-8")
                    if fail:
                        raise SystemExit("failed")
                    return subprocess.CompletedProcess(cmd, 0)

                def parse(path, top):
                    self.assertNotEqual(path, shared)
                    self.assertEqual(path.read_text(encoding="utf-8"), "own results")
                    self.assertEqual(shared.read_text(encoding="utf-8"), "previous")
                    return []

                with (patch.object(solver, "ensure_user_lexicon", return_value=UserLexicon(root, (), "token")),
                      patch.object(solver, "_run_key", return_value="run"),
                      patch.object(solver, "_run", side_effect=fake_run),
                      patch.object(solver, "parse_results", side_effect=parse),
                      contextlib.redirect_stdout(io.StringIO())):
                    if fail:
                        with self.assertRaises(SystemExit):
                            solver.main(["abc", "--work-root", tmp])
                    else:
                        self.assertEqual(solver.main(["abc", "--work-root", tmp]), 0)
                self.assertEqual(shared.read_text(encoding="utf-8"), "previous" if fail else "own results")
                self.assertTrue(exports)
                self.assertTrue(all(not path.exists() for path in exports))
