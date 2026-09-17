"""Real subprocess tests; CI provisions runtime corpora before enabling these."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from anagram_paths import PROJECT_DIR
from anagram_suite import case_by_id, target_for_case


@unittest.skipUnless(
    os.environ.get("ANAGRAM_INTEGRATION") == "1",
    "set ANAGRAM_INTEGRATION=1 after provisioning corpora",
)
class CliIntegrationTests(unittest.TestCase):
    def test_empty_vocabulary_is_a_successful_no_match_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            command = [
                sys.executable,
                str(PROJECT_DIR / "anagram_solver.py"),
                "zzzz",
                "--min-zipf",
                "9",
                "--work-root",
                tmp,
            ]
            completed = subprocess.run(
                [*command, "--json"],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            value = json.loads(completed.stdout)
            self.assertEqual(value["results"], [])
            self.assertEqual(value["search"]["vocabulary_size"], 0)
            self.assertTrue(value["search"]["ranking_skipped"])
            self.assertEqual(list(Path(tmp).rglob("reranked.txt")), [])
            human = subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=True
            )
            self.assertIn("No usable words remain", human.stdout)

    def test_concurrent_cold_runs_warm_reuse_and_corruption_recovery(self):
        case = case_by_id("knowledge_power")
        answer = str(case["answer"])
        with tempfile.TemporaryDirectory() as tmp:
            base = [
                sys.executable,
                str(PROJECT_DIR / "anagram_solver.py"),
                target_for_case(case),
                "--words",
                str(len(answer.split())),
                "--hint",
                answer.split()[0],
                "--workers",
                "1",
                "--quick",
                "--json",
                "--work-root",
                tmp,
            ]
            processes = []
            try:
                for _ in range(2):
                    processes.append(
                        subprocess.Popen(
                            base,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                        )
                    )
                values = []
                for process in processes:
                    stdout, stderr = process.communicate(timeout=120)
                    self.assertEqual(process.returncode, 0, stderr)
                    values.append(json.loads(stdout))
                self.assertEqual(values[0]["results"], values[1]["results"])
                self.assertEqual(values[0]["results"][0]["phrase"], answer)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.communicate()

            def run(*options):
                completed = subprocess.run(
                    [*base, *options],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True,
                )
                return json.loads(completed.stdout)

            warm = run("--top", "1")
            self.assertTrue(warm["search"]["ranking_cached"])
            self.assertEqual(len(warm["results"]), 1)
            rebuilt = run("--rebuild")
            self.assertFalse(rebuilt["search"]["ranking_cached"])
            self.assertTrue(run("--top", "1")["search"]["ranking_cached"])
            cache_files = list(Path(tmp).rglob("ranking-*.json"))
            self.assertEqual(len(cache_files), 1)
            cache_files[0].write_text("interrupted JSON", encoding="utf-8")
            recovered = run()
            self.assertFalse(recovered["search"]["ranking_cached"])
            self.assertEqual(recovered["results"], values[0]["results"])
            changed = run("--order-candidates", "8")
            self.assertFalse(changed["search"]["ranking_cached"])
            self.assertEqual(len(list(Path(tmp).rglob("ranking-*.json"))), 2)
            candidate = next(Path(tmp).rglob("candidates.txt"))
            candidate.write_text("interrupted candidate export", encoding="utf-8")
            repaired = run()
            self.assertEqual(repaired["results"], values[0]["results"])
            self.assertEqual(list(Path(tmp).rglob("*.tmp")), [])
