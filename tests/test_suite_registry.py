from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anagram_suite import (
    DEFAULT_CASES,
    ORDERING_GATE,
    PERFORMANCE_PROBE,
    REGISTRY_SCHEMA,
    SMOKE_CASES,
    case_by_id,
    load_cases,
    validate_registry,
)


class SuiteRegistryTests(unittest.TestCase):
    def test_default_registry_is_internally_consistent(self) -> None:
        self.assertEqual(validate_registry(), ())

    def test_catalog_schema_and_profiles_are_explicit(self) -> None:
        payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], REGISTRY_SCHEMA)
        self.assertEqual(
            set(payload["profiles"]),
            {"normal_user_cli", "ordering_gate", "performance_probe"},
        )

    def test_shared_ci_case_ids_resolve(self) -> None:
        referenced = {ORDERING_GATE.cross_bag_case_id}
        referenced.update(case_id for case_id, _ in ORDERING_GATE.target_max_ranks)
        for case_id in referenced:
            with self.subTest(case_id=case_id):
                self.assertEqual(case_by_id(case_id)["id"], case_id)

    def test_smoke_cases_have_unique_ids_and_targets(self) -> None:
        self.assertEqual(len({case.id for case in SMOKE_CASES}), len(SMOKE_CASES))
        self.assertEqual(
            len({case.target for case in SMOKE_CASES}),
            len(SMOKE_CASES),
        )

    def test_smoke_case_link_to_benchmark_uses_same_letters(self) -> None:
        smoke = next(case for case in SMOKE_CASES if case.id == "shakira_control")
        benchmark = case_by_id("shakira_control")
        target_letters = sorted(ch.lower() for ch in smoke.target if ch.isalpha())
        answer_letters = sorted(
            ch.lower() for ch in str(benchmark["answer"]) if ch.isalpha()
        )
        self.assertEqual(target_letters, answer_letters)

    def test_performance_workload_is_stable_and_nonempty(self) -> None:
        self.assertTrue(PERFORMANCE_PROBE.frame_words)
        self.assertTrue(PERFORMANCE_PROBE.function_words)
        self.assertTrue(PERFORMANCE_PROBE.order_bags)
        self.assertTrue(all(bag for bag in PERFORMANCE_PROBE.order_bags))

    def test_loader_rejects_duplicate_ids(self) -> None:
        payload = {
            "cases": [
                {"id": "same", "answer": "one two"},
                {"id": "same", "answer": "two one"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate benchmark case id"):
                load_cases(path)

    def test_loader_rejects_unknown_selected_id(self) -> None:
        with self.assertRaisesRegex(KeyError, "definitely_missing"):
            load_cases(DEFAULT_CASES, {"definitely_missing"})


if __name__ == "__main__":
    unittest.main()
