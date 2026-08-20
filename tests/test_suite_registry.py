from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anagram_suite import (
    CASE_SUITES,
    DEFAULT_CASES,
    PERFORMANCE_PROBE,
    REGISTRY_SCHEMA,
    case_by_id,
    case_options,
    cases_for,
    load_cases,
    normal_user_case,
    validate_registry,
)


class SuiteRegistryTests(unittest.TestCase):
    @staticmethod
    def _custom_registry() -> dict[str, object]:
        return {
            "schema": REGISTRY_SCHEMA,
            "description": "minimal custom test registry",
            "defaults": {
                "suites": ["all"],
                "normal_user_cli": {
                    "timeout_seconds": 30,
                    "verbose": True,
                    "expect_answer": True,
                },
            },
            "profiles": {
                "ordering_gate": {
                    "min_recall_1": 0.0,
                    "min_recall_10": 0.0,
                    "min_recall_50": 0.0,
                    "min_mrr": 0.0,
                    "min_cross_bag_margin": 0.0,
                    "malformed_bags": [["bad", "bag"]],
                },
                "phrase_ordering": {"order_candidates": 8},
                "performance_probe": {
                    "frame_words": ["runs"],
                    "function_words": ["the"],
                },
            },
            "cases": [
                {
                    "id": "custom_case",
                    "target": "ABC",
                    "answer": "a b c",
                    "category": "test",
                    "ordering": {"cross_bag_reference": True},
                }
            ],
        }

    @staticmethod
    def _write_json(directory: str, payload: object) -> Path:
        path = Path(directory) / "cases.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_default_registry_is_internally_consistent(self) -> None:
        self.assertEqual(validate_registry(), ())

    def test_catalog_schema_profiles_and_all_suite_default_are_explicit(self) -> None:
        payload = json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], REGISTRY_SCHEMA)
        self.assertEqual(payload["defaults"]["suites"], ["all"])
        self.assertEqual(
            set(payload["profiles"]),
            {"ordering_gate", "phrase_ordering", "performance_probe"},
        )

    def test_user_testing_anagrams_is_one_case_in_every_suite(self) -> None:
        for suite in CASE_SUITES:
            with self.subTest(suite=suite):
                ids = {str(case["id"]) for case in cases_for(suite)}
                self.assertIn("user_testing_anagrams", ids)

    def test_core_group_drives_all_core_consumers(self) -> None:
        ordering = {str(case["id"]) for case in cases_for("ordering")}
        phrase = {str(case["id"]) for case in cases_for("phrase_ordering")}
        refinement = {str(case["id"]) for case in cases_for("refinement")}
        ranker = {str(case["id"]) for case in cases_for("feature_ranker")}
        self.assertEqual(ordering, phrase)
        self.assertEqual(ordering, refinement)
        self.assertEqual(ordering, ranker)
        self.assertIn("birds_feather", ordering)
        self.assertNotIn(
            "birds_feather",
            {str(case["id"]) for case in cases_for("normal_user_cli")},
        )

    def test_cli_smokes_are_real_registry_cases(self) -> None:
        ids = {str(case["id"]) for case in cases_for("normal_user_cli")}
        self.assertEqual(
            ids,
            {
                "cli_so_cozy",
                "cli_hi_everyone",
                "cli_proper_name_control",
                "shakira_control",
                "user_testing_anagrams",
            },
        )

    def test_normal_user_case_resolves_case_owned_solver_options(self) -> None:
        run = normal_user_case(case_by_id("user_testing_anagrams"))
        self.assertEqual(run.target, "IAMTESTINGANAGRAMS")
        self.assertEqual(run.expected_phrase, "i am testing anagrams")
        self.assertTrue(run.verbose)
        self.assertIn("--words", run.solver_args)
        self.assertEqual(run.solver_args[run.solver_args.index("--words") + 1], "4")
        self.assertIn("--min-zipf", run.solver_args)
        self.assertIn("--verbose", run.solver_args)

    def test_suite_override_can_remove_common_hint_and_answer_assertion(self) -> None:
        case = case_by_id("shakira_control")
        full = case_options(case, "full")
        cli = normal_user_case(case)
        self.assertEqual(full["hints"], ["dont"])
        self.assertEqual(full["max_results"], 0)
        self.assertNotIn("--hint", cli.solver_args)
        self.assertIsNone(cli.expected_phrase)

    def test_performance_ordering_bags_are_selected_by_case_membership(self) -> None:
        self.assertTrue(PERFORMANCE_PROBE.frame_words)
        self.assertTrue(PERFORMANCE_PROBE.function_words)
        self.assertEqual(
            {str(case["id"]) for case in cases_for("performance")},
            {
                "actions_words",
                "united_stand",
                "user_testing_anagrams",
                "dog_ball",
                "phone_charge",
                "watched_pot",
                "fortune_bold",
                "quiet_focus",
            },
        )

    def test_omitted_suites_defaults_to_every_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, self._custom_registry())
            self.assertEqual(validate_registry(path), ())
            for suite in CASE_SUITES:
                with self.subTest(suite=suite):
                    self.assertEqual(
                        [case["id"] for case in cases_for(suite, path=path)],
                        ["custom_case"],
                    )

    def test_custom_registry_schema_is_validated(self) -> None:
        payload = self._custom_registry()
        payload["schema"] = REGISTRY_SCHEMA + 1
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertEqual(len(errors), 1)
        self.assertIn("test registry schema", errors[0])

    def test_custom_registry_rejects_empty_performance_workload(self) -> None:
        payload = self._custom_registry()
        profiles = payload["profiles"]
        assert isinstance(profiles, dict)
        performance = profiles["performance_probe"]
        assert isinstance(performance, dict)
        performance["frame_words"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertIn("performance probe word workloads must be non-empty", errors)

    def test_custom_registry_rejects_nonfinite_numeric_threshold(self) -> None:
        payload = self._custom_registry()
        profiles = payload["profiles"]
        assert isinstance(profiles, dict)
        ordering = profiles["ordering_gate"]
        assert isinstance(ordering, dict)
        ordering["min_cross_bag_margin"] = float("nan")
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertTrue(any("must be finite" in error for error in errors))

    def test_disabled_cross_bag_reference_is_not_counted(self) -> None:
        payload = self._custom_registry()
        raw_cases = payload["cases"]
        assert isinstance(raw_cases, list)
        case = raw_cases[0]
        assert isinstance(case, dict)
        case["enabled"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertTrue(
            any("exactly one enabled cross_bag_reference" in error for error in errors)
        )

    def test_source_derived_target_must_match_answer(self) -> None:
        payload = self._custom_registry()
        raw_cases = payload["cases"]
        assert isinstance(raw_cases, list)
        case = raw_cases[0]
        assert isinstance(case, dict)
        case.pop("target")
        case["source"] = "a b x"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertIn("case custom_case source is not an anagram of answer", errors)

    def test_performance_case_must_have_benchmark_tokens(self) -> None:
        payload = self._custom_registry()
        raw_cases = payload["cases"]
        assert isinstance(raw_cases, list)
        case = raw_cases[0]
        assert isinstance(case, dict)
        case["target"] = "!!!"
        case["answer"] = "!!!"
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertIn("case custom_case performance answer has no benchmark tokens", errors)

    def test_registry_rejects_solver_word_count_outside_cli_range(self) -> None:
        payload = self._custom_registry()
        raw_cases = payload["cases"]
        assert isinstance(raw_cases, list)
        case = raw_cases[0]
        assert isinstance(case, dict)
        case["solver"] = {"words": 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertTrue(any("--words must be >= 1" in error for error in errors))

    def test_registry_rejects_solver_cross_field_word_range(self) -> None:
        payload = self._custom_registry()
        raw_cases = payload["cases"]
        assert isinstance(raw_cases, list)
        case = raw_cases[0]
        assert isinstance(case, dict)
        case["solver"] = {"min_words": 5, "max_words": 3}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            errors = validate_registry(path)
        self.assertTrue(
            any("Invalid --min-words/--max-words range" in error for error in errors)
        )

    def test_missing_registry_is_a_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            errors = validate_registry(path)
        self.assertEqual(len(errors), 1)
        self.assertIn("missing.json", errors[0])

    def test_loader_rejects_duplicate_ids(self) -> None:
        payload = {
            "cases": [
                {"id": "same", "answer": "one two"},
                {"id": "same", "answer": "two one"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                load_cases(path)

    def test_loader_rejects_non_string_stable_id(self) -> None:
        payload = {"cases": [{"id": 7, "answer": "one two"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                load_cases(path)

    def test_loader_compatibility_mode_accepts_answer_only_training_case(self) -> None:
        payload = {"cases": [{"answer": "knowledge is power"}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_json(tmp, payload)
            cases = load_cases(path, require_ids=False)
        self.assertEqual(cases, [{"answer": "knowledge is power"}])

    def test_loader_rejects_unknown_selected_id(self) -> None:
        with self.assertRaisesRegex(KeyError, "definitely_missing"):
            load_cases(DEFAULT_CASES, {"definitely_missing"})


if __name__ == "__main__":
    unittest.main()
