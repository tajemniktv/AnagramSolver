from __future__ import annotations

import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import anagram_generate as generator
import anagram_solver as solver
import anagram_user_generate as user_generate
import anagram_user_lexicon as lexicon


class LexicalCoverageTests(unittest.TestCase):
    def test_corpus_short_selection_admits_common_hi_but_not_rare_qi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = Path(tmp) / "words.txt"
            dictionary.write_text("hi\nqi\nwe\n", encoding="utf-8")
            model = generator.UnigramModel(
                counts={"hi": 1_000_000, "qi": 1, "we": 2_000_000},
                total=3_000_001,
            )
            selected = lexicon.select_corpus_short_words(
                dictionary,
                model,
                min_zipf=lexicon.DEFAULT_SHORT_WORD_MIN_ZIPF,
            )

        self.assertIn("hi", selected)
        self.assertIn("we", selected)
        self.assertNotIn("qi", selected)

    def test_short_selection_is_generic_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dictionary = Path(tmp) / "words.txt"
            dictionary.write_text("zz\naa\nzz\na\nlong\n", encoding="utf-8")
            model = generator.UnigramModel(
                counts={"aa": 100_000, "zz": 100_000},
                total=200_000,
            )
            selected = lexicon.select_corpus_short_words(
                dictionary,
                model,
                min_zipf=4.0,
            )

        self.assertEqual(selected, ("aa", "zz"))

    def test_augmented_dictionary_adds_missing_standard_forms_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.txt"
            output = Path(tmp) / "augmented.txt"
            base.write_text("done\ndont\nhello\n", encoding="utf-8")

            lexicon.build_augmented_dictionary(
                base,
                output,
                {"dont", "cant"},
            )
            words = [
                generator.normalize_token(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(words.count("dont"), 1)
        self.assertEqual(words.count("cant"), 1)
        self.assertIn("hello", words)

    def test_augmented_contraction_remains_subject_to_generator_zipf_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.txt"
            augmented = Path(tmp) / "augmented.txt"
            base.write_text("done\n", encoding="utf-8")
            lexicon.build_augmented_dictionary(base, augmented, {"dont"})

            common = generator.UnigramModel(counts={"dont": 10_000}, total=10_000)
            admitted = generator.load_words(
                augmented,
                generator.counts("dont"),
                2,
                99,
                set(),
                [],
                set(),
                2.7,
                "common",
                set(generator.DEFAULT_SHORT_WORDS),
                set(),
                common,
            )
            rare = generator.UnigramModel(counts={"dont": 1}, total=1_000_000_000)
            filtered = generator.load_words(
                augmented,
                generator.counts("dont"),
                2,
                99,
                set(),
                [],
                set(),
                2.7,
                "common",
                set(generator.DEFAULT_SHORT_WORDS),
                set(),
                rare,
            )

        self.assertEqual([candidate.word for candidate in admitted], ["dont"])
        self.assertEqual(filtered, [])

    def test_all_pretty_contractions_are_lexical_supplements(self) -> None:
        supplements = frozenset(generator.PRETTY_CONTRACTIONS)
        self.assertIn("dont", supplements)
        self.assertIn("cant", supplements)
        self.assertTrue(all("'" not in word for word in supplements))

    def test_malformed_policy_cache_shapes_are_cache_misses(self) -> None:
        policy_source = "test-policy-source"
        bad_payloads = (
            "null\n",
            "[]\n",
            '{}\n',
            json.dumps(
                {
                    "schema": 2,
                    "policy_source": policy_source,
                    "dictionary_stamp": None,
                    "unigram_stamp": [3, 4],
                    "extra_short_words": ["hi"],
                }
            )
            + "\n",
            json.dumps(
                {
                    "schema": 2,
                    "policy_source": policy_source,
                    "dictionary_stamp": [1, 2],
                    "unigram_stamp": "bad",
                    "extra_short_words": ["hi"],
                }
            )
            + "\n",
            json.dumps(
                {
                    "schema": 2,
                    "policy_source": policy_source,
                    "dictionary_stamp": [1, 2],
                    "unigram_stamp": [3, 4],
                    "extra_short_words": None,
                }
            )
            + "\n",
            json.dumps(
                {
                    "schema": 2,
                    "policy_source": policy_source,
                    "dictionary_stamp": [1, 2],
                    "unigram_stamp": [3, 4],
                    "extra_short_words": ["toolong"],
                }
            )
            + "\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "augmented.txt"
            policy = root / "policy.json"
            augmented.write_text("hi\n", encoding="utf-8")
            with (
                patch.object(lexicon, "AUGMENTED_DICTIONARY", augmented),
                patch.object(lexicon, "POLICY_CACHE", policy),
                patch.object(
                    lexicon,
                    "_policy_source_token",
                    return_value=policy_source,
                ),
            ):
                for payload in bad_payloads:
                    with self.subTest(payload=payload):
                        policy.write_text(payload, encoding="utf-8")
                        self.assertIsNone(
                            lexicon._load_cached_policy((1, 2), (3, 4))
                        )

    def test_policy_cache_is_invalidated_by_policy_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "augmented.txt"
            policy = root / "policy.json"
            augmented.write_text("hi\n", encoding="utf-8")
            policy.write_text(
                '{"schema":2,"policy_source":"old-policy",'
                '"dictionary_stamp":[1,2],"unigram_stamp":[3,4],'
                '"extra_short_words":["hi"]}\n',
                encoding="utf-8",
            )
            with (
                patch.object(lexicon, "AUGMENTED_DICTIONARY", augmented),
                patch.object(lexicon, "POLICY_CACHE", policy),
                patch.object(lexicon, "_policy_source_token", return_value="new-policy"),
            ):
                self.assertIsNone(lexicon._load_cached_policy((1, 2), (3, 4)))

    def test_valid_policy_cache_returns_source_dependent_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            augmented = root / "augmented.txt"
            policy = root / "policy.json"
            augmented.write_text("hi\n", encoding="utf-8")
            policy_source = lexicon._policy_source_token()
            policy.write_text(
                json.dumps(
                    {
                        "schema": 2,
                        "policy_source": policy_source,
                        "dictionary_stamp": [1, 2],
                        "unigram_stamp": [3, 4],
                        "extra_short_words": ["hi"],
                    },
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            with (
                patch.object(lexicon, "AUGMENTED_DICTIONARY", augmented),
                patch.object(lexicon, "POLICY_CACHE", policy),
            ):
                cached = lexicon._load_cached_policy((1, 2), (3, 4))

        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(cached.extra_short_words, ("hi",))
        self.assertEqual(
            cached.cache_token,
            lexicon._policy_token((1, 2), (3, 4), ("hi",)),
        )
        self.assertNotEqual(
            cached.cache_token,
            lexicon._policy_token((1, 3), (3, 4), ("hi",)),
        )

    def test_custom_sources_use_isolated_derived_lexicons_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dictionary_dir = root / "derived"
            ngram_dir = root / "ngrams"
            ngram_dir.mkdir()
            (ngram_dir / "count_1w.txt").write_text(
                "alpha\t100\nbeta\t100\n",
                encoding="utf-8",
            )
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("alpha\n", encoding="utf-8")
            second.write_text("beta\n", encoding="utf-8")

            def provision(path: Path) -> lexicon.UserLexicon:
                return lexicon.ensure_user_lexicon(
                    dictionary_source=str(path),
                    ngram_dir=ngram_dir,
                )

            with patch.object(lexicon, "DICTIONARY_DIR", dictionary_dir):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    first_result, second_result = list(
                        pool.map(provision, (first, second))
                    )

            self.assertNotEqual(first_result.dictionary, second_result.dictionary)
            self.assertNotEqual(first_result.cache_token, second_result.cache_token)
            self.assertIn(
                "alpha",
                first_result.dictionary.read_text(encoding="utf-8").splitlines(),
            )
            self.assertNotIn(
                "beta",
                first_result.dictionary.read_text(encoding="utf-8").splitlines(),
            )
            self.assertIn(
                "beta",
                second_result.dictionary.read_text(encoding="utf-8").splitlines(),
            )
            self.assertNotIn(
                "alpha",
                second_result.dictionary.read_text(encoding="utf-8").splitlines(),
            )

    def test_solver_run_key_changes_with_effective_lexicon_token(self) -> None:
        args = solver.build_parser().parse_args(["OEEEVHYNRI"])
        solver._validate_args(args)
        self.assertNotEqual(
            solver._run_key(args, user_lexicon_token="policy-a"),
            solver._run_key(args, user_lexicon_token="policy-b"),
        )

    def test_user_wrapper_help_does_not_provision_lexicon(self) -> None:
        with (
            patch.object(sys, "argv", ["anagram_user_generate.py", "--help"]),
            patch.object(user_generate, "ensure_user_lexicon") as ensure,
            patch.object(user_generate.generator, "main", return_value=0) as run,
        ):
            self.assertEqual(user_generate.main(), 0)

        ensure.assert_not_called()
        run.assert_called_once_with()

    def test_user_wrapper_forwards_source_settings_and_injects_before_separator(self) -> None:
        captured: list[str] = []

        def fake_generator_main() -> int:
            captured.extend(sys.argv[1:])
            return 0

        user_lexicon = lexicon.UserLexicon(
            Path("/tmp/augmented.txt"),
            ("hi", "we"),
            "policy-token",
        )
        original = [
            "anagram_user_generate.py",
            "--dict",
            "/tmp/base.txt",
            "--ngram-dir",
            "/tmp/ngrams",
            "--refresh",
            "--",
            "--letters",
        ]
        with (
            patch.object(sys, "argv", original),
            patch.object(
                user_generate,
                "ensure_user_lexicon",
                return_value=user_lexicon,
            ) as ensure,
            patch.object(
                user_generate.generator,
                "main",
                side_effect=fake_generator_main,
            ),
        ):
            self.assertEqual(user_generate.main(), 0)

        ensure.assert_called_once_with(
            dictionary_source="/tmp/base.txt",
            ngram_dir=Path("/tmp/ngrams"),
            refresh=True,
        )
        separator = captured.index("--")
        self.assertEqual(
            captured[separator - 4:separator],
            [
                "--dict",
                "/tmp/augmented.txt",
                "--extra-short-words",
                "hi,we",
            ],
        )
        self.assertEqual(captured[separator + 1:], ["--letters"])


if __name__ == "__main__":
    unittest.main()
