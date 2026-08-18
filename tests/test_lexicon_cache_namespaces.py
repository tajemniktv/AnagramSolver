from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_user_lexicon as lexicon


class LexiconCacheNamespaceTests(unittest.TestCase):
    def test_ngram_changes_reuse_dictionary_artifact_but_split_policy_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dictionary = root / "words.txt"
            dictionary.write_text("hello\n", encoding="utf-8")
            derived = root / "derived"

            with patch.object(lexicon, "DICTIONARY_DIR", derived):
                first_dictionary, first_policy, first_token = lexicon._derived_cache_paths(
                    str(dictionary),
                    root / "ngrams-a",
                )
                second_dictionary, second_policy, second_token = lexicon._derived_cache_paths(
                    str(dictionary),
                    root / "ngrams-b",
                )

            self.assertEqual(first_dictionary, second_dictionary)
            self.assertNotEqual(first_policy, second_policy)
            self.assertNotEqual(first_token, second_token)

    def test_dictionary_changes_still_isolate_augmented_dictionary_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("alpha\n", encoding="utf-8")
            second.write_text("beta\n", encoding="utf-8")
            derived = root / "derived"

            with patch.object(lexicon, "DICTIONARY_DIR", derived):
                first_dictionary, _, _ = lexicon._derived_cache_paths(
                    str(first),
                    root / "ngrams",
                )
                second_dictionary, _, _ = lexicon._derived_cache_paths(
                    str(second),
                    root / "ngrams",
                )

            self.assertNotEqual(first_dictionary, second_dictionary)

    def test_identical_augmented_dictionary_is_not_rewritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base.txt"
            output = root / "augmented.txt"
            base.write_text("hello\n", encoding="utf-8")
            supplements = frozenset({"dont"})

            lexicon.build_augmented_dictionary(base, output, supplements)
            with patch.object(lexicon.os, "replace", wraps=lexicon.os.replace) as replace:
                lexicon.build_augmented_dictionary(base, output, supplements)

            replace.assert_not_called()
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["hello", "dont"],
            )


if __name__ == "__main__":
    unittest.main()
