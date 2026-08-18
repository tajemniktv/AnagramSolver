from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anagram_generate as generator
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
                min_zipf=4.5,
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


if __name__ == "__main__":
    unittest.main()
