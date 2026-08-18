from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

import anagram_generate as generator


class LexicalCoverageTests(unittest.TestCase):
    def _load(
        self,
        words: str,
        target: str,
        *,
        unigrams: generator.UnigramModel | None,
        min_zipf: float = 0.0,
        excluded: set[str] | None = None,
        supplements: set[str] | None = None,
    ) -> list[generator.Candidate]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text(words, encoding="utf-8")
            return generator.load_words(
                path,
                generator.counts(target),
                2,
                99,
                excluded or set(),
                [],
                set(),
                min_zipf,
                "common",
                set(generator.DEFAULT_SHORT_WORDS),
                set(),
                unigrams,
                supplemental_words=supplements or set(),
            )

    def test_common_short_policy_admits_only_high_frequency_extra_words(self) -> None:
        model = generator.UnigramModel(
            counts={"hi": 1_000_000, "qi": 1},
            total=1_000_001,
        )
        candidates = self._load("hi\nqi\n", "hiqi", unigrams=model)
        self.assertIn("hi", {candidate.word for candidate in candidates})
        self.assertNotIn("qi", {candidate.word for candidate in candidates})

    def test_common_short_policy_stays_whitelist_only_without_frequency_data(self) -> None:
        candidates = self._load("hi\n", "hi", unigrams=None)
        self.assertNotIn("hi", {candidate.word for candidate in candidates})

    def test_standard_contraction_can_be_injected_when_dictionary_omits_it(self) -> None:
        model = generator.UnigramModel(counts={"dont": 10_000}, total=10_000)
        candidates = self._load(
            "done\n",
            "dont",
            unigrams=model,
            min_zipf=2.7,
            supplements={"dont"},
        )
        self.assertEqual([candidate.word for candidate in candidates], ["dont"])

    def test_supplement_respects_exclusion_and_frequency_filter(self) -> None:
        common = generator.UnigramModel(counts={"dont": 10_000}, total=10_000)
        excluded = self._load(
            "",
            "dont",
            unigrams=common,
            min_zipf=2.7,
            excluded={"dont"},
            supplements={"dont"},
        )
        self.assertEqual(excluded, [])

        rare = generator.UnigramModel(counts={"dont": 1}, total=1_000_000_000)
        filtered = self._load(
            "",
            "dont",
            unigrams=rare,
            min_zipf=2.7,
            supplements={"dont"},
        )
        self.assertEqual(filtered, [])

    def test_supplement_respects_regex_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "words.txt"
            path.write_text("", encoding="utf-8")
            model = generator.UnigramModel(counts={"dont": 10_000}, total=10_000)
            candidates = generator.load_words(
                path,
                generator.counts("dont"),
                2,
                99,
                set(),
                [re.compile(r"^dont$")],
                set(),
                0.0,
                "common",
                set(generator.DEFAULT_SHORT_WORDS),
                set(),
                model,
                supplemental_words={"dont"},
            )
        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
