from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anagram_rerank as rerank
import anagram_rerank_core as core
from anagram_performance import performance_hooks


def _lexicon() -> rerank.WordNetLexicon:
    return rerank.WordNetLexicon(
        nouns={"ball", "balls", "dog", "dogs", "room"},
        verbs={"chase", "run", "help"},
        adjs={"quiet", "quick"},
        advs={"fast"},
        noun_exc={},
        verb_exc={},
        verb_frames={"chase": frozenset({8}), "run": frozenset({1})},
    )


def _row(words: tuple[str, ...], rank: int) -> core.Row:
    return core.Row(
        words=words,
        word_count=len(words),
        old_rank=rank,
        old_pre=55.0,
        lex=0.73,
        fam=0.67,
        old_pair=0.5,
        hint=0.0,
        zavg=5.0,
        zmin=4.0,
        old_pcov=0.0,
        hints=(),
    )


def _canonicalize(rows: list[core.Row]) -> None:
    for row in rows:
        row.words = tuple(sorted(row.words))
    rows.sort(key=lambda row: (row.word_count, row.words))


class PreparationMemoizationTests(unittest.TestCase):
    def test_optimized_preparation_matches_core_fields_exactly(self) -> None:
        source = (
            (("dogs", "chase", "balls"), 1),
            (("quick", "dogs", "run"), 2),
            (("a", "quiet", "room"), 3),
            (("dogs", "chase", "balls"), 4),
        )
        expected = [_row(words, rank) for words, rank in source]
        actual = [_row(words, rank) for words, rank in source]

        expected_lex = _lexicon()
        _canonicalize(expected)
        with performance_hooks():
            rerank._CORE_PREPARE_ROWS(expected, expected_lex)

        rerank.prepare_rows(actual, _lexicon())

        self.assertEqual(
            [rerank._row_to_cache_dict(row) for row in actual],
            [rerank._row_to_cache_dict(row) for row in expected],
        )

    def test_repeated_bags_reuse_word_and_directed_pair_work(self) -> None:
        rows = [_row(("dogs", "chase", "balls"), rank) for rank in range(1, 41)]
        original_pair = core.pair_grammar
        original_family = core.morphology_family_word
        pair_calls = 0
        family_calls = 0

        def counted_pair(left: str, right: str, lex: core.WordNetLexicon) -> float:
            nonlocal pair_calls
            pair_calls += 1
            return original_pair(left, right, lex)

        def counted_family(word: str, lex: core.WordNetLexicon) -> str:
            nonlocal family_calls
            family_calls += 1
            return original_family(word, lex)

        with (
            patch.object(core, "pair_grammar", side_effect=counted_pair),
            patch.object(core, "morphology_family_word", side_effect=counted_family),
        ):
            rerank.prepare_rows(rows, _lexicon())

        self.assertEqual(pair_calls, 6)
        self.assertEqual(family_calls, 3)


class PreparedCacheCompressionTests(unittest.TestCase):
    def test_writer_uses_fast_gzip_and_round_trips(self) -> None:
        rows = [_row(("dogs", "chase", "balls"), 1)]
        rerank.prepare_rows(rows, _lexicon())

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "prepared.pickle"
            real_open = gzip.open
            levels: list[int] = []

            def recording_open(*args: object, **kwargs: object):
                level = kwargs.get("compresslevel")
                if isinstance(level, int):
                    levels.append(level)
                return real_open(*args, **kwargs)

            with patch.object(rerank.gzip, "open", side_effect=recording_open):
                rerank.save_prepared_cache(cache_path, rows)

            self.assertEqual(levels, [rerank.PREPARED_CACHE_COMPRESSLEVEL])
            self.assertEqual(rerank.PREPARED_CACHE_COMPRESSLEVEL, 1)
            loaded = rerank.load_prepared_cache(cache_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(
                [rerank._row_to_cache_dict(row) for row in loaded],
                [rerank._row_to_cache_dict(row) for row in rows],
            )

    def test_loader_accepts_existing_level_six_cache(self) -> None:
        rows = [_row(("quick", "dogs", "run"), 1)]
        rerank.prepare_rows(rows, _lexicon())
        payload = {
            "schema": rerank.PREPARED_CACHE_SCHEMA,
            "rows": [rerank._row_to_cache_dict(row) for row in rows],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "legacy.pickle"
            with gzip.open(
                cache_path, "wt", encoding="utf-8", compresslevel=6
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                )

            loaded = rerank.load_prepared_cache(cache_path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(
            [rerank._row_to_cache_dict(row) for row in loaded],
            [rerank._row_to_cache_dict(row) for row in rows],
        )


if __name__ == "__main__":
    unittest.main()
