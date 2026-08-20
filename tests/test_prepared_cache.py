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


def _preparation_row(words: tuple[str, ...], rank: int) -> core.Row:
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


class PreparedCacheTests(unittest.TestCase):
    @staticmethod
    def _cached_row() -> rerank.Row:
        row = rerank.Row(
            words=("these", "dont", "lie", "hips"),
            word_count=4,
            old_rank=7,
            old_pre=50.0,
            lex=0.8,
            fam=0.7,
            old_pair=0.2,
            hint=0.9,
            zavg=4.5,
            zmin=3.6,
            old_pcov=0.0,
            hints=("dont",),
        )
        row.wn_coverage = 1.0
        row.grammar_potential = 0.8
        row.grammar_potential_norm = 0.9
        row.pre_score = 87.0
        row.family_key = ("dont", "hip", "lie", "these")
        row.deep = True
        row.best_order = ("these", "hips", "dont", "lie")
        row.final = 99.0
        return row

    def test_safe_json_cache_roundtrip_resets_deep_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prepared.cache"
            rerank.save_prepared_cache(path, [self._cached_row()])
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                text = handle.read(100)
            self.assertIn(rerank.PREPARED_CACHE_SCHEMA, text)

            loaded = rerank.load_prepared_cache(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[0].words, ("these", "dont", "lie", "hips"))
            self.assertEqual(loaded[0].family_key, ("dont", "hip", "lie", "these"))
            self.assertEqual(loaded[0].pre_score, 87.0)
            self.assertFalse(loaded[0].deep)
            self.assertEqual(loaded[0].best_order, ())
            self.assertEqual(loaded[0].final, 0.0)

    def test_malformed_or_pickle_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prepared.cache"
            path.write_bytes(b"not a gzip/json cache")
            self.assertIsNone(rerank.load_prepared_cache(path))

    def test_old_schema_is_rejected_without_row_parsing(self) -> None:
        self.assertEqual(rerank.PREPARED_CACHE_SCHEMA, "topk-prepared-json-gzip-2")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "old-cache.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(
                    {"schema": "topk-prepared-json-gzip-1", "rows": []},
                    handle,
                )
            self.assertIsNone(rerank.load_prepared_cache(path))

    def test_writer_uses_fast_gzip_and_round_trips(self) -> None:
        rows = [_preparation_row(("dogs", "chase", "balls"), 1)]
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
        rows = [_preparation_row(("quick", "dogs", "run"), 1)]
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


class PreparationMemoizationTests(unittest.TestCase):
    def test_optimized_preparation_matches_core_fields_exactly(self) -> None:
        source = (
            (("dogs", "chase", "balls"), 1),
            (("quick", "dogs", "run"), 2),
            (("a", "quiet", "room"), 3),
            (("dogs", "chase", "balls"), 4),
        )
        expected = [_preparation_row(words, rank) for words, rank in source]
        actual = [_preparation_row(words, rank) for words, rank in source]

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
        rows = [
            _preparation_row(("dogs", "chase", "balls"), rank)
            for rank in range(1, 41)
        ]
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

    def test_tiny_cache_limits_preserve_core_fields(self) -> None:
        source = (
            (("dogs", "chase", "balls"), 1),
            (("quick", "dogs", "run"), 2),
            (("a", "quiet", "room"), 3),
        )
        expected = [_preparation_row(words, rank) for words, rank in source]
        actual = [_preparation_row(words, rank) for words, rank in source]
        _canonicalize(expected)
        with performance_hooks():
            rerank._CORE_PREPARE_ROWS(expected, _lexicon())

        with (
            patch.object(rerank, "_PREPARE_PAIR_CACHE_LIMIT", 2),
            patch.object(rerank, "_PREPARE_WORD_CACHE_LIMIT", 2),
        ):
            rerank.prepare_rows(actual, _lexicon())

        self.assertEqual(
            [rerank._row_to_cache_dict(row) for row in actual],
            [rerank._row_to_cache_dict(row) for row in expected],
        )

    def test_prepare_uses_core_grammar_potential_formula(self) -> None:
        rows = [_preparation_row(("dogs", "chase", "balls"), 1)]
        with patch.object(core, "grammar_potential", return_value=0.321) as grammar:
            rerank.prepare_rows(rows, _lexicon())

        grammar.assert_called_once()
        self.assertEqual(rows[0].grammar_potential_norm, 0.321)


if __name__ == "__main__":
    unittest.main()
