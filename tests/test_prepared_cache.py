from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

import anagram_rerank as rerank


class PreparedCacheTests(unittest.TestCase):
    def _row(self) -> rerank.Row:
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
        row.v13_pre = 87.0
        row.family_key = ("dont", "hip", "lie", "these")
        row.deep = True
        row.best_order = ("these", "hips", "dont", "lie")
        row.final = 99.0
        return row

    def test_safe_json_cache_roundtrip_resets_deep_state(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prepared.cache"
            rerank.save_prepared_cache(path, [self._row()])
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                text = handle.read(100)
            self.assertIn(rerank.PREPARED_CACHE_SCHEMA, text)

            loaded = rerank.load_prepared_cache(path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[0].words, ("these", "dont", "lie", "hips"))
            self.assertEqual(loaded[0].family_key, ("dont", "hip", "lie", "these"))
            self.assertEqual(loaded[0].v13_pre, 87.0)
            self.assertFalse(loaded[0].deep)
            self.assertEqual(loaded[0].best_order, ())
            self.assertEqual(loaded[0].final, 0.0)

    def test_malformed_or_pickle_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "prepared.cache"
            path.write_bytes(b"not a gzip/json cache")
            self.assertIsNone(rerank.load_prepared_cache(path))


if __name__ == "__main__":
    unittest.main()
