from __future__ import annotations

import io
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import anagram_cache_io as cache_io
import anagram_generate as generator
import anagram_run_cache as cache
import anagram_solver as solver


class RankingCacheTests(unittest.TestCase):
    def test_key_tracks_options_candidates_sources_and_corpora(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = root / "candidates.txt"
            candidates.write_text("first", encoding="utf-8")
            with (
                patch.object(cache, "PROJECT_DIR", root),
                patch.object(cache, "NGRAM_DIR", root / "ngrams"),
                patch.object(cache, "WORDNET_DIR", root / "wordnet"),
            ):
                initial = cache.ranking_key(candidates, ["8"], None)
                self.assertNotEqual(
                    initial, cache.ranking_key(candidates, ["16"], None)
                )
                source = root / "anagram_fixture.py"
                source.write_text("changed", encoding="utf-8")
                changed = cache.ranking_key(candidates, ["8"], None)
                self.assertNotEqual(initial, changed)
                corpus = root / "wordnet" / "index.noun"
                corpus.parent.mkdir()
                corpus.write_text("noun", encoding="utf-8")
                provisioned = cache.ranking_key(candidates, ["8"], None)
                self.assertNotEqual(changed, provisioned)
                corpus.write_text("updated noun data", encoding="utf-8")
                self.assertNotEqual(
                    provisioned, cache.ranking_key(candidates, ["8"], None)
                )
                candidates.write_text("second", encoding="utf-8")
                self.assertNotEqual(
                    provisioned, cache.ranking_key(candidates, ["8"], None)
                )
                self.assertNotEqual(
                    cache.ranking_key(candidates, ["8"], corpus),
                    cache.ranking_key(candidates, ["8"], None),
                )

    def test_candidate_integrity_and_nonsemantic_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.txt"
            second = Path(tmp) / "second.txt"
            for path, seconds in ((first, 1.0), (second, 2.0)):
                generator.write_full_export(
                    path,
                    [],
                    True,
                    search={"generated": 0, "truncated": False, "seconds": seconds},
                )
                self.assertIsNotNone(solver._search_summary(path))
            self.assertEqual(
                cache.candidate_content_hash(first),
                cache.candidate_content_hash(second),
            )
            first.write_bytes(first.read_bytes()[:-10])
            self.assertIsNone(solver._search_summary(first))
            second.write_bytes(
                second.read_bytes().replace(b'"generated": 0', b'"generated": 1')
            )
            self.assertIsNone(solver._search_summary(second))

    def test_corrupt_result_cache_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            payload = {
                "summary": {},
                "results": [
                    {"word_count": 2, "rank": 1, "score": 42.0, "phrase": "ab cd"}
                ],
            }
            cache.save_results(path, payload)
            self.assertEqual(cache.load_results(path), payload)
            path.write_text(
                path.read_text(encoding="utf-8").replace("ab cd", "bad data"),
                encoding="utf-8",
            )
            self.assertIsNone(cache.load_results(path))
            path.write_bytes(b"\xff")
            self.assertIsNone(cache.load_results(path))
            path.write_text("{", encoding="utf-8")
            self.assertIsNone(cache.load_results(path))

    def test_interrupted_download_preserves_old_data_and_cleans_temporary(self):
        class Interrupted(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection interrupted")
                return super().read(size)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus"
            for existing in (False, True):
                if existing:
                    path.write_bytes(b"old corpus")
                with patch.object(
                    cache_io.urllib.request,
                    "urlopen",
                    return_value=Interrupted(b"partial"),
                ):
                    with self.assertRaisesRegex(OSError, "interrupted"):
                        cache_io.download_atomic(
                            urllib.request.Request("https://example.invalid"),
                            path,
                            timeout=1,
                        )
                self.assertEqual(
                    path.read_bytes() if path.exists() else None,
                    b"old corpus" if existing else None,
                )
                self.assertEqual(list(Path(tmp).glob(".*.tmp")), [])

    def test_short_http_response_is_not_published(self):
        class ShortResponse(io.BytesIO):
            headers = {"Content-Length": "100"}

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus"
            with patch.object(
                cache_io.urllib.request, "urlopen", return_value=ShortResponse(b"short")
            ):
                with self.assertRaisesRegex(OSError, "Incomplete download"):
                    cache_io.download_atomic(
                        urllib.request.Request("https://example.invalid"),
                        path,
                        timeout=1,
                    )
            self.assertFalse(path.exists())
