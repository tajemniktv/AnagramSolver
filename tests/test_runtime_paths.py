from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import anagram_benchmark
import anagram_generate
import anagram_paths
import anagram_rerank
import anagram_rerank_core
import anagram_solver
import build_wikimedia_phrase_index
import ci_phrase_matrix


class RuntimePathTests(unittest.TestCase):
    def test_default_runtime_paths_stay_under_project_data_root(self) -> None:
        data = anagram_paths.DATA_DIR.resolve()
        paths = (
            anagram_generate.DEFAULT_CACHE_DIR,
            anagram_generate.DEFAULT_DICT_CACHE,
            anagram_generate.DEFAULT_NGRAM_DIR,
            anagram_rerank_core.DEFAULT_WORDNET_DIR,
            anagram_rerank_core.DEFAULT_PREPARED_CACHE_DIR,
            anagram_rerank_core.DEFAULT_NGRAM_DIR,
            anagram_solver.DEFAULT_RUN_ROOT,
            anagram_benchmark.DEFAULT_CACHE,
            build_wikimedia_phrase_index.DEFAULT_CACHE,
            ci_phrase_matrix.RESULTS_DIR,
        )
        for path in paths:
            resolved = Path(path).resolve()
            self.assertTrue(resolved == data or data in resolved.parents, path)

    def test_project_data_root_is_next_to_scripts(self) -> None:
        self.assertEqual(anagram_paths.DATA_DIR.parent, anagram_paths.PROJECT_DIR)
        self.assertEqual(anagram_paths.PROJECT_DIR, Path(anagram_paths.__file__).resolve().parent)

    def test_python_defaults_do_not_reference_legacy_home_cache(self) -> None:
        for path in anagram_paths.PROJECT_DIR.glob("*.py"):
            self.assertNotIn(".multi_anagram", path.read_text(encoding="utf-8"), path.name)

    def test_prepared_cache_key_is_independent_of_project_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "copy-a" / "candidates.txt"
            right = root / "copy-b" / "candidates.txt"
            wordnet = root / "wordnet"
            left.parent.mkdir()
            right.parent.mkdir()
            wordnet.mkdir()
            payload = "same candidate export\n"
            left.write_text(payload, encoding="utf-8")
            right.write_text(payload, encoding="utf-8")

            self.assertEqual(
                anagram_rerank._prepared_cache_key(left, wordnet),
                anagram_rerank._prepared_cache_key(right, wordnet),
            )


if __name__ == "__main__":
    unittest.main()
