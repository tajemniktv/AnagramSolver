#!/usr/bin/env python3
"""Project-local runtime/cache paths used by AnagramSolver tools."""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / ".anagram_data"

DICTIONARY_DIR = DATA_DIR / "dictionary"
NGRAM_DIR = DATA_DIR / "ngrams"
WORDNET_DIR = DATA_DIR / "wordnet31"
PREPARED_CACHE_DIR = DATA_DIR / "prepared_cache"
SOLVER_RUNS_DIR = DATA_DIR / "solver_runs"
WIKIMEDIA_TITLES_DIR = DATA_DIR / "wikimedia_titles"
PHRASE_INDEX_DIR = DATA_DIR / "phrase_indexes"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
CI_BENCHMARK_RESULTS_DIR = DATA_DIR / "ci_benchmark_results"
