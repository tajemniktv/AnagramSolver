#!/usr/bin/env python3
"""Apply normal-user lexicon/search policy, then run the research generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anagram_generate as generator
import anagram_user_search as user_search
from anagram_user_lexicon import ensure_user_lexicon


def _pre_separator_args(argv: list[str]) -> tuple[list[str], int]:
    try:
        separator = argv.index("--")
    except ValueError:
        separator = len(argv)
    return argv[:separator], separator


def _lexicon_settings(argv: list[str]) -> argparse.Namespace:
    visible, _ = _pre_separator_args(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dict", dest="dictionary", default=generator.DEFAULT_DICT_URL)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--ngram-dir", default=str(generator.DEFAULT_NGRAM_DIR))
    settings, _ = parser.parse_known_args(visible)
    return settings


def main() -> int:
    original_args = list(sys.argv[1:])
    visible, separator = _pre_separator_args(original_args)
    if any(option in {"-h", "--help"} for option in visible):
        return generator.main()

    settings = _lexicon_settings(original_args)
    ngram_dir = Path(settings.ngram_dir).expanduser()
    lexicon = ensure_user_lexicon(
        dictionary_source=settings.dictionary,
        ngram_dir=ngram_dir,
        refresh=settings.refresh,
    )
    policy = ["--dict", str(lexicon.dictionary)]
    if lexicon.extra_short_words:
        policy += ["--extra-short-words", ",".join(lexicon.extra_short_words)]

    argv = original_args
    argv[separator:separator] = policy

    previous_argv = sys.argv
    previous_solve = generator.solve
    previous_load_unigrams = generator.load_unigram_model
    previous_search_bigram_loader = user_search._load_search_bigram_model
    unigram_cache: dict[Path, generator.UnigramModel] = {}

    def cached_load_unigram_model(path: Path) -> generator.UnigramModel:
        """Share the generator's parsed unigram model with bounded search."""
        key = path.expanduser().resolve()
        cached = unigram_cache.get(key)
        if cached is None:
            cached = previous_load_unigrams(path)
            unigram_cache[key] = cached
        return cached

    def load_active_search_bigrams(
        candidates: list[generator.Candidate],
    ) -> generator.BigramModel:
        return previous_search_bigram_loader(
            candidates,
            ngram_dir=ngram_dir,
            refresh=settings.refresh,
        )

    try:
        # These hooks are scoped to the dedicated child process. Direct research
        # calls to anagram_generate.py keep the historical DFS/loading behavior.
        generator.load_unigram_model = cached_load_unigram_model
        user_search._load_search_bigram_model = load_active_search_bigrams
        generator.solve = user_search.make_quality_guided_solve(previous_solve)
        sys.argv = [str(previous_argv[0]), *argv]
        return generator.main()
    finally:
        sys.argv = previous_argv
        generator.solve = previous_solve
        generator.load_unigram_model = previous_load_unigrams
        user_search._load_search_bigram_model = previous_search_bigram_loader


if __name__ == "__main__":
    raise SystemExit(main())
