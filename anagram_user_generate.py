#!/usr/bin/env python3
"""Apply the normal-user lexicon policy, then run the research generator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anagram_generate as generator
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
    lexicon = ensure_user_lexicon(
        dictionary_source=settings.dictionary,
        ngram_dir=Path(settings.ngram_dir),
        refresh=settings.refresh,
    )
    policy = ["--dict", str(lexicon.dictionary)]
    if lexicon.extra_short_words:
        policy += ["--extra-short-words", ",".join(lexicon.extra_short_words)]

    argv = original_args
    argv[separator:separator] = policy

    previous = sys.argv
    try:
        sys.argv = [str(previous[0]), *argv]
        return generator.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
