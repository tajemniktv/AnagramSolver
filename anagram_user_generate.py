#!/usr/bin/env python3
"""Apply the normal-user lexicon policy, then run the research generator."""

from __future__ import annotations

import sys

import anagram_generate as generator
from anagram_user_lexicon import ensure_user_lexicon


def main() -> int:
    lexicon = ensure_user_lexicon()
    argv = [
        *sys.argv[1:],
        "--dict",
        str(lexicon.dictionary),
    ]
    if lexicon.extra_short_words:
        argv += ["--extra-short-words", ",".join(lexicon.extra_short_words)]

    previous = sys.argv
    try:
        sys.argv = [str(previous[0]), *argv]
        return generator.main()
    finally:
        sys.argv = previous


if __name__ == "__main__":
    raise SystemExit(main())
