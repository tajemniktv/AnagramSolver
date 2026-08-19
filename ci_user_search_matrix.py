#!/usr/bin/env python3
"""Informational A/B for contextual partial-beam diversity on unrelated holdouts."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import anagram_generate as generator
import anagram_user_lexicon as lexicon
import anagram_user_search as search

NORMAL_MIN_WORDS = 2
NORMAL_MAX_WORDS = 6
NORMAL_MAX_RESULTS = 100_000


@dataclass(frozen=True, slots=True)
class Holdout:
    name: str
    answer: str
    min_zipf: float = 2.7


HOLDOUTS = (
    Holdout("knowledge_power", "knowledge is power"),
    Holdout("better_late", "better late than never"),
    Holdout("practice_perfect", "practice makes perfect"),
    Holdout("still_waters", "still waters run deep"),
    Holdout("look_leap", "look before you leap"),
    Holdout("actions_words", "actions speak louder than words"),
    Holdout("dog_ball", "the dog chased the ball"),
    Holdout("shakira", "these hips dont lie"),
)


def _contains_expected(
    bags: list[tuple[str, ...]],
    expected: tuple[str, ...],
) -> bool:
    return any(tuple(sorted(bag)) == expected for bag in bags)


def _run_beam(
    target: tuple[int, ...],
    candidates: list[generator.Candidate],
    word_count: int,
    result_limit: int,
    bigrams: generator.BigramModel,
    *,
    context_champions: int,
) -> tuple[bool, int, int, float, list[tuple[str, ...]]]:
    previous = search.CONTEXT_CHAMPIONS_PER_GROUP
    search.CONTEXT_CHAMPIONS_PER_GROUP = context_champions
    started = time.perf_counter()
    try:
        bags, exact, expansions = search._beam_bags_for_word_count(
            target,
            candidates,
            word_count,
            result_limit,
            True,
            bigrams,
        )
    finally:
        search.CONTEXT_CHAMPIONS_PER_GROUP = previous
    return bool(bags), exact, expansions, time.perf_counter() - started, bags


def main() -> int:
    user_lexicon = lexicon.ensure_user_lexicon()
    ngram_dir = Path(generator.DEFAULT_NGRAM_DIR)
    one_path, two_path = generator.ensure_ngram_data(
        ngram_dir,
        refresh=False,
        need_bigrams=True,
    )
    if two_path is None:
        raise RuntimeError("Bigram corpus unavailable for search matrix")
    unigrams = generator.load_unigram_model(one_path)
    short_words = set(generator.DEFAULT_SHORT_WORDS)
    short_words.update(user_lexicon.extra_short_words)

    prepared: list[
        tuple[Holdout, tuple[int, ...], tuple[str, ...], list[generator.Candidate]]
    ] = []
    vocabulary: set[str] = set()
    for holdout in HOLDOUTS:
        target = generator.counts(holdout.answer)
        expected = tuple(sorted(generator.tokenize_words(holdout.answer)))
        candidates = generator.load_words(
            user_lexicon.dictionary,
            target,
            min_len=2,
            max_len=sum(target),
            excluded_words=set(),
            exclude_regexes=[],
            forbid_chars=set(),
            min_zipf=holdout.min_zipf,
            short_policy="common",
            short_whitelist=short_words,
            forced_words=set(),
            unigrams=unigrams,
        )
        prepared.append((holdout, target, expected, candidates))
        vocabulary.update(candidate.word for candidate in candidates)

    bigrams = generator.load_bigram_model(two_path, unigrams, vocabulary)
    nominal_quota = math.ceil(
        NORMAL_MAX_RESULTS / (NORMAL_MAX_WORDS - NORMAL_MIN_WORDS + 1)
    )

    baseline_hits = 0
    context_hits = 0
    comparable = 0
    print(
        "case                    words vocab limit  anchor context  "
        "exact(anchor/context)  expansions(anchor/context)  seconds(anchor/context)"
    )
    print("-" * 122)

    for holdout, target, expected, candidates in prepared:
        word_count = len(expected)
        missing = sorted(set(expected) - {candidate.word for candidate in candidates})
        if missing:
            print(
                f"{holdout.name:<23} {word_count:>5} {len(candidates):>5} "
                f"  n/a   lexical-miss={','.join(missing)}"
            )
            continue

        result_limit = search._bucket_result_cap(word_count, nominal_quota)
        _, base_exact, base_exp, base_seconds, base_bags = _run_beam(
            target,
            candidates,
            word_count,
            result_limit,
            bigrams,
            context_champions=0,
        )
        _, ctx_exact, ctx_exp, ctx_seconds, ctx_bags = _run_beam(
            target,
            candidates,
            word_count,
            result_limit,
            bigrams,
            context_champions=search.CONTEXT_CHAMPIONS_PER_GROUP,
        )
        base_hit = _contains_expected(base_bags, expected)
        ctx_hit = _contains_expected(ctx_bags, expected)
        baseline_hits += int(base_hit)
        context_hits += int(ctx_hit)
        comparable += 1
        print(
            f"{holdout.name:<23} {word_count:>5} {len(candidates):>5} {result_limit:>5}  "
            f"{base_hit!s:>6} {ctx_hit!s:>7}  "
            f"{base_exact:>7}/{ctx_exact:<7}  {base_exp:>9}/{ctx_exp:<9}  "
            f"{base_seconds:>7.3f}/{ctx_seconds:<7.3f}"
        )

    print()
    print(
        f"target-bag retention: anchor-only={baseline_hits}/{comparable}; "
        f"contextual={context_hits}/{comparable}; "
        f"delta={context_hits - baseline_hits:+d}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
