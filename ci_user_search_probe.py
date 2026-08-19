#!/usr/bin/env python3
"""Temporary PR diagnostic for where a known holdout bag falls out of the beam."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import anagram_generate as generator
import anagram_user_lexicon as lexicon
import anagram_user_search as search

TARGET = "ODITIHNSLSHEEEPT"
EXPECTED = ("these", "hips", "dont", "lie")
WORD_COUNT = len(EXPECTED)
NORMAL_MIN_WORDS = 2
NORMAL_MAX_WORDS = 6
NORMAL_MAX_RESULTS = 100_000
ANCHORED_ENUMERATION_CAP = 200_000


def _candidate_index(candidates: list[generator.Candidate], word: str) -> int:
    for index, candidate in enumerate(candidates):
        if candidate.word == word:
            return index
    raise SystemExit(f"Expected probe word absent from user vocabulary: {word}")


def main() -> int:
    user_lexicon = lexicon.ensure_user_lexicon()
    ngram_dir = Path(generator.DEFAULT_NGRAM_DIR)
    one_path, _ = generator.ensure_ngram_data(
        ngram_dir,
        refresh=False,
        need_bigrams=False,
    )
    unigrams = generator.load_unigram_model(one_path)
    target = generator.counts(TARGET)
    short_words = set(generator.DEFAULT_SHORT_WORDS)
    short_words.update(user_lexicon.extra_short_words)
    candidates = generator.load_words(
        user_lexicon.dictionary,
        target,
        min_len=2,
        max_len=sum(target),
        excluded_words=set(),
        exclude_regexes=[],
        forbid_chars=set(),
        min_zipf=2.7,
        short_policy="common",
        short_whitelist=short_words,
        forced_words=set(),
        unigrams=unigrams,
    )
    bigrams = search._load_search_bigram_model(
        candidates,
        unigrams=unigrams,
        ngram_dir=ngram_dir,
        refresh=False,
    )

    indices = tuple(sorted(_candidate_index(candidates, word) for word in EXPECTED))
    words = tuple(candidates[index].word for index in indices)
    expected_bag = tuple(sorted(words))
    print("Shakira canonical candidate path:")
    print(
        "  "
        + " -> ".join(
            f"{candidate.word}[idx={index},z={candidate.zipf:.2f}]"
            for index in indices
            for candidate in (candidates[index],)
        )
    )

    sparse = [search._sparse_signature(candidate.sig) for candidate in candidates]
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_signature[candidate.sig].append(index)

    remaining = target
    remaining_len = sum(remaining)
    start = 0
    chosen: tuple[int, ...] = ()
    min_len = min(candidate.length for candidate in candidates)
    max_len = max(candidate.length for candidate in candidates)
    branch_limit = search._branch_width(WORD_COUNT)

    for depth, target_index in enumerate(indices[:-1]):
        words_left_before = WORD_COUNT - depth
        words_left_after = words_left_before - 1
        # Ask the shared production expansion helper for every feasible branch;
        # compare the target's ordinal with the real production branch cutoff.
        expansions = list(
            search._iter_state_expansions(
                remaining,
                remaining_len,
                start,
                chosen,
                candidates=candidates,
                sparse_signatures=sparse,
                by_signature=by_signature,
                min_candidate_len=min_len,
                max_candidate_len=max_len,
                words_left_after=words_left_after,
                allow_repeat=True,
                branch_limit=len(candidates),
            )
        )
        accepted = [expansion.index for expansion in expansions]
        try:
            ordinal = accepted.index(target_index) + 1
        except ValueError:
            ordinal = 0
        status = "inside" if 0 < ordinal <= branch_limit else "OUTSIDE"
        print(
            f"  depth {depth + 1}: choose {candidates[target_index].word}; "
            f"feasible ordinal={ordinal}/{len(accepted)}, branch_limit={branch_limit} -> {status}"
        )

        target_expansion = next(
            (expansion for expansion in expansions if expansion.index == target_index),
            None,
        )
        if target_expansion is None:
            print("  target path is infeasible under production pruning; stopping path probe")
            break
        remaining = target_expansion.remaining
        remaining_len = target_expansion.remaining_len
        start = target_expansion.next_start
        chosen = target_expansion.chosen

    final_index = indices[-1]
    final_ok = final_index >= start and candidates[final_index].sig == remaining
    print(
        f"  final closure: {candidates[final_index].word}[idx={final_index}] "
        f"exact={final_ok}"
    )

    anchor = candidates[final_index]
    prefix_remaining = generator.subtract_counts(target, anchor.sig)
    if prefix_remaining is None:
        raise SystemExit("Probe anchor does not fit target")
    prefix_candidates = candidates[: final_index + 1]
    word_to_index = {candidate.word: index for index, candidate in enumerate(candidates)}
    anchored: list[
        tuple[tuple[float, float, float], float, tuple[str, ...]]
    ] = []
    for prefix in generator.solve(
        prefix_remaining,
        prefix_candidates,
        WORD_COUNT - 1,
        WORD_COUNT - 1,
        ANCHORED_ENUMERATION_CAP,
        True,
    ):
        prefix_indices = tuple(word_to_index[word] for word in prefix)
        if any(index > final_index for index in prefix_indices):
            continue
        bag_indices = (*prefix_indices, final_index)
        anchored.append(
            (
                search._pair_priority(bag_indices, candidates, bigrams),
                search._lexical_score(bag_indices, candidates),
                (*prefix, anchor.word),
            )
        )

    lexical_ranked = sorted(anchored, key=lambda item: (-item[1], item[2]))
    pair_ranked = sorted(anchored, key=lambda item: (item[0], item[1]), reverse=True)
    lexical_rank = next(
        (
            rank
            for rank, (_, _, bag) in enumerate(lexical_ranked, 1)
            if tuple(sorted(bag)) == expected_bag
        ),
        0,
    )
    pair_rank = next(
        (
            rank
            for rank, (_, _, bag) in enumerate(pair_ranked, 1)
            if tuple(sorted(bag)) == expected_bag
        ),
        0,
    )
    target_pair = search._pair_priority(indices, candidates, bigrams)
    target_lexical = search._lexical_score(indices, candidates)
    print(
        f"  final anchor={anchor.word}: lexical={target_lexical:.3f} "
        f"rank={lexical_rank}/{len(anchored)}; pair={target_pair} "
        f"rank={pair_rank}/{len(anchored)}"
    )

    nominal_quota = math.ceil(
        NORMAL_MAX_RESULTS / (NORMAL_MAX_WORDS - NORMAL_MIN_WORDS + 1)
    )
    result_limit = search._bucket_result_cap(WORD_COUNT, nominal_quota)
    bags, exact_examined, expansions = search._beam_bags_for_word_count(
        target,
        candidates,
        WORD_COUNT,
        result_limit,
        True,
        bigrams,
    )
    present = any(tuple(sorted(bag)) == expected_bag for bag in bags)
    print(
        f"  actual {WORD_COUNT}-word multi-view beam: target_present={present}; "
        f"limit={result_limit}; retained={len(bags)}; "
        f"exact_evaluated={exact_examined}; partial_expansions={expansions}"
    )
    if not present:
        print("  diagnostic result: target still drops out of the bounded multi-view beam")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
