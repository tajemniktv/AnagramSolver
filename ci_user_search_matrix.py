#!/usr/bin/env python3
"""Informational A/B for a feasibility-aware partial-bag lexical bound."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anagram_generate as generator
import anagram_user_lexicon as lexicon
import anagram_user_search as search

# This is deliberately a small fixed-work research probe. It asks whether a
# better partial-state score retains intended answer paths more efficiently,
# rather than proving that a sufficiently huge beam can brute-force them.
RESEARCH_BEAM_WIDTH = 384


@dataclass(frozen=True, slots=True)
class Holdout:
    name: str
    answer: str
    min_zipf: float = 2.7


@dataclass(frozen=True, slots=True)
class TraceResult:
    survived: bool
    drop_depth: int | None
    expansions: int
    seconds: float


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


def _best_feasible_future_zipf(
    remaining: tuple[int, ...],
    remaining_len: int,
    next_start: int,
    words_left: int,
    candidates: list[generator.Candidate],
    sparse_signatures: list[tuple[tuple[int, int], ...]],
    min_candidate_len: int,
    max_candidate_len: int,
) -> float:
    """Return the best suffix Zipf that can actually participate in completion.

    Candidate order is descending Zipf, so the first suffix candidate that fits
    the remaining letters and leaves a feasible amount of text for the other
    future words is the strongest one-word frequency bound available to this
    state. Repeating that value is still optimistic when multiple words remain,
    but unlike the current global suffix bound it does not reward an impossible
    future word merely because that word is frequent.
    """
    if words_left <= 0:
        return 0.0
    after_words = words_left - 1
    min_this_len = max(
        min_candidate_len,
        remaining_len - after_words * max_candidate_len,
    )
    max_this_len = min(
        max_candidate_len,
        remaining_len - after_words * min_candidate_len,
    )

    for index in range(next_start, len(candidates)):
        candidate = candidates[index]
        if candidate.length < min_this_len or candidate.length > max_this_len:
            continue
        if search._fits_sparse(sparse_signatures[index], remaining):
            return candidate.zipf
    return -99.0


def _target_indices(
    expected: tuple[str, ...],
    candidates: list[generator.Candidate],
) -> tuple[int, ...] | None:
    by_word = {candidate.word: index for index, candidate in enumerate(candidates)}
    try:
        return tuple(sorted(by_word[word] for word in expected))
    except KeyError:
        return None


def _trace_target_path(
    target: tuple[int, ...],
    target_indices: tuple[int, ...],
    candidates: list[generator.Candidate],
    bigrams: generator.BigramModel,
    *,
    feasible_future: bool,
) -> TraceResult:
    started = time.perf_counter()
    word_count = len(target_indices)
    sparse_signatures = [search._sparse_signature(candidate.sig) for candidate in candidates]
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_signature[candidate.sig].append(index)

    min_candidate_len = min(candidate.length for candidate in candidates)
    max_candidate_len = max(candidate.length for candidate in candidates)
    states: list[search.BeamState] = [(target, sum(target), 0, ())]
    branch_limit = search._branch_width(word_count)
    serial = 0
    expansions = 0

    for depth in range(max(0, word_count - 1)):
        words_left_after = word_count - depth - 1
        lexical_heap: list[search.ScoredItem] = []
        pair_heap: list[search.PairItem] = []
        anchor_heaps: dict[int, list[search.ScoredItem]] = defaultdict(list)

        for rem, rem_len, start, chosen in states:
            for expansion in search._iter_state_expansions(
                rem,
                rem_len,
                start,
                chosen,
                candidates=candidates,
                sparse_signatures=sparse_signatures,
                by_signature=by_signature,
                min_candidate_len=min_candidate_len,
                max_candidate_len=max_candidate_len,
                words_left_after=words_left_after,
                allow_repeat=True,
                branch_limit=branch_limit,
            ):
                new_chosen = expansion.chosen
                if words_left_after:
                    if feasible_future:
                        best_future_zipf = _best_feasible_future_zipf(
                            expansion.remaining,
                            expansion.remaining_len,
                            expansion.next_start,
                            words_left_after,
                            candidates,
                            sparse_signatures,
                            min_candidate_len,
                            max_candidate_len,
                        )
                    else:
                        best_future_zipf = candidates[expansion.next_start].zipf
                    lexical_priority = search._optimistic_score(
                        new_chosen,
                        words_left_after,
                        best_future_zipf,
                        candidates,
                    )
                else:
                    lexical_priority = search._lexical_score(new_chosen, candidates)

                payload: tuple[Any, ...] = (
                    expansion.remaining,
                    expansion.remaining_len,
                    expansion.next_start,
                    new_chosen,
                )
                lexical_item: search.ScoredItem = (lexical_priority, -serial, payload)
                pair_priority = search._pair_priority(new_chosen, candidates, bigrams)
                pair_item: search.PairItem = (*pair_priority, -serial, payload)
                serial += 1
                expansions += 1
                search._push_bounded(lexical_heap, lexical_item, RESEARCH_BEAM_WIDTH)
                search._push_pair_bounded(pair_heap, pair_item, RESEARCH_BEAM_WIDTH)
                search._push_bounded(
                    anchor_heaps[new_chosen[-1]],
                    lexical_item,
                    search.ANCHOR_CHAMPIONS_PER_WORD,
                )

        selected = search._select_multi_view(
            lexical_heap,
            pair_heap,
            anchor_heaps,
            RESEARCH_BEAM_WIDTH,
        )
        states = [
            (payload[0], payload[1], payload[2], payload[3])
            for payload in selected
        ]
        target_prefix = target_indices[: depth + 1]
        if not any(chosen == target_prefix for _, _, _, chosen in states):
            return TraceResult(
                False,
                depth + 1,
                expansions,
                time.perf_counter() - started,
            )

    if word_count == 1:
        survived = any(
            index == target_indices[0]
            for index in by_signature.get(target, ())
        )
    else:
        prefix = target_indices[:-1]
        final_index = target_indices[-1]
        survived = any(
            chosen == prefix
            and final_index >= start
            and candidates[final_index].sig == rem
            for rem, _rem_len, start, chosen in states
        )
    return TraceResult(
        survived,
        None if survived else word_count,
        expansions,
        time.perf_counter() - started,
    )


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
        tuple[
            Holdout,
            tuple[int, ...],
            tuple[str, ...],
            list[generator.Candidate],
            tuple[int, ...] | None,
        ]
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
        indices = _target_indices(expected, candidates)
        prepared.append((holdout, target, expected, candidates, indices))
        vocabulary.update(candidate.word for candidate in candidates)

    bigrams = generator.load_bigram_model(two_path, unigrams, vocabulary)

    baseline_hits = 0
    feasible_hits = 0
    comparable = 0
    print(
        f"partial-state A/B: beam={RESEARCH_BEAM_WIDTH:,}; "
        "baseline suffix-Zipf vs feasible suffix-Zipf",
        flush=True,
    )
    print(
        "case                    words vocab  baseline feasible  "
        "drop(base/fit)  expansions(base/fit)  seconds(base/fit)",
        flush=True,
    )
    print("-" * 108, flush=True)

    for holdout, target, expected, candidates, indices in prepared:
        if indices is None:
            missing = sorted(set(expected) - {candidate.word for candidate in candidates})
            print(
                f"{holdout.name:<23} {len(expected):>5} {len(candidates):>5}  "
                f"lexical-miss={','.join(missing)}",
                flush=True,
            )
            continue

        baseline = _trace_target_path(
            target,
            indices,
            candidates,
            bigrams,
            feasible_future=False,
        )
        feasible = _trace_target_path(
            target,
            indices,
            candidates,
            bigrams,
            feasible_future=True,
        )
        baseline_hits += int(baseline.survived)
        feasible_hits += int(feasible.survived)
        comparable += 1
        base_drop = "-" if baseline.drop_depth is None else str(baseline.drop_depth)
        fit_drop = "-" if feasible.drop_depth is None else str(feasible.drop_depth)
        print(
            f"{holdout.name:<23} {len(expected):>5} {len(candidates):>5}  "
            f"{baseline.survived!s:>8} {feasible.survived!s:>8}  "
            f"{base_drop:>4}/{fit_drop:<4}  "
            f"{baseline.expansions:>9}/{feasible.expansions:<9}  "
            f"{baseline.seconds:>7.3f}/{feasible.seconds:<7.3f}",
            flush=True,
        )

    print(flush=True)
    print(
        f"target-path retention: baseline={baseline_hits}/{comparable}; "
        f"feasible-future={feasible_hits}/{comparable}; "
        f"delta={feasible_hits - baseline_hits:+d}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
