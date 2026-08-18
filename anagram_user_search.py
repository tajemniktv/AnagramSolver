"""Quality-guided bounded exact-bag search for normal solver runs.

The low-level generator keeps its historical DFS for research, clues, and true
exhaustive enumeration. Normal bounded searches need a different contract: the
candidate cap should retain plausible word bags rather than whichever exact
covers happen to occur first in DFS order.

This module uses a bounded lexical beam over partial word bags. It is explicitly
approximate: generation only needs to preserve a strong, varied shortlist for
the deeper linguistic reranker, not prove the globally exact lexical top-K.
"""

from __future__ import annotations

import heapq
import math
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import anagram_generate as generator

SolveCallable = Callable[..., Iterator[tuple[str, ...]]]
BeamState = tuple[tuple[int, ...], int, int, tuple[int, ...]]
ScoredItem = tuple[float, int, tuple[Any, ...]]
ANCHOR_CHAMPIONS_PER_WORD = 2


def _sparse_signature(sig: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    return tuple((letter, amount) for letter, amount in enumerate(sig) if amount)


def _fits_sparse(
    sparse: tuple[tuple[int, int], ...],
    remaining: tuple[int, ...],
) -> bool:
    return all(remaining[letter] >= amount for letter, amount in sparse)


def _subtract_sparse(
    remaining: tuple[int, ...],
    sparse: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    mutable = list(remaining)
    for letter, amount in sparse:
        mutable[letter] -= amount
    return tuple(mutable)


def _lexical_score(indices: tuple[int, ...], candidates: list[generator.Candidate]) -> float:
    """Approximate the generator's lexical score for a complete fixed-size bag."""
    zipfs = [candidates[index].zipf for index in indices]
    if not zipfs:
        return 0.0
    average = sum(zipfs) / len(zipfs)
    low_count = min(2, len(zipfs))
    low_tail = sum(sorted(zipfs)[:low_count]) / low_count
    duplicate_penalty = 1.25 * (len(indices) - len(set(indices)))
    zero_frequency_penalty = 0.35 * sum(value <= 0.0 for value in zipfs)
    return 0.78 * average + 0.22 * low_tail - duplicate_penalty - zero_frequency_penalty


def _optimistic_score(
    chosen_indices: tuple[int, ...],
    words_left: int,
    best_future_zipf: float,
    candidates: list[generator.Candidate],
) -> float:
    """Return an optimistic lexical score for a partial bag.

    Candidate indices are monotonic and the source list is descending by Zipf.
    Filling every unknown slot with the best Zipf still available at ``start``
    therefore cannot undershoot a real completion. Non-positive duplicate/junk
    penalties are omitted as well. The beam uses this as a ranking heuristic;
    unlike the previous exact branch-and-bound experiment, no correctness claim
    depends on keeping every state whose bound is competitive.
    """
    values = [candidates[index].zipf for index in chosen_indices]
    values.extend([best_future_zipf] * words_left)
    if not values:
        return 0.0
    average = sum(values) / len(values)
    low_count = min(2, len(values))
    low_tail = sum(sorted(values)[:low_count]) / low_count
    return 0.78 * average + 0.22 * low_tail


def _bucket_result_cap(word_count: int, requested: int) -> int:
    """Bound how much lexical-only material a bucket sends to deep reranking."""
    if word_count <= 4:
        ceiling = 10_000
    elif word_count == 5:
        ceiling = 6_000
    else:
        ceiling = 4_000
    return max(1, min(requested, ceiling))


def _beam_width(word_count: int, result_limit: int) -> int:
    if word_count <= 4:
        floor = 8_000
    elif word_count == 5:
        floor = 5_000
    else:
        floor = 3_000
    return max(result_limit, floor)


def _branch_width(word_count: int) -> int:
    if word_count <= 4:
        return 192
    if word_count == 5:
        return 112
    return 72


def _push_bounded(
    heap: list[ScoredItem],
    item: ScoredItem,
    limit: int,
) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item[:2] > heap[0][:2]:
        heapq.heapreplace(heap, item)


def _quality_anchor_limits(limit: int, candidate_count: int) -> tuple[int, int]:
    """Split capacity between global quality and rare-word anchor champions."""
    if limit <= 1:
        return max(limit, 0), 0
    possible_anchor_items = candidate_count * ANCHOR_CHAMPIONS_PER_WORD
    anchor_capacity = min(limit // 4, possible_anchor_items)
    return limit - anchor_capacity, anchor_capacity


def _select_quality_with_anchors(
    quality_heap: list[ScoredItem],
    anchor_heaps: dict[int, list[ScoredItem]],
    limit: int,
) -> list[ScoredItem]:
    """Combine the global score core with per-rarest-word champions.

    Candidate indices are monotonic in a word bag, so the last chosen index is
    also its least frequent chosen word. Keeping a couple of best states/bags per
    such anchor prevents the entire bounded search from becoming thousands of
    near-duplicates made only of the corpus's most frequent words.
    """
    selected = sorted(quality_heap, reverse=True)
    seen = {item[2] for item in selected}
    anchor_candidates = sorted(
        (item for heap in anchor_heaps.values() for item in heap),
        reverse=True,
    )
    for item in anchor_candidates:
        if len(selected) >= limit:
            break
        if item[2] in seen:
            continue
        selected.append(item)
        seen.add(item[2])
    selected.sort(reverse=True)
    return selected


def _beam_bags_for_word_count(
    remaining: tuple[int, ...],
    candidates: list[generator.Candidate],
    word_count: int,
    limit: int,
    allow_repeat: bool,
) -> tuple[list[tuple[str, ...]], int, int]:
    """Return a deterministic quality-plus-anchor beam for one word count."""
    if limit <= 0 or not candidates:
        return [], 0, 0

    min_candidate_len = min(candidate.length for candidate in candidates)
    max_candidate_len = max(candidate.length for candidate in candidates)
    remaining_len = sum(remaining)
    if remaining_len < word_count * min_candidate_len:
        return [], 0, 0
    if remaining_len > word_count * max_candidate_len:
        return [], 0, 0

    sparse_signatures = [_sparse_signature(candidate.sig) for candidate in candidates]
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_signature[candidate.sig].append(index)

    states: list[BeamState] = [(remaining, remaining_len, 0, ())]
    width = _beam_width(word_count, limit)
    quality_width, _ = _quality_anchor_limits(width, len(candidates))
    branch_limit = _branch_width(word_count)
    partial_expansions = 0
    serial = 0

    # Choose every word except the final exact signature closure. The final step
    # is handled through by_signature so rare-but-required last words are never
    # lost merely because they fall outside the local branch width.
    for depth in range(max(0, word_count - 1)):
        words_left_before = word_count - depth
        words_left_after = words_left_before - 1
        quality_heap: list[ScoredItem] = []
        anchor_heaps: dict[int, list[ScoredItem]] = defaultdict(list)

        for rem, rem_len, start, chosen in states:
            min_this_len = max(
                min_candidate_len,
                rem_len - words_left_after * max_candidate_len,
            )
            max_this_len = min(
                max_candidate_len,
                rem_len - words_left_after * min_candidate_len,
            )
            accepted_branches = 0

            for index in range(start, len(candidates)):
                candidate = candidates[index]
                if candidate.length < min_this_len or candidate.length > max_this_len:
                    continue
                sparse = sparse_signatures[index]
                if not _fits_sparse(sparse, rem):
                    continue

                new_rem = _subtract_sparse(rem, sparse)
                new_rem_len = rem_len - candidate.length
                next_start = index if allow_repeat else index + 1
                if words_left_after > 0 and next_start >= len(candidates):
                    continue

                # At the penultimate depth, avoid filling the beam with partial
                # states whose residual signature has no legal exact closure.
                if words_left_after == 1 and not any(
                    last_index >= next_start
                    for last_index in by_signature.get(new_rem, ())
                ):
                    continue

                new_chosen = (*chosen, index)
                if words_left_after:
                    best_future_zipf = candidates[next_start].zipf
                    priority = _optimistic_score(
                        new_chosen,
                        words_left_after,
                        best_future_zipf,
                        candidates,
                    )
                else:
                    priority = _lexical_score(new_chosen, candidates)

                item: ScoredItem = (
                    priority,
                    -serial,
                    (new_rem, new_rem_len, next_start, new_chosen),
                )
                serial += 1
                partial_expansions += 1
                _push_bounded(quality_heap, item, quality_width)
                _push_bounded(
                    anchor_heaps[new_chosen[-1]],
                    item,
                    ANCHOR_CHAMPIONS_PER_WORD,
                )

                accepted_branches += 1
                if accepted_branches >= branch_limit:
                    break

        selected = _select_quality_with_anchors(
            quality_heap,
            anchor_heaps,
            width,
        )
        if not selected:
            return [], 0, partial_expansions
        states = [
            (payload[0], payload[1], payload[2], payload[3])
            for _, _, payload in selected
        ]

    exact_examined = 0
    result_quality_width, _ = _quality_anchor_limits(limit, len(candidates))
    result_quality_heap: list[ScoredItem] = []
    result_anchor_heaps: dict[int, list[ScoredItem]] = defaultdict(list)
    result_serial = 0

    if word_count == 1:
        states = [(remaining, remaining_len, 0, ())]

    for rem, _rem_len, start, chosen in states:
        for last_index in by_signature.get(rem, ()):
            if last_index < start:
                continue
            indices = (*chosen, last_index)
            exact_examined += 1
            score = _lexical_score(indices, candidates)
            item: ScoredItem = (score, -result_serial, (indices,))
            result_serial += 1
            _push_bounded(result_quality_heap, item, result_quality_width)
            _push_bounded(
                result_anchor_heaps[indices[-1]],
                item,
                ANCHOR_CHAMPIONS_PER_WORD,
            )

    ranked = _select_quality_with_anchors(
        result_quality_heap,
        result_anchor_heaps,
        limit,
    )
    bags = [
        tuple(candidates[index].word for index in payload[0])
        for _, _, payload in ranked
    ]
    return bags, exact_examined, partial_expansions


def quality_guided_bounded_solve(
    remaining: tuple[int, ...],
    candidates: list[generator.Candidate],
    min_words: int,
    max_words: int,
    max_results: int,
    allow_repeat: bool,
    *,
    stats: generator.SearchStats | None = None,
) -> Iterator[tuple[str, ...]]:
    """Yield a bounded lexical shortlist spread across requested word counts."""
    if max_results <= 0:
        return

    search_stats = stats if stats is not None else generator.SearchStats()
    word_counts = tuple(range(min_words, max_words + 1))
    if not word_counts:
        return

    nominal_quota = max(1, math.ceil(max_results / len(word_counts)))
    bucket_results: dict[int, list[tuple[str, ...]]] = {}
    total_examined = 0
    total_partial_expansions = 0

    for word_count in word_counts:
        result_limit = _bucket_result_cap(word_count, nominal_quota)
        bags, examined, expansions = _beam_bags_for_word_count(
            remaining,
            candidates,
            word_count,
            result_limit,
            allow_repeat,
        )
        bucket_results[word_count] = bags
        total_examined += examined
        total_partial_expansions += expansions

    retained_total = sum(len(bags) for bags in bucket_results.values())
    search_stats.exact_examined += total_examined
    search_stats.accepted += retained_total

    for word_count in word_counts:
        yield from bucket_results.get(word_count, ())

    print(
        f"Quality-guided bounded search evaluated {total_examined:,} exact bag(s), "
        f"expanded {total_partial_expansions:,} partial state(s), and retained "
        f"{retained_total:,} across {min_words}-{max_words} words.",
        file=sys.stderr,
    )


def make_quality_guided_solve(fallback: SolveCallable) -> SolveCallable:
    """Return a generator.solve-compatible scoped normal-user search function."""

    def solve(
        remaining: tuple[int, ...],
        candidates: list[generator.Candidate],
        min_words: int,
        max_words: int,
        max_results: int,
        allow_repeat: bool,
        *,
        clue_words: set[str] | None = None,
        hint_mode: str = "any",
        initial_clue_words: set[str] | None = None,
        stats: generator.SearchStats | None = None,
        **kwargs: Any,
    ) -> Iterator[tuple[str, ...]]:
        clues = clue_words or set()
        # Clue-aware DFS has semantics and pruning specifically designed for
        # hints. Unlimited search must also stay genuinely exhaustive. Finally,
        # if no corpus frequencies were loaded, lexical guidance has no signal.
        if (
            clues
            or max_results <= 0
            or not candidates
            or not any(candidate.zipf > 0.0 for candidate in candidates)
            or hint_mode not in {"any", "exactly-one"}
        ):
            yield from fallback(
                remaining,
                candidates,
                min_words,
                max_words,
                max_results,
                allow_repeat,
                clue_words=clue_words,
                hint_mode=hint_mode,
                initial_clue_words=initial_clue_words,
                stats=stats,
                **kwargs,
            )
            return

        yield from quality_guided_bounded_solve(
            remaining,
            candidates,
            min_words,
            max_words,
            max_results,
            allow_repeat,
            stats=stats,
        )

    return solve
