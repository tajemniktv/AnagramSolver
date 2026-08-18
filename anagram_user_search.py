"""Quality-guided bounded exact-bag search for normal solver runs.

The low-level generator keeps its historical DFS for research, clues, and true
exhaustive enumeration. Normal bounded searches need a different contract: the
candidate cap should retain plausible word bags rather than whichever exact
covers happen to occur first in DFS order. This module keeps the best exact bags
per word-count bucket using an admissible lexical upper bound and bounded result
heaps, so memory grows with the requested shortlist rather than the search tree.
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
_EPSILON = 1e-12


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
    """Admissible lexical upper bound for a partial bag.

    The lexical average and low-tail terms are monotone in each Zipf value. By
    filling every unknown slot with the best currently feasible Zipf score and
    omitting non-positive duplicate/junk penalties, this can overestimate but
    cannot underestimate any completion beneath the state.
    """
    values = [candidates[index].zipf for index in chosen_indices]
    values.extend([best_future_zipf] * words_left)
    if not values:
        return 0.0
    average = sum(values) / len(values)
    low_count = min(2, len(values))
    low_tail = sum(sorted(values)[:low_count]) / low_count
    return 0.78 * average + 0.22 * low_tail


def _top_bags_for_word_count(
    remaining: tuple[int, ...],
    candidates: list[generator.Candidate],
    word_count: int,
    limit: int,
    allow_repeat: bool,
) -> tuple[list[tuple[str, ...]], int]:
    if limit <= 0 or not candidates:
        return [], 0

    min_candidate_len = min(candidate.length for candidate in candidates)
    max_candidate_len = max(candidate.length for candidate in candidates)
    remaining_len = sum(remaining)
    if remaining_len < word_count * min_candidate_len:
        return [], 0
    if remaining_len > word_count * max_candidate_len:
        return [], 0

    sparse_signatures = [_sparse_signature(candidate.sig) for candidate in candidates]
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_signature[candidate.sig].append(index)

    # Min-heap of retained complete bags. The score is the pruning authority;
    # serial keeps equal-score retention deterministic in DFS discovery order.
    best: list[tuple[float, int, tuple[int, ...]]] = []
    serial = 0
    exact_examined = 0

    def retain(indices: tuple[int, ...]) -> None:
        nonlocal serial, exact_examined
        exact_examined += 1
        score = _lexical_score(indices, candidates)
        item = (score, -serial, indices)
        serial += 1
        if len(best) < limit:
            heapq.heappush(best, item)
            return
        if score > best[0][0] + _EPSILON:
            heapq.heapreplace(best, item)

    def best_feasible_zipf(
        rem: tuple[int, ...],
        rem_len: int,
        start: int,
        words_left: int,
    ) -> float | None:
        if words_left <= 0:
            return 0.0 if rem_len == 0 else None
        min_this_len = max(
            min_candidate_len,
            rem_len - (words_left - 1) * max_candidate_len,
        )
        max_this_len = min(
            max_candidate_len,
            rem_len - (words_left - 1) * min_candidate_len,
        )
        for index in range(start, len(candidates)):
            candidate = candidates[index]
            if candidate.length < min_this_len or candidate.length > max_this_len:
                continue
            if _fits_sparse(sparse_signatures[index], rem):
                return candidate.zipf
        return None

    def dfs(
        rem: tuple[int, ...],
        rem_len: int,
        start: int,
        words_left: int,
        chosen_indices: tuple[int, ...],
    ) -> None:
        if words_left == 0:
            if rem_len == 0:
                retain(chosen_indices)
            return
        if rem_len == 0:
            return
        if rem_len < words_left * min_candidate_len:
            return
        if rem_len > words_left * max_candidate_len:
            return

        future_zipf = best_feasible_zipf(rem, rem_len, start, words_left)
        if future_zipf is None:
            return
        if len(best) >= limit:
            upper = _optimistic_score(
                chosen_indices,
                words_left,
                future_zipf,
                candidates,
            )
            if upper <= best[0][0] + _EPSILON:
                return

        if words_left == 1:
            for index in by_signature.get(rem, []):
                if index < start:
                    continue
                retain((*chosen_indices, index))
            return

        min_this_len = max(
            min_candidate_len,
            rem_len - (words_left - 1) * max_candidate_len,
        )
        max_this_len = min(
            max_candidate_len,
            rem_len - (words_left - 1) * min_candidate_len,
        )
        for index in range(start, len(candidates)):
            candidate = candidates[index]
            if candidate.length < min_this_len or candidate.length > max_this_len:
                continue
            sparse = sparse_signatures[index]
            if not _fits_sparse(sparse, rem):
                continue
            new_rem = _subtract_sparse(rem, sparse)
            next_start = index if allow_repeat else index + 1
            dfs(
                new_rem,
                rem_len - candidate.length,
                next_start,
                words_left - 1,
                (*chosen_indices, index),
            )

    dfs(remaining, remaining_len, 0, word_count, ())
    ranked = sorted(
        best,
        key=lambda item: (-item[0], item[2]),
    )
    return [
        tuple(candidates[index].word for index in indices)
        for _, _, indices in ranked
    ], exact_examined


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
    """Yield a balanced lexical shortlist across requested word-count buckets."""
    if max_results <= 0:
        return

    search_stats = stats if stats is not None else generator.SearchStats()
    remaining_budget = max_results
    word_counts = tuple(range(min_words, max_words + 1))
    total_examined = 0
    bucket_results: dict[int, list[tuple[str, ...]]] = {}
    bucket_limits: dict[int, int] = {}

    # First pass rolls sparse early buckets forward immediately. This is cheap
    # for common 2/3-word cases and reserves meaningful space for longer bags.
    for position, word_count in enumerate(word_counts):
        if remaining_budget <= 0:
            bucket_results[word_count] = []
            bucket_limits[word_count] = 0
            continue
        buckets_left = len(word_counts) - position
        quota = max(1, math.ceil(remaining_budget / buckets_left))
        bags, examined = _top_bags_for_word_count(
            remaining,
            candidates,
            word_count,
            quota,
            allow_repeat,
        )
        total_examined += examined
        bucket_results[word_count] = bags
        bucket_limits[word_count] = quota
        remaining_budget -= len(bags)

    # If later buckets were sparse, spend their unused reservation on an earlier
    # bucket that proved it had at least its full quota. Re-running only happens
    # in this spillover case and keeps the final total at the requested cap when
    # enough exact bags exist anywhere in the requested range.
    if remaining_budget > 0:
        for word_count in word_counts:
            if remaining_budget <= 0:
                break
            previous = bucket_results.get(word_count, [])
            previous_limit = bucket_limits.get(word_count, 0)
            if previous_limit <= 0 or len(previous) < previous_limit:
                continue
            expanded_limit = previous_limit + remaining_budget
            expanded, examined = _top_bags_for_word_count(
                remaining,
                candidates,
                word_count,
                expanded_limit,
                allow_repeat,
            )
            total_examined += examined
            added = max(0, len(expanded) - len(previous))
            if added <= 0:
                continue
            take = min(added, remaining_budget)
            bucket_results[word_count] = expanded[: len(previous) + take]
            bucket_limits[word_count] = expanded_limit
            remaining_budget -= take

    retained_total = 0
    for word_count in word_counts:
        for bag in bucket_results.get(word_count, ()):
            search_stats.exact_examined += 1
            search_stats.accepted += 1
            retained_total += 1
            yield bag

    print(
        f"Quality-guided bounded search examined {total_examined:,} exact bag(s); "
        f"retained {retained_total:,} across {min_words}-{max_words} words.",
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
