"""Quality-guided bounded exact-bag search for normal solver runs.

The low-level generator keeps its historical DFS for research, clues, and true
exhaustive enumeration. Normal bounded searches need a different contract: the
candidate cap should retain plausible word bags rather than whichever exact
covers happen to occur first in DFS order.

Normal bounded search therefore keeps several cheap views of the same partial
and completed bags: unigram lexical quality, observed pair/collocation evidence,
rare-word anchors, and bounded rare-pair context champions. The deeper linguistic
reranker remains the final authority.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anagram_generate as generator

SolveCallable = Callable[..., Iterator[tuple[str, ...]]]
BeamState = tuple[tuple[int, ...], int, int, tuple[int, ...]]
ScoredItem = tuple[float, int, tuple[Any, ...]]
PairItem = tuple[float, float, float, int, tuple[Any, ...]]
ANCHOR_CHAMPIONS_PER_WORD = 2
CONTEXT_CHAMPIONS_PER_GROUP = 1
LEXICAL_SHARE = 0.55
PAIR_SHARE = 0.35
CONTEXT_DIVERSITY_SHARE = 0.60


@dataclass(frozen=True, slots=True)
class BeamExpansion:
    """One valid branch from a partial beam state."""

    index: int
    remaining: tuple[int, ...]
    remaining_len: int
    next_start: int
    chosen: tuple[int, ...]


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
    """Return an optimistic unigram score for a partial bag."""
    values = [candidates[index].zipf for index in chosen_indices]
    values.extend([best_future_zipf] * words_left)
    if not values:
        return 0.0
    average = sum(values) / len(values)
    low_count = min(2, len(values))
    low_tail = sum(sorted(values)[:low_count]) / low_count
    return 0.78 * average + 0.22 * low_tail


def _pair_priority(
    indices: tuple[int, ...],
    candidates: list[generator.Candidate],
    bigrams: generator.BigramModel | None,
) -> tuple[float, float, float]:
    """Return an order-independent collocation priority for a partial/full bag."""
    lexical = _lexical_score(indices, candidates)
    if bigrams is None or len(indices) < 2:
        return 0.0, -99.0, lexical
    words = tuple(candidates[index].word for index in indices)
    pair_raw, coverage = generator.bigram_pair_potential(words, bigrams)
    # Observed-edge coverage is deliberately first. Search only needs to retain
    # candidates with evidence that their words can connect; the deeper reranker
    # later decides whether those connections form a valid sentence.
    return coverage, pair_raw, lexical


def _rare_context_key(
    indices: tuple[int, ...],
    candidates: list[generator.Candidate],
) -> tuple[int, ...]:
    """Group partial bags by their one/two least-common selected words.

    This is deliberately generic context diversity rather than a phrase-specific
    rule. A global beam can otherwise fill with many variants sharing common
    words and erase a useful low-frequency pair before exact closure. Keeping one
    champion for each rare-word context gives structurally different partial bags
    a bounded route forward without widening the global beam.
    """
    if not indices:
        return ()
    rarest = sorted(
        indices,
        key=lambda index: (
            candidates[index].zipf,
            candidates[index].word,
            index,
        ),
    )[:2]
    return tuple(sorted(rarest))


def _load_search_bigram_model(
    candidates: list[generator.Candidate],
    *,
    unigrams: generator.UnigramModel | None = None,
    ngram_dir: Path | str = generator.DEFAULT_NGRAM_DIR,
    refresh: bool = False,
) -> generator.BigramModel:
    """Load only pair rows relevant to this puzzle's candidate vocabulary."""
    one_path, two_path = generator.ensure_ngram_data(
        Path(ngram_dir).expanduser(),
        refresh=refresh,
        need_bigrams=True,
    )
    if two_path is None:
        raise RuntimeError("Bigram n-gram data unavailable for quality-guided search")
    active_unigrams = unigrams if unigrams is not None else generator.load_unigram_model(one_path)
    return generator.load_bigram_model(
        two_path,
        active_unigrams,
        {candidate.word for candidate in candidates},
    )


def _bucket_result_cap(word_count: int, requested: int) -> int:
    """Bound how much cheap material a bucket sends to deep reranking."""
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


def _push_pair_bounded(
    heap: list[PairItem],
    item: PairItem,
    limit: int,
) -> None:
    if limit <= 0:
        return
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item[:4] > heap[0][:4]:
        heapq.heapreplace(heap, item)


def _view_quotas(limit: int) -> tuple[int, int]:
    if limit <= 1:
        return max(limit, 0), 0
    lexical = max(1, int(limit * LEXICAL_SHARE))
    pair = max(1, int(limit * PAIR_SHARE))
    if lexical + pair > limit:
        pair = max(0, limit - lexical)
    return lexical, pair


def _select_multi_view(
    lexical_heap: list[ScoredItem],
    pair_heap: list[PairItem],
    anchor_heaps: dict[int, list[ScoredItem]],
    limit: int,
    *,
    context_heaps: dict[tuple[int, ...], list[PairItem]] | None = None,
) -> list[tuple[Any, ...]]:
    """Combine quality and bounded structural-context views without duplicates."""
    lexical = sorted(lexical_heap, reverse=True)
    pair = sorted(pair_heap, reverse=True)
    anchors = sorted(
        (item for heap in anchor_heaps.values() for item in heap),
        reverse=True,
    )
    contexts = sorted(
        (
            item
            for heap in (context_heaps or {}).values()
            for item in heap
        ),
        reverse=True,
    )
    lexical_quota, pair_quota = _view_quotas(limit)

    selected: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()

    def add_payload(payload: tuple[Any, ...]) -> bool:
        if payload in seen or len(selected) >= limit:
            return False
        selected.append(payload)
        seen.add(payload)
        return True

    lexical_added = 0
    for _, _, payload in lexical:
        if lexical_added >= lexical_quota:
            break
        if add_payload(payload):
            lexical_added += 1

    pair_added = 0
    for _, _, _, _, payload in pair:
        if pair_added >= pair_quota:
            break
        if add_payload(payload):
            pair_added += 1

    diversity_slots = max(0, limit - lexical_quota - pair_quota)
    if contexts:
        context_quota = max(1, math.ceil(diversity_slots * CONTEXT_DIVERSITY_SHARE))
        context_quota = min(context_quota, diversity_slots)
    else:
        context_quota = 0
    anchor_quota = diversity_slots - context_quota

    context_added = 0
    for _, _, _, _, payload in contexts:
        if context_added >= context_quota:
            break
        if add_payload(payload):
            context_added += 1

    anchor_added = 0
    for _, _, payload in anchors:
        if anchor_added >= anchor_quota:
            break
        if add_payload(payload):
            anchor_added += 1

    # Diversity champions often overlap the quality cores. Fill resulting holes
    # from every view instead of silently returning fewer states than budgeted.
    if len(selected) < limit:
        for _, _, _, _, payload in contexts:
            if len(selected) >= limit:
                break
            add_payload(payload)
    if len(selected) < limit:
        for _, _, payload in anchors:
            if len(selected) >= limit:
                break
            add_payload(payload)
    if len(selected) < limit:
        for _, _, payload in lexical:
            if len(selected) >= limit:
                break
            add_payload(payload)
    if len(selected) < limit:
        for _, _, _, _, payload in pair:
            if len(selected) >= limit:
                break
            add_payload(payload)

    return selected


def _iter_state_expansions(
    rem: tuple[int, ...],
    rem_len: int,
    start: int,
    chosen: tuple[int, ...],
    *,
    candidates: list[generator.Candidate],
    sparse_signatures: list[tuple[tuple[int, int], ...]],
    by_signature: dict[tuple[int, ...], list[int]],
    min_candidate_len: int,
    max_candidate_len: int,
    words_left_after: int,
    allow_repeat: bool,
    branch_limit: int,
) -> Iterator[BeamExpansion]:
    """Yield exactly the branches considered by one beam state.

    ``generator.load_words`` orders candidates by descending Zipf score, then
    descending length and lexical word order. The beam relies on that contract:
    ``candidates[next_start].zipf`` is an optimistic frequency bound for the
    monotonic candidate suffix, and the branch limit keeps its best-frequency
    feasible prefix.
    """
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
        if words_left_after == 1 and not any(
            last_index >= next_start
            for last_index in by_signature.get(new_rem, ())
        ):
            continue

        yield BeamExpansion(
            index=index,
            remaining=new_rem,
            remaining_len=new_rem_len,
            next_start=next_start,
            chosen=(*chosen, index),
        )
        accepted_branches += 1
        if accepted_branches >= branch_limit:
            return


def _beam_bags_for_word_count(
    remaining: tuple[int, ...],
    candidates: list[generator.Candidate],
    word_count: int,
    limit: int,
    allow_repeat: bool,
    bigrams: generator.BigramModel | None,
) -> tuple[list[tuple[str, ...]], int, int]:
    """Return a deterministic multi-view beam shortlist for one word count."""
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
    branch_limit = _branch_width(word_count)
    partial_expansions = 0
    serial = 0

    # Choose every word except the final exact signature closure. The final step
    # is handled through by_signature so rare-but-required last words are never
    # lost merely because they fall outside the local branch width.
    for depth in range(max(0, word_count - 1)):
        words_left_before = word_count - depth
        words_left_after = words_left_before - 1
        lexical_heap: list[ScoredItem] = []
        pair_heap: list[PairItem] = []
        anchor_heaps: dict[int, list[ScoredItem]] = defaultdict(list)
        context_heaps: dict[tuple[int, ...], list[PairItem]] = defaultdict(list)

        for rem, rem_len, start, chosen in states:
            for expansion in _iter_state_expansions(
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
                allow_repeat=allow_repeat,
                branch_limit=branch_limit,
            ):
                new_chosen = expansion.chosen
                if words_left_after:
                    best_future_zipf = candidates[expansion.next_start].zipf
                    lexical_priority = _optimistic_score(
                        new_chosen,
                        words_left_after,
                        best_future_zipf,
                        candidates,
                    )
                else:
                    lexical_priority = _lexical_score(new_chosen, candidates)

                payload: tuple[Any, ...] = (
                    expansion.remaining,
                    expansion.remaining_len,
                    expansion.next_start,
                    new_chosen,
                )
                lexical_item: ScoredItem = (lexical_priority, -serial, payload)
                pair_priority = _pair_priority(new_chosen, candidates, bigrams)
                pair_item: PairItem = (*pair_priority, -serial, payload)
                serial += 1
                partial_expansions += 1
                _push_bounded(lexical_heap, lexical_item, width)
                _push_pair_bounded(pair_heap, pair_item, width)
                _push_bounded(
                    anchor_heaps[new_chosen[-1]],
                    lexical_item,
                    ANCHOR_CHAMPIONS_PER_WORD,
                )
                _push_pair_bounded(
                    context_heaps[_rare_context_key(new_chosen, candidates)],
                    pair_item,
                    CONTEXT_CHAMPIONS_PER_GROUP,
                )

        selected = _select_multi_view(
            lexical_heap,
            pair_heap,
            anchor_heaps,
            width,
            context_heaps=context_heaps,
        )
        if not selected:
            return [], 0, partial_expansions
        states = [
            (payload[0], payload[1], payload[2], payload[3])
            for payload in selected
        ]

    exact_examined = 0
    result_lexical_heap: list[ScoredItem] = []
    result_pair_heap: list[PairItem] = []
    result_anchor_heaps: dict[int, list[ScoredItem]] = defaultdict(list)
    result_context_heaps: dict[tuple[int, ...], list[PairItem]] = defaultdict(list)
    result_serial = 0

    if word_count == 1:
        states = [(remaining, remaining_len, 0, ())]

    for rem, _rem_len, start, chosen in states:
        for last_index in by_signature.get(rem, ()):
            if last_index < start:
                continue
            indices = (*chosen, last_index)
            exact_examined += 1
            lexical_score = _lexical_score(indices, candidates)
            payload: tuple[Any, ...] = (indices,)
            lexical_item: ScoredItem = (lexical_score, -result_serial, payload)
            pair_priority = _pair_priority(indices, candidates, bigrams)
            pair_item: PairItem = (*pair_priority, -result_serial, payload)
            result_serial += 1
            _push_bounded(result_lexical_heap, lexical_item, limit)
            _push_pair_bounded(result_pair_heap, pair_item, limit)
            _push_bounded(
                result_anchor_heaps[indices[-1]],
                lexical_item,
                ANCHOR_CHAMPIONS_PER_WORD,
            )
            _push_pair_bounded(
                result_context_heaps[_rare_context_key(indices, candidates)],
                pair_item,
                CONTEXT_CHAMPIONS_PER_GROUP,
            )

    selected_results = _select_multi_view(
        result_lexical_heap,
        result_pair_heap,
        result_anchor_heaps,
        limit,
        context_heaps=result_context_heaps,
    )
    bags = [
        tuple(candidates[index].word for index in payload[0])
        for payload in selected_results
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
    bigrams: generator.BigramModel | None = None,
) -> Iterator[tuple[str, ...]]:
    """Yield a globally capped multi-view shortlist across requested word counts."""
    if max_results <= 0:
        return

    search_stats = stats if stats is not None else generator.SearchStats()
    word_counts = tuple(range(min_words, max_words + 1))
    if not word_counts:
        return

    bucket_results: dict[int, list[tuple[str, ...]]] = {}
    total_examined = 0
    remaining_budget = max_results

    for position, word_count in enumerate(word_counts):
        if remaining_budget <= 0:
            break
        buckets_left = len(word_counts) - position
        fair_share = max(1, math.ceil(remaining_budget / buckets_left))
        result_limit = min(
            remaining_budget,
            _bucket_result_cap(word_count, fair_share),
        )
        bags, examined, _expansions = _beam_bags_for_word_count(
            remaining,
            candidates,
            word_count,
            result_limit,
            allow_repeat,
            bigrams,
        )
        accepted = bags[:remaining_budget]
        bucket_results[word_count] = accepted
        total_examined += examined
        remaining_budget -= len(accepted)

    retained_total = sum(len(bags) for bags in bucket_results.values())
    search_stats.exact_examined += total_examined
    search_stats.accepted += retained_total

    for word_count in word_counts:
        yield from bucket_results.get(word_count, ())


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
        unigrams: generator.UnigramModel | None = None,
        ngram_dir: Path | str = generator.DEFAULT_NGRAM_DIR,
        refresh: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[str, ...]]:
        clues = clue_words or set()
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
                unigrams=unigrams,
                ngram_dir=ngram_dir,
                refresh=refresh,
                **kwargs,
            )
            return

        bigrams = _load_search_bigram_model(
            candidates,
            unigrams=unigrams,
            ngram_dir=ngram_dir,
            refresh=refresh,
        )
        yield from quality_guided_bounded_solve(
            remaining,
            candidates,
            min_words,
            max_words,
            max_results,
            allow_repeat,
            stats=stats,
            bigrams=bigrams,
        )

    return solve
