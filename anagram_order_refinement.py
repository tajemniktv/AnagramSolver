"""Deterministic black-box k-opt refinement for complete word orders.

Beam search necessarily prunes partial orders using cheap local evidence. This
module gives a more expensive full-order scorer a bounded second chance: start
from complete beam seeds and repeatedly search 3/4/5-token permutation windows
for the best improving move. Exact search does not need this because it already
scores every permutation.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Order = tuple[str, ...]
OrderScorer = Callable[[Order], float]


@dataclass(slots=True, frozen=True)
class RefinementResult:
    order: Order
    score: float
    evaluated: int
    rounds: int
    improved: bool


def window_neighbors(
    order: Sequence[str],
    *,
    min_window: int = 3,
    max_window: int = 5,
) -> Iterable[Order]:
    """Yield deterministic unique local k-opt permutations of contiguous windows."""
    source = tuple(order)
    n = len(source)
    seen: set[Order] = set()
    for width in range(min_window, min(max_window, n) + 1):
        for start in range(n - width + 1):
            prefix = source[:start]
            window = source[start : start + width]
            suffix = source[start + width :]
            for replacement in itertools.permutations(window):
                if replacement == window:
                    continue
                candidate = (*prefix, *replacement, *suffix)
                if candidate == source or candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate


def refine_order(
    seed: Sequence[str],
    scorer: OrderScorer,
    *,
    min_window: int = 3,
    max_window: int = 5,
    max_rounds: int = 3,
    max_evaluations: int = 512,
    epsilon: float = 1e-12,
) -> RefinementResult:
    """Hill-climb from one complete order using best-improvement k-opt moves."""
    if min_window < 2:
        raise ValueError("min_window must be >= 2")
    if max_window < min_window:
        raise ValueError("max_window must be >= min_window")
    if max_rounds < 0:
        raise ValueError("max_rounds must be >= 0")
    if max_evaluations < 1:
        raise ValueError("max_evaluations must be >= 1")

    current = tuple(seed)
    current_score = float(scorer(current))
    evaluated = 1
    rounds = 0
    changed = False
    score_cache: dict[Order, float] = {current: current_score}

    while rounds < max_rounds and evaluated < max_evaluations:
        best_order = current
        best_score = current_score
        for candidate in window_neighbors(
            current,
            min_window=min_window,
            max_window=max_window,
        ):
            score = score_cache.get(candidate)
            if score is None:
                if evaluated >= max_evaluations:
                    break
                score = float(scorer(candidate))
                score_cache[candidate] = score
                evaluated += 1
            if score > best_score + epsilon or (
                abs(score - best_score) <= epsilon and candidate < best_order
            ):
                best_order = candidate
                best_score = score

        if best_order == current or best_score <= current_score + epsilon:
            break
        current = best_order
        current_score = best_score
        rounds += 1
        changed = True

    return RefinementResult(
        order=current,
        score=current_score,
        evaluated=evaluated,
        rounds=rounds,
        improved=changed,
    )


def refine_seed_pool(
    seeds: Sequence[Sequence[str]],
    scorer: OrderScorer,
    *,
    seed_limit: int = 8,
    min_window: int = 3,
    max_window: int = 5,
    max_rounds: int = 3,
    max_evaluations_per_seed: int = 512,
) -> tuple[RefinementResult, ...]:
    """Refine several complete seeds and deduplicate final orders best-first."""
    if seed_limit < 1:
        raise ValueError("seed_limit must be >= 1")
    by_order: dict[Order, RefinementResult] = {}
    for seed in seeds[:seed_limit]:
        result = refine_order(
            seed,
            scorer,
            min_window=min_window,
            max_window=max_window,
            max_rounds=max_rounds,
            max_evaluations=max_evaluations_per_seed,
        )
        previous = by_order.get(result.order)
        if previous is None or result.score > previous.score:
            by_order[result.order] = result
    return tuple(
        sorted(
            by_order.values(),
            key=lambda result: (-result.score, result.order),
        )
    )
