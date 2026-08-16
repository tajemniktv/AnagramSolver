"""Deterministic structural diversification for retained word orders.

The grammar/structure objective remains the authority for the score-ranked head.
This module only decides which lower-ranked runner-up orders survive for later
positive corpus rescoring. A large fixed quality core keeps diversity from
turning the retained set into a novelty contest with English as collateral.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Protocol, TypeVar

DEFAULT_QUALITY_CORE = 48
DEFAULT_POOL_EXTRA = 8
DEFAULT_MAX_POOL = 128
DEFAULT_DIVERSITY_STRENGTH = 0.12


class OrderLike(Protocol):
    order: tuple[str, ...]
    objective: float
    phrase_kind: str


CandidateT = TypeVar("CandidateT", bound=OrderLike)


def raw_pool_size(
    retained: int,
    *,
    quality_core: int = DEFAULT_QUALITY_CORE,
    pool_extra: int = DEFAULT_POOL_EXTRA,
    max_pool: int = DEFAULT_MAX_POOL,
) -> int:
    """Return the score-ranked pool size needed before diversity selection."""
    if retained < 1:
        raise ValueError("retained must be >= 1")
    if quality_core < 1:
        raise ValueError("quality_core must be >= 1")
    if pool_extra < 0:
        raise ValueError("pool_extra must be >= 0")
    if max_pool < 1:
        raise ValueError("max_pool must be >= 1")

    # Explicit K at or below the quality core remains score-only and avoids the
    # cost of a wider search. Above it, spend a bounded fixed number of extra
    # raw slots on structural alternatives rather than multiplying beam cost as
    # K grows. If callers deliberately request more than max_pool, honor their K
    # and simply stop widening beyond it.
    if retained <= quality_core:
        return retained
    return max(retained, min(max_pool, retained + pool_extra))


def _adjacency_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) <= 1 or len(right) <= 1:
        return 1.0 if tuple(left) == tuple(right) else 0.0

    left_pairs = Counter(zip(left, left[1:]))
    right_pairs = Counter(zip(right, right[1:]))
    overlap = sum((left_pairs & right_pairs).values())
    return overlap / max(sum(left_pairs.values()), sum(right_pairs.values()), 1)


def order_similarity(left: OrderLike, right: OrderLike) -> float:
    """Structural similarity in [0, 1] between two realized word orders.

    Directed adjacency receives the most weight because local word relations
    are precisely where near-duplicate permutations tend to cluster. Position,
    sentence endpoints, and the hand-written parser's construction kind provide
    progressively weaker signals. Counter-based adjacency keeps repeated words
    well-defined instead of pretending every token is unique.
    """
    a = left.order
    b = right.order
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    common_positions = sum(x == y for x, y in zip(a, b))
    position = common_positions / max(len(a), len(b))
    endpoints = 0.5 * float(a[0] == b[0]) + 0.5 * float(a[-1] == b[-1])
    same_kind = float(left.phrase_kind == right.phrase_kind)

    similarity = (
        0.55 * _adjacency_similarity(a, b)
        + 0.30 * position
        + 0.10 * endpoints
        + 0.05 * same_kind
    )
    return max(0.0, min(1.0, similarity))


def select_diverse_orders(
    candidates: Sequence[CandidateT],
    top_k: int,
    *,
    quality_core: int = DEFAULT_QUALITY_CORE,
    diversity_strength: float = DEFAULT_DIVERSITY_STRENGTH,
) -> tuple[CandidateT, ...]:
    """Keep the score head, then greedily spend remaining slots on novelty.

    ``candidates`` must already be in best-first grammar/structure order. The
    first ``quality_core`` candidates are preserved byte-for-byte. Remaining
    slots use a small maximal-marginal-relevance penalty against the most
    similar already-selected order. The final tuple is returned in the original
    score order so downstream grammar ranking and deterministic tie behaviour do
    not change merely because retention became more diverse.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if quality_core < 1:
        raise ValueError("quality_core must be >= 1")
    if not 0.0 <= diversity_strength <= 1.0:
        raise ValueError("diversity_strength must be between 0 and 1")
    if not candidates:
        return ()

    limit = min(top_k, len(candidates))
    core_count = min(quality_core, limit)
    selected_indices = list(range(core_count))
    remaining = set(range(core_count, len(candidates)))

    while len(selected_indices) < limit and remaining:
        best_index: int | None = None
        best_key: tuple[float, float, tuple[str, ...]] | None = None

        for index in remaining:
            candidate = candidates[index]
            max_similarity = max(
                order_similarity(candidate, candidates[chosen])
                for chosen in selected_indices
            )
            utility = candidate.objective - diversity_strength * max_similarity
            # min() semantics encoded explicitly: highest utility/objective first,
            # lexical order as the deterministic final tie break.
            key = (-utility, -candidate.objective, candidate.order)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index

        assert best_index is not None
        selected_indices.append(best_index)
        remaining.remove(best_index)

    selected_indices.sort()
    return tuple(candidates[index] for index in selected_indices)
