"""Deterministic structural diversification for retained word orders.

The grammar/structure objective remains the authority for the score-ranked head.
This module only decides which lower-ranked runner-up orders survive for later
positive corpus rescoring. A large fixed quality core keeps diversity from
turning the retained set into a novelty contest with English as collateral.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
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


@dataclass(slots=True)
class _OrderFingerprint:
    """Precomputed structural facts reused by the greedy diversity pass."""

    order: tuple[str, ...]
    adjacency: Counter[tuple[str, str]]
    adjacency_total: int
    phrase_kind: str


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


def _fingerprint(candidate: OrderLike) -> _OrderFingerprint:
    adjacency = Counter(pairwise(candidate.order))
    return _OrderFingerprint(
        order=candidate.order,
        adjacency=adjacency,
        adjacency_total=sum(adjacency.values()),
        phrase_kind=candidate.phrase_kind,
    )


def _adjacency_overlap(
    left: _OrderFingerprint,
    right: _OrderFingerprint,
) -> float:
    """Directed-adjacency multiset overlap used by structural similarity."""
    if len(left.order) <= 1 or len(right.order) <= 1:
        return 1.0 if left.order == right.order else 0.0

    # Iterate the smaller precomputed mapping. This is equivalent to Counter
    # intersection and preserves repeated-token multiplicity through min counts.
    if len(left.adjacency) <= len(right.adjacency):
        smaller, larger = left.adjacency, right.adjacency
    else:
        smaller, larger = right.adjacency, left.adjacency
    overlap = sum(min(count, larger.get(edge, 0)) for edge, count in smaller.items())
    return overlap / max(left.adjacency_total, right.adjacency_total, 1)


def _fingerprint_similarity(
    left: _OrderFingerprint,
    right: _OrderFingerprint,
) -> float:
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
    adjacency = _adjacency_overlap(left, right)

    similarity = (
        0.55 * adjacency
        + 0.30 * position
        + 0.10 * endpoints
        + 0.05 * same_kind
    )
    return max(0.0, min(1.0, similarity))


def order_similarity(left: OrderLike, right: OrderLike) -> float:
    """Structural similarity in [0, 1] between two realized word orders.

    Directed adjacency receives the most weight because local word relations
    are precisely where near-duplicate permutations tend to cluster. Position,
    sentence endpoints, and the hand-written parser's construction kind provide
    progressively weaker signals. Counter-based adjacency keeps repeated words
    well-defined instead of pretending every token is unique.
    """
    return _fingerprint_similarity(_fingerprint(left), _fingerprint(right))


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
    slots use the same maximal-marginal-relevance objective as before, but cache
    each candidate's current maximum similarity to the selected set. Adding one
    winner therefore requires only one new comparison per remaining candidate
    instead of rescanning the entire selected prefix. The final tuple remains in
    original score order, preserving downstream ranking and deterministic ties.
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
    if len(selected_indices) >= limit or not remaining:
        return tuple(candidates[index] for index in selected_indices)

    fingerprints = [_fingerprint(candidate) for candidate in candidates]
    max_similarity: dict[int, float] = {}
    for index in remaining:
        max_similarity[index] = max(
            _fingerprint_similarity(fingerprints[index], fingerprints[chosen])
            for chosen in selected_indices
        )

    while len(selected_indices) < limit and remaining:
        best_index: int | None = None
        best_key: tuple[float, float, tuple[str, ...]] | None = None

        for index in remaining:
            candidate = candidates[index]
            utility = candidate.objective - diversity_strength * max_similarity[index]
            # min() semantics encoded explicitly: highest utility/objective first,
            # lexical order as the deterministic final tie break.
            key = (-utility, -candidate.objective, candidate.order)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index

        assert best_index is not None
        selected_indices.append(best_index)
        remaining.remove(best_index)
        max_similarity.pop(best_index, None)

        if len(selected_indices) >= limit:
            break

        winner = fingerprints[best_index]
        for index in remaining:
            similarity = _fingerprint_similarity(fingerprints[index], winner)
            if similarity > max_similarity[index]:
                max_similarity[index] = similarity

    selected_indices.sort()
    return tuple(candidates[index] for index in selected_indices)
