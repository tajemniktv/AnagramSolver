"""Deterministic structural diversification for retained word orders.

The grammar/structure objective remains the authority for the score-ranked head.
This module only decides which lower-ranked runner-up orders survive for later
positive corpus rescoring. A large fixed quality core keeps diversity from
turning the retained set into a novelty contest with English as collateral.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, SupportsIndex, TypeVar, overload

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


def _fingerprint_for_order(order: Sequence[str]) -> _OrderFingerprint:
    """Build an adjacency-only fingerprint for the legacy test helper."""
    realized = tuple(order)
    adjacency = Counter(pairwise(realized))
    return _OrderFingerprint(realized, adjacency, sum(adjacency.values()), "")


def _adjacency_overlap(
    left: _OrderFingerprint,
    right: _OrderFingerprint,
) -> float:
    """Directed-adjacency multiset overlap used by structural similarity."""
    if len(left.order) <= 1 or len(right.order) <= 1:
        return 1.0 if left.order == right.order else 0.0

    if len(left.adjacency) <= len(right.adjacency):
        smaller, larger = left.adjacency, right.adjacency
    else:
        smaller, larger = right.adjacency, left.adjacency
    overlap = sum(min(count, larger.get(edge, 0)) for edge, count in smaller.items())
    return overlap / max(left.adjacency_total, right.adjacency_total, 1)


def _adjacency_similarity(left: Sequence[str], right: Sequence[str]) -> float:
    """Compatibility helper for existing tests; production uses fingerprints."""
    return _adjacency_overlap(_fingerprint_for_order(left), _fingerprint_for_order(right))


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


def _select_diverse_orders_eager(
    candidates: Sequence[CandidateT],
    top_k: int,
    *,
    quality_core: int,
    diversity_strength: float,
) -> tuple[CandidateT, ...]:
    """Materialize the historical greedy diversity decision exactly once."""
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


class _DeferredDiverseOrders(tuple[CandidateT, ...]):
    """Tuple-compatible lazy view over a widened score-ranked order pool.

    The deep-analysis facade installs these objects in its side table immediately
    after workers return. Truth checks and access to candidate zero stay O(1),
    because diversity always preserves the score winner. All other tuple-visible
    operations resolve the retained set first, so callers can never observe the
    hidden widened pool through inherited tuple behavior. Serialization resolves
    to a plain tuple, preserving the historical public value contract.
    """

    _top_k: int
    _quality_core: int
    _diversity_strength: float
    _materialized: tuple[CandidateT, ...] | None

    def __new__(
        cls,
        candidates: Sequence[CandidateT],
        top_k: int,
        quality_core: int,
        diversity_strength: float,
    ) -> _DeferredDiverseOrders[CandidateT]:
        obj = super().__new__(cls, candidates)
        obj._top_k = top_k
        obj._quality_core = quality_core
        obj._diversity_strength = diversity_strength
        obj._materialized = None
        return obj

    def _raw(self) -> tuple[CandidateT, ...]:
        return tuple(tuple.__iter__(self))

    def _resolved(self) -> tuple[CandidateT, ...]:
        if self._materialized is None:
            self._materialized = _select_diverse_orders_eager(
                self._raw(),
                self._top_k,
                quality_core=self._quality_core,
                diversity_strength=self._diversity_strength,
            )
        return self._materialized

    @property
    def is_materialized(self) -> bool:
        """Expose lazy state for focused regression and performance tests."""
        return self._materialized is not None

    def __bool__(self) -> bool:
        return tuple.__len__(self) > 0

    def __len__(self) -> int:
        return len(self._resolved())

    def __iter__(self) -> Iterator[CandidateT]:
        return iter(self._resolved())

    @overload
    def __getitem__(self, key: SupportsIndex, /) -> CandidateT: ...

    @overload
    def __getitem__(self, key: slice, /) -> tuple[CandidateT, ...]: ...

    def __getitem__(
        self,
        key: SupportsIndex | slice,
        /,
    ) -> CandidateT | tuple[CandidateT, ...]:
        # Candidate zero is guaranteed to survive because quality_core >= 1.
        # The corpus-admission fast path relies on reading it without triggering
        # the expensive diversity pass for every deep row.
        if not isinstance(key, slice) and key.__index__() == 0 and tuple.__len__(self) > 0:
            return tuple.__getitem__(self, 0)
        return self._resolved()[key]

    def __contains__(self, item: object) -> bool:
        return item in self._resolved()

    def count(self, value: object, /) -> int:
        return self._resolved().count(value)

    def index(self, value: object, *args: SupportsIndex) -> int:
        if not args:
            return self._resolved().index(value)
        if len(args) == 1:
            return self._resolved().index(value, args[0].__index__())
        if len(args) == 2:
            return self._resolved().index(
                value,
                args[0].__index__(),
                args[1].__index__(),
            )
        raise TypeError(f"index expected at most 3 arguments, got {len(args) + 1}")

    def __add__(
        self,
        value: tuple[CandidateT, ...],
        /,
    ) -> tuple[CandidateT, ...]:
        return self._resolved() + value

    def __radd__(
        self,
        value: tuple[CandidateT, ...],
        /,
    ) -> tuple[CandidateT, ...]:
        return value + self._resolved()

    def __mul__(self, value: SupportsIndex, /) -> tuple[CandidateT, ...]:
        return self._resolved() * value.__index__()

    def __rmul__(self, value: SupportsIndex, /) -> tuple[CandidateT, ...]:
        return self._resolved() * value.__index__()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return self._resolved() == tuple(other)
        return False

    def __ne__(self, other: object) -> bool:
        if isinstance(other, tuple):
            return self._resolved() != tuple(other)
        return True

    def __lt__(self, other: tuple[object, ...], /) -> bool:
        return self._resolved() < other

    def __le__(self, other: tuple[object, ...], /) -> bool:
        return self._resolved() <= other

    def __gt__(self, other: tuple[object, ...], /) -> bool:
        return self._resolved() > other

    def __ge__(self, other: tuple[object, ...], /) -> bool:
        return self._resolved() >= other

    def __hash__(self) -> int:
        return hash(self._resolved())

    def __reduce_ex__(
        self,
        protocol: SupportsIndex,
    ) -> tuple[object, tuple[tuple[CandidateT, ...]]]:
        del protocol
        return tuple, (self._resolved(),)

    def __repr__(self) -> str:
        return repr(self._resolved())


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
    instead of rescanning the entire selected prefix. The final retained order
    remains identical to the eager algorithm.

    When a widened pool actually needs diversity, the returned value is a
    tuple-compatible deferred view. Ordinary score-prefix cases stay eager.
    This moves parent-side diversity work out of the deep-worker stage and into
    the point where alternative orders are genuinely consumed.
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
    if core_count >= limit or len(candidates) <= limit:
        return tuple(candidates[index] for index in range(limit))

    # The facade widens only the runner-up tail. Deferring this exact selection
    # means its post-worker table rewrite is constant-time; consumers still see
    # precisely the same retained tuple once they need more than the winner.
    return _DeferredDiverseOrders(
        candidates,
        top_k,
        quality_core,
        diversity_strength,
    )
