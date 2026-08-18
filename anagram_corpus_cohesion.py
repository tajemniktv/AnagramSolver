"""Generic corpus-span cohesion evidence for ordered anagram candidates.

The phrase index already tells us whether individual n-grams are attested. This
module asks a different question: how economically can an entire candidate order
be explained by non-overlapping attested spans?

The dynamic program is intentionally small and deterministic. It rewards
coverage and long spans, charges for every independent corpus fragment, and
uses frequency only as a bounded tie-strength signal. Missing corpus evidence
is neutral rather than negative.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

CountsLookup = Callable[[Sequence[str]], Mapping[str, int]]


@dataclass(slots=True, frozen=True)
class CohesionSpan:
    """One positive corpus span chosen by the best segmentation."""

    start: int
    end: int
    text: str
    count: int
    strength: float

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True, frozen=True)
class CohesionResult:
    """Normalized evidence produced by the corpus segmentation."""

    score: float
    coverage: float
    longest_fraction: float
    segments: int
    splice_penalty: float
    frequency_strength: float
    spans: tuple[CohesionSpan, ...]


@dataclass(slots=True, frozen=True)
class _Path:
    utility: float
    covered: int
    frequency_sum: float
    spans: tuple[CohesionSpan, ...]

    @property
    def segments(self) -> int:
        return len(self.spans)

    @property
    def longest(self) -> int:
        return max((span.length for span in self.spans), default=0)


def _path_key(path: _Path) -> tuple[float, int, int, int, float, tuple[str, ...]]:
    """Best-first deterministic ordering for alternative segmentations."""
    return (
        path.utility,
        path.covered,
        -path.segments,
        path.longest,
        path.frequency_sum,
        tuple(span.text for span in path.spans),
    )


def _frequency_strength(count: int) -> float:
    """Bound corpus frequency so giant sources cannot dominate segmentation."""
    if count <= 0:
        return 0.0
    return min(1.0, math.log10(count + 1.0) / 5.0)


def score_corpus_cohesion(
    words: Sequence[str],
    *,
    counts: CountsLookup,
    max_n: int,
) -> CohesionResult:
    """Return positive-only cohesion evidence for one complete word order.

    All positive corpus spans of length >= 2 are considered. A DP then chooses
    a non-overlapping explanation that balances covered words against a fixed
    per-fragment cost. That makes one long attested expression preferable to a
    "Frankenphrase" assembled from many unrelated short hits, while still
    allowing two strong chunks to rescue an otherwise unseen full phrase.

    Empty tokens are malformed solver orders. They are rejected as neutral
    evidence rather than removed, because removing one could create a false
    adjacency such as ``("foo", "", "bar") -> "foo bar"``.
    """
    ordered = tuple(words)
    n = len(ordered)
    if n < 2 or max_n < 2 or any(not word for word in ordered):
        return CohesionResult(0.0, 0.0, 0.0, 0, 0.0, 0.0, ())

    span_text: dict[tuple[int, int], str] = {}
    queries: list[str] = []
    upper = min(max_n, n)
    for length in range(2, upper + 1):
        for start in range(n - length + 1):
            end = start + length
            text = " ".join(ordered[start:end])
            span_text[(start, end)] = text
            queries.append(text)

    hit_counts = counts(tuple(queries))
    by_start: dict[int, list[CohesionSpan]] = {}
    for (start, end), text in span_text.items():
        count = int(hit_counts.get(text, 0))
        if count <= 0:
            continue
        span = CohesionSpan(
            start=start,
            end=end,
            text=text,
            count=count,
            strength=_frequency_strength(count),
        )
        by_start.setdefault(start, []).append(span)

    if not by_start:
        return CohesionResult(0.0, 0.0, 0.0, 0, 0.0, 0.0, ())

    # A covered word is worth 1.0 utility. Each independent corpus fragment
    # pays 0.55, so full coverage by two good chunks can beat one almost-full
    # chunk, but gratuitous chains of bigrams quickly become unattractive.
    fragment_cost = 0.55
    frequency_weight = 0.12
    dp: list[_Path | None] = [None] * (n + 1)
    dp[0] = _Path(0.0, 0, 0.0, ())

    for position in range(n):
        current = dp[position]
        if current is None:
            continue

        # Corpus absence is neutral: skipping one uncovered token has no cost.
        skipped = current
        existing = dp[position + 1]
        if existing is None or _path_key(skipped) > _path_key(existing):
            dp[position + 1] = skipped

        for span in by_start.get(position, ()):
            length = span.length
            candidate = _Path(
                utility=(
                    current.utility
                    + float(length)
                    - fragment_cost
                    + frequency_weight * span.strength
                ),
                covered=current.covered + length,
                frequency_sum=current.frequency_sum + span.strength,
                spans=(*current.spans, span),
            )
            previous = dp[span.end]
            if previous is None or _path_key(candidate) > _path_key(previous):
                dp[span.end] = candidate

    best = dp[n]
    if best is None or not best.spans:
        return CohesionResult(0.0, 0.0, 0.0, 0, 0.0, 0.0, ())

    coverage = best.covered / n
    longest_fraction = best.longest / n
    segments = best.segments
    frequency = best.frequency_sum / segments
    splice_penalty = (
        (segments - 1) / max(1, n - 1)
        if segments > 1
        else 0.0
    )

    # Coverage is mandatory. Long explanations dominate, bounded frequency
    # nudges ties, and each additional splice discounts the final evidence.
    shape = 0.52 + 0.30 * longest_fraction + 0.18 * frequency
    score = coverage * shape * (1.0 - 0.22 * splice_penalty)
    return CohesionResult(
        score=max(0.0, min(1.0, score)),
        coverage=coverage,
        longest_fraction=longest_fraction,
        segments=segments,
        splice_penalty=splice_penalty,
        frequency_strength=frequency,
        spans=best.spans,
    )


def blend_phrase_cohesion(base_score: float, cohesion: CohesionResult) -> float:
    """Add cohesion as positive-only evidence without weakening existing hits."""
    return min(1.0, max(base_score, 0.92 * cohesion.score))
