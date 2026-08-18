"""Corpus-derived directional argument-fit evidence around candidate verbs.

This deliberately contains no verb-specific semantic table.  It asks the phrase
corpus which noun/verb and verb/noun spans are actually attested in the proposed
order, giving extra weight to spans centered on a WordNet-recognized verb.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

CountsLookup = Callable[[Sequence[str]], Mapping[str, int]]
TokenPredicate = Callable[[str], bool]


@dataclass(slots=True, frozen=True)
class ThematicSpan:
    side: str
    text: str
    count: int
    strength: float


@dataclass(slots=True, frozen=True)
class ThematicFitResult:
    score: float
    subject_strength: float
    object_strength: float
    verb_coverage: float
    verbs_with_evidence: int
    spans: tuple[ThematicSpan, ...]


def _strength(count: int) -> float:
    if count <= 0:
        return 0.0
    return min(1.0, math.log10(count + 1.0) / 5.0)


def score_thematic_fit(
    words: Sequence[str],
    *,
    counts: CountsLookup,
    is_verb: TokenPredicate,
    is_nominal: TokenPredicate,
    max_span: int = 4,
) -> ThematicFitResult:
    """Score positive directional corpus evidence around recognized verbs.

    For each candidate verb, query short contiguous spans ending at the verb and
    starting at the verb.  A span is considered role-like only when its non-verb
    side contains at least one nominal token. This emphasizes evidence such as
    ``dog chased`` / ``chased the ball`` without encoding that dogs, balls,
    phones, charges, or any particular verb belong to hand-maintained classes.
    """
    ordered = tuple(word for word in words if word)
    n = len(ordered)
    if n < 2 or max_span < 2:
        return ThematicFitResult(0.0, 0.0, 0.0, 0.0, 0, ())

    verb_positions = [index for index, word in enumerate(ordered) if is_verb(word)]
    if not verb_positions:
        return ThematicFitResult(0.0, 0.0, 0.0, 0.0, 0, ())

    queries: list[str] = []
    metadata: dict[str, tuple[str, int]] = {}
    upper = min(max_span, n)
    for verb_index in verb_positions:
        for width in range(2, upper + 1):
            start = verb_index - width + 1
            if start >= 0 and any(is_nominal(word) for word in ordered[start:verb_index]):
                text = " ".join(ordered[start : verb_index + 1])
                queries.append(text)
                metadata[text] = ("subject", verb_index)

            end = verb_index + width
            if end <= n and any(is_nominal(word) for word in ordered[verb_index + 1 : end]):
                text = " ".join(ordered[verb_index:end])
                queries.append(text)
                metadata[text] = ("object", verb_index)

    if not queries:
        return ThematicFitResult(0.0, 0.0, 0.0, 0.0, 0, ())

    hit_counts = counts(tuple(dict.fromkeys(queries)))
    best_subject: dict[int, ThematicSpan] = {}
    best_object: dict[int, ThematicSpan] = {}
    for text, (side, verb_index) in metadata.items():
        count = int(hit_counts.get(text, 0))
        if count <= 0:
            continue
        span = ThematicSpan(side, text, count, _strength(count))
        target = best_subject if side == "subject" else best_object
        previous = target.get(verb_index)
        if previous is None or (span.strength, len(span.text), span.text) > (
            previous.strength,
            len(previous.text),
            previous.text,
        ):
            target[verb_index] = span

    evidence_verbs = sorted(set(best_subject) | set(best_object))
    if not evidence_verbs:
        return ThematicFitResult(0.0, 0.0, 0.0, 0.0, 0, ())

    subject_values = [best_subject[index].strength for index in evidence_verbs if index in best_subject]
    object_values = [best_object[index].strength for index in evidence_verbs if index in best_object]
    subject_strength = sum(subject_values) / len(subject_values) if subject_values else 0.0
    object_strength = sum(object_values) / len(object_values) if object_values else 0.0

    per_verb: list[float] = []
    chosen: list[ThematicSpan] = []
    for index in evidence_verbs:
        left = best_subject.get(index)
        right = best_object.get(index)
        if left is not None:
            chosen.append(left)
        if right is not None:
            chosen.append(right)
        left_score = left.strength if left else 0.0
        right_score = right.strength if right else 0.0
        if left_score > 0.0 and right_score > 0.0:
            balanced = math.sqrt(left_score * right_score)
            per_verb.append(0.35 * max(left_score, right_score) + 0.65 * balanced)
        else:
            per_verb.append(0.55 * max(left_score, right_score))

    verb_coverage = len(evidence_verbs) / len(verb_positions)
    score = (sum(per_verb) / len(per_verb)) * (0.72 + 0.28 * verb_coverage)
    return ThematicFitResult(
        score=max(0.0, min(1.0, score)),
        subject_strength=subject_strength,
        object_strength=object_strength,
        verb_coverage=verb_coverage,
        verbs_with_evidence=len(evidence_verbs),
        spans=tuple(chosen),
    )
