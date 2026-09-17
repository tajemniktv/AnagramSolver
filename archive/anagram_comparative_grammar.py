"""Deterministic comparative morphology and compact comparative-span parsing.

A spelling can support a comparative reading without that reading being the
only, or even the dominant, lexical interpretation.  For example ``closer`` is
also a noun, while ``number`` can be both the ordinary noun and the regular
comparative of adjective ``numb``.  Treating that ambiguity as a boolean POS
veto either loses real comparatives or rewards homographs too strongly.

This module therefore exposes graded comparative evidence.  Local ordering can
use the confidence conservatively, while a complete ``... comparative than X``
construction can use the surrounding syntax to disambiguate the same form.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import anagram_rerank_core as core


class LexiconLike(Protocol):
    """Minimal lexical contract needed by comparative morphology."""

    def features(self, raw_word: str) -> core.Features: ...


ComparativeSource = Literal[
    "none",
    "lexical",
    "irregular",
    "regular-surface",
    "regular-recovered",
    "regular-ambiguous",
]

_IRREGULAR_COMPARATIVE_WORDS = frozenset({"elder", "farther", "further"})
_PRONOUN_CLASSES = frozenset({"PRON", "PRON_12", "PRON_PL", "PRON_SG3"})


@dataclass(slots=True, frozen=True)
class ComparativeEvidence:
    """Evidence that a token can carry a comparative reading.

    ``confidence`` is intentionally not a probability.  It is a bounded
    deterministic weight describing how directly the available lexical data
    supports the comparative interpretation.  A homonymous noun/verb surface
    remains possible but receives much less local ordering weight.
    """

    confidence: float
    base: str | None
    source: ComparativeSource

    @property
    def present(self) -> bool:
        return self.confidence > 0.0


NO_COMPARATIVE_EVIDENCE = ComparativeEvidence(0.0, None, "none")


def comparative_base_candidates(word: str) -> tuple[str, ...]:
    """Return conservative regular bases for a surface ``-er`` spelling."""
    word = core.norm_token(word)
    if len(word) <= 3 or not word.endswith("er"):
        return ()

    stem = word[:-2]
    candidates = {stem, word[:-1]}
    if stem.endswith("i") and len(stem) > 1:
        candidates.add(stem[:-1] + "y")
    if len(stem) >= 2 and stem[-1] == stem[-2]:
        candidates.add(stem[:-1])
    return tuple(sorted(candidate for candidate in candidates if candidate))


def comparative_evidence(word: str, lex: LexiconLike) -> ComparativeEvidence:
    """Return graded evidence for a comparative reading of ``word``.

    Lexical/irregular comparatives are strongest.  Regular ``-er`` forms must
    recover an adjective/adverb base.  When the surface itself is known only as
    another content POS, the comparative reading is retained but marked
    ambiguous instead of being either discarded or treated as fully certain.
    """
    word = core.norm_token(word)
    if word in core.COMPARATIVE_WORDS:
        return ComparativeEvidence(1.0, None, "lexical")
    if word in _IRREGULAR_COMPARATIVE_WORDS:
        return ComparativeEvidence(0.98, None, "irregular")
    if not word.endswith("er"):
        return NO_COMPARATIVE_EVIDENCE

    eligible: list[tuple[str, core.Features]] = []
    for base in comparative_base_candidates(word):
        features = lex.features(base)
        if features.adj or features.adv:
            eligible.append((base, features))
    if not eligible:
        return NO_COMPARATIVE_EVIDENCE

    # Prefer a base supported as both adjective and adverb, then a stable
    # lexical order.  The selected base is diagnostic; confidence is determined
    # from the surface ambiguity rather than pretending WordNet has sense
    # frequencies for these inflected forms.
    base, _base_features = max(
        eligible,
        key=lambda item: (int(item[1].adj) + int(item[1].adv), item[0]),
    )
    surface = lex.features(word)
    if surface.adj or surface.adv:
        return ComparativeEvidence(0.96, base, "regular-surface")
    if surface.recognized:
        return ComparativeEvidence(0.38, base, "regular-ambiguous")
    return ComparativeEvidence(0.90, base, "regular-recovered")


def comparative_span_starting_at(
    words: Sequence[str],
    start: int,
    lex: core.WordNetLexicon,
) -> tuple[int, float] | None:
    """Parse a compact comparative span using graded morphology evidence.

    The complete ``... than ...`` topology supplies contextual evidence, so an
    ambiguous homograph is allowed here even though it receives only a weak
    standalone adjacency bonus elsewhere.
    """
    if start >= len(words):
        return None

    max_end = min(len(words), start + 5)
    for than_idx in range(start + 1, max_end):
        if words[than_idx] != "than":
            continue

        left = words[start:than_idx]
        right = words[than_idx + 1 :]
        if not left or not right:
            continue

        evidences = tuple(comparative_evidence(word, lex) for word in left)
        best = max(evidences, key=lambda item: item.confidence)
        if not best.present:
            continue

        right_consumed = 0
        np = core._np_span_starting_at(right, 0, lex)
        if np is not None:
            right_consumed = np[0] + 1
        else:
            features = lex.features(right[0])
            right_class = core.function_class(right[0])
            if (
                features.noun
                or features.adj
                or features.adv
                or right_class in _PRONOUN_CLASSES
                or right_class in {"NEG", "NUM_DET"}
            ):
                right_consumed = 1

        if right_consumed <= 0:
            continue

        end = than_idx + right_consumed
        first = evidences[0]
        quality = 0.82 + 0.16 * best.confidence
        if first.present:
            quality += 0.02
        return end, min(0.99, quality)

    return None
