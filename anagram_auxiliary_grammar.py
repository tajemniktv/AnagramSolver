"""Auxiliary-chain grammar support layered over the core structural scorer.

This module recognizes finite auxiliary chains that the original hand-written
structure parser deliberately handled only in a few special cases. It stays
symbolic and deterministic: morphology comes from WordNet, valency comes from
the existing frame parser, and no language model or learned score is involved.
It also wires the narrow clause-validity layer used to reject obvious POS-
ambiguity failures before they can dominate final ranking.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import anagram_clause_validity as validity
import anagram_rerank_core as core


class LexiconLike(Protocol):
    """Minimal lexicon contract needed by morphology-only helpers."""

    def features(self, raw_word: str) -> core.Features: ...


BaseStructureFn = Callable[[Sequence[str], core.WordNetLexicon], core.StructureResult]
BaseTablesFn = Callable[
    [tuple[str, ...], core.WordNetLexicon],
    tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]],
]

_FINITE_BE = frozenset(
    {"am", "is", "are", "was", "were", "isnt", "arent", "wasnt", "werent"}
)


@dataclass(slots=True, frozen=True)
class AuxiliaryChain:
    main_idx: int
    kind: str
    quality: float
    passive: bool = False


def _skip_negation(words: Sequence[str], idx: int) -> tuple[int, float]:
    quality = 1.0
    if idx < len(words) and core.function_class(words[idx]) == "NEG":
        return idx + 1, 0.97
    return idx, quality


def _lexical_verb(
    words: Sequence[str],
    idx: int,
    lex: LexiconLike,
    *,
    form: str,
) -> bool:
    if idx >= len(words) or core.function_class(words[idx]) is not None:
        return False
    features = lex.features(words[idx])
    if form == "base":
        return features.verb_base
    if form == "past":
        return features.verb_past
    if form == "ing":
        return features.verb_ing
    raise ValueError(f"unsupported verb form: {form}")


def _parse_be_chain(
    words: Sequence[str],
    be_idx: int,
    lex: LexiconLike,
) -> AuxiliaryChain | None:
    idx, neg_quality = _skip_negation(words, be_idx + 1)
    if idx >= len(words):
        return None

    # BE + being + past participle: "are being tested".
    if words[idx] == "being" and core.function_class(words[idx]) == "BE_AUX":
        main_idx, inner_quality = _skip_negation(words, idx + 1)
        if _lexical_verb(words, main_idx, lex, form="past"):
            return AuxiliaryChain(
                main_idx,
                "progressive-passive",
                0.97 * neg_quality * inner_quality,
                passive=True,
            )
        return None

    if _lexical_verb(words, idx, lex, form="ing"):
        return AuxiliaryChain(idx, "progressive", 0.99 * neg_quality)

    if _lexical_verb(words, idx, lex, form="past"):
        return AuxiliaryChain(idx, "passive", 0.96 * neg_quality, passive=True)

    return None


def _parse_have_chain(
    words: Sequence[str],
    have_idx: int,
    lex: LexiconLike,
) -> AuxiliaryChain | None:
    idx, neg_quality = _skip_negation(words, have_idx + 1)
    if idx >= len(words):
        return None

    # HAVE + been + (V-ing | V-en): perfect progressive/passive.
    if words[idx] == "been" and core.function_class(words[idx]) == "BE_AUX":
        nested = _parse_be_chain(words, idx, lex)
        if nested is None:
            return None
        return AuxiliaryChain(
            nested.main_idx,
            f"perfect-{nested.kind}",
            0.98 * neg_quality * nested.quality,
            passive=nested.passive,
        )

    if _lexical_verb(words, idx, lex, form="past"):
        return AuxiliaryChain(idx, "perfect", 0.98 * neg_quality)

    return None


def parse_auxiliary_chain(
    words: Sequence[str],
    aux_idx: int,
    lex: LexiconLike,
) -> AuxiliaryChain | None:
    """Parse a finite auxiliary chain beginning at ``aux_idx``.

    Supported families include progressive, passive, perfect, modal, do-support,
    and the common nested combinations built from modal + have/be and
    have + been + progressive/passive. Non-finite BE forms are accepted only
    when reached inside a finite modal/perfect chain, never as root auxiliaries.
    """
    if aux_idx >= len(words):
        return None

    word = words[aux_idx]
    cls = core.function_class(word)

    if cls == "BE_AUX":
        if word not in _FINITE_BE:
            return None
        return _parse_be_chain(words, aux_idx, lex)

    if cls == "HAVE_AUX":
        return _parse_have_chain(words, aux_idx, lex)

    if cls == "MODAL":
        idx, neg_quality = _skip_negation(words, aux_idx + 1)
        if idx >= len(words):
            return None
        if words[idx] == "be" and core.function_class(words[idx]) == "BE_AUX":
            nested = _parse_be_chain(words, idx, lex)
            if nested is not None:
                return AuxiliaryChain(
                    nested.main_idx,
                    f"modal-{nested.kind}",
                    0.98 * neg_quality * nested.quality,
                    passive=nested.passive,
                )
        if words[idx] == "have" and core.function_class(words[idx]) == "HAVE_AUX":
            nested = _parse_have_chain(words, idx, lex)
            if nested is not None:
                return AuxiliaryChain(
                    nested.main_idx,
                    f"modal-{nested.kind}",
                    0.98 * neg_quality * nested.quality,
                    passive=nested.passive,
                )
        if _lexical_verb(words, idx, lex, form="base"):
            return AuxiliaryChain(idx, "modal", 0.97 * neg_quality)
        return None

    if cls in {"DONT", "DOESNT", "DO_AUX"}:
        idx, neg_quality = _skip_negation(words, aux_idx + 1)
        if _lexical_verb(words, idx, lex, form="base"):
            return AuxiliaryChain(idx, "do-support", 0.96 * neg_quality)

    return None


def _be_agreement(subject: str, number: str, auxiliary: str) -> float:
    form = {
        "isnt": "is",
        "arent": "are",
        "wasnt": "was",
        "werent": "were",
    }.get(auxiliary, auxiliary)

    if form == "am":
        return 1.0 if subject == "i" else 0.05
    if form == "is":
        return 1.0 if number == "3sg" else (0.10 if number == "non3sg" else 0.55)
    if form == "are":
        if subject == "i":
            return 0.05
        return 1.0 if number == "non3sg" else (0.15 if number == "3sg" else 0.55)
    if form == "was":
        if subject == "i" or number == "3sg":
            return 1.0
        return 0.15 if number == "non3sg" else 0.55
    if form == "were":
        if subject == "i" or number == "3sg":
            return 0.15
        return 1.0 if number == "non3sg" else 0.55
    return 0.55


def _have_agreement(subject: str, number: str, auxiliary: str) -> float:
    del subject
    form = {"hasnt": "has", "havent": "have", "hadnt": "had"}.get(
        auxiliary, auxiliary
    )
    if form == "had":
        return 0.96
    if form == "has":
        return 1.0 if number == "3sg" else (0.10 if number == "non3sg" else 0.55)
    if form == "have":
        return 1.0 if number == "non3sg" else (0.12 if number == "3sg" else 0.55)
    return 0.55


def auxiliary_agreement(
    subject: str,
    number: str,
    auxiliary: str,
    lex: core.WordNetLexicon,
) -> float:
    """Score agreement for a finite auxiliary against its subject."""
    cls = core.function_class(auxiliary)
    if cls == "MODAL":
        return 0.98
    if cls == "BE_AUX":
        return _be_agreement(subject, number, auxiliary)
    if cls == "HAVE_AUX":
        return _have_agreement(subject, number, auxiliary)
    return core._subject_agreement(
        subject,
        auxiliary,
        lex,
        auxiliary=True,
        number_override=number,
    )


def _consume_passive_adverbs(
    tail: Sequence[str],
    start: int,
    lex: core.WordNetLexicon,
) -> int:
    """Consume consecutive adverb/negative adjuncts beginning at ``start``."""
    consumed = start
    while consumed < len(tail):
        feature = lex.features(tail[consumed])
        cls = core.function_class(tail[consumed])
        if cls == "NEG" or (feature.adv and cls not in {"PREP", "CONJ"}):
            consumed += 1
            continue
        break
    return consumed


def _passive_tail(
    tail: Sequence[str],
    lex: core.WordNetLexicon,
) -> tuple[float, int]:
    if not tail:
        return 0.96, 0

    if tail[0] == "by" and len(tail) > 1:
        np = core._np_span_starting_at(tail, 1, lex)
        if np is not None:
            # _np_span_starting_at returns an inclusive end index. Therefore
            # ``end + 1`` is the exact token count for ``by <NP>``. Continue
            # through genuine adverbial adjuncts, but do not hide arbitrary
            # trailing tokens under the coverage cap.
            consumed = _consume_passive_adverbs(tail, np[0] + 1, lex)
            return 0.99, consumed

    # A passive clause is already valency-complete. Consume ordinary trailing
    # adverbs as adjuncts without pretending that WordNet requires them.
    consumed = _consume_passive_adverbs(tail, 0, lex)
    if consumed:
        return 0.92, consumed

    return 0.90, 0


def auxiliary_structure(
    words: Sequence[str],
    lex: core.WordNetLexicon,
) -> core.StructureResult | None:
    """Return the best full-clause auxiliary-chain interpretation, if any."""
    words = tuple(words)
    n = len(words)
    if n < 3:
        return None

    candidates: list[core.StructureResult] = []

    det_collisions = sum(
        1
        for left, right in itertools.pairwise(words)
        if core._det_class(left) is not None and core._det_class(right) is not None
    )

    for aux_idx, auxiliary in enumerate(words):
        if aux_idx == 0 or core.function_class(auxiliary) not in {
            "BE_AUX",
            "HAVE_AUX",
            "MODAL",
            "DONT",
            "DOESNT",
            "DO_AUX",
        }:
            continue

        chain = parse_auxiliary_chain(words, aux_idx, lex)
        if chain is None:
            continue

        subj_head_idx = aux_idx - 1
        if not validity.valid_subject_head(words[subj_head_idx], lex):
            continue
        subject_span = core._np_span_ending_at(words, subj_head_idx, lex)
        if subject_span is None:
            continue
        subj_start, subj_coh = subject_span
        subject = words[subj_head_idx]
        number = core._subject_number_from_span(words, subj_start, subj_head_idx, lex)
        agreement = auxiliary_agreement(subject, number, auxiliary, lex)

        tail = words[chain.main_idx + 1 :]
        if chain.passive:
            valency, tail_consumed = _passive_tail(tail, lex)
        else:
            valency, tail_consumed = core._valency_for_tail(
                words[chain.main_idx], tail, lex
            )

        subject_tokens = aux_idx - subj_start
        chain_tokens = chain.main_idx - aux_idx + 1
        consumed = subject_tokens + chain_tokens + tail_consumed
        coverage = min(1.0, consumed / n)
        if subj_start > 0:
            coverage *= 0.90

        norm = (
            0.18 * subj_coh
            + 0.22 * agreement
            + 0.20 * chain.quality
            + 0.18 * valency
            + 0.22 * coverage
        )
        norm *= 0.42 + 0.58 * (coverage**1.3)
        if agreement <= 0.20:
            norm = min(norm, 0.42)
        norm -= 0.22 * det_collisions
        norm = max(0.0, min(0.98, norm))

        candidates.append(
            core.StructureResult(
                norm,
                valency,
                coverage,
                agreement,
                f"aux-{chain.kind}",
                4.0 * norm,
            )
        )

    if not candidates:
        return None
    return max(candidates, key=lambda result: (result.norm, result.coverage, result.valency))


def phrase_structure_with_auxiliaries(
    words: Sequence[str],
    lex: core.WordNetLexicon,
    base_structure: BaseStructureFn,
) -> core.StructureResult:
    """Combine legacy, auxiliary and narrow clause-validity structure evidence."""
    words = tuple(words)
    base = validity.adjust_base_clause_structure(words, lex, base_structure(words, lex))
    auxiliary = auxiliary_structure(words, lex)
    winner = (
        base
        if auxiliary is None
        else max(
            (base, auxiliary),
            key=lambda result: (result.norm, result.coverage, result.valency),
        )
    )
    return validity.apply_surface_structure_penalties(words, lex, winner)


def _auxiliary_pair_bonus_for_class(
    left_class: str | None,
    right: str,
    lex: LexiconLike,
) -> float:
    """Return an auxiliary-transition bonus with the left class precomputed."""
    right_cls = core.function_class(right)
    features = lex.features(right)

    if left_class == "BE_AUX":
        if right == "being" and right_cls == "BE_AUX":
            return 1.10
        if right_cls is None and features.verb_ing:
            return 1.20
        if right_cls is None and features.verb_past:
            return 1.00

    if left_class == "HAVE_AUX":
        if right == "been" and right_cls == "BE_AUX":
            return 1.05
        if right_cls is None and features.verb_past:
            return 1.10

    return 0.0


def auxiliary_pair_bonus(left: str, right: str, lex: LexiconLike) -> float:
    """Extra local evidence for morphologically valid auxiliary transitions."""
    return _auxiliary_pair_bonus_for_class(core.function_class(left), right, lex)


def order_local_tables_with_auxiliaries(
    words: tuple[str, ...],
    lex: core.WordNetLexicon,
    base_tables: BaseTablesFn,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    pair, starts, ends = base_tables(words, lex)
    rows = [list(row) for row in pair]
    for i, left in enumerate(words):
        left_class = core.function_class(left)
        for j, right in enumerate(words):
            if i == j:
                continue
            rows[i][j] += validity.pair_validity_adjustment(left, right, lex)
            if left_class in {"BE_AUX", "HAVE_AUX"}:
                rows[i][j] += _auxiliary_pair_bonus_for_class(left_class, right, lex)
    return tuple(tuple(row) for row in rows), starts, ends


def local_grammar_raw_with_auxiliaries(
    words: Sequence[str],
    lex: core.WordNetLexicon,
    base_tables: BaseTablesFn,
) -> float:
    """Compute the local raw score while touching only realized adjacencies."""
    words = tuple(words)
    if not words:
        return 0.0
    pair, starts, ends = base_tables(words, lex)
    if len(words) == 1:
        return starts[0] + ends[0]
    total = starts[0] + ends[-1]
    total += sum(
        pair[i][i + 1]
        + auxiliary_pair_bonus(words[i], words[i + 1], lex)
        + validity.pair_validity_adjustment(words[i], words[i + 1], lex)
        for i in range(len(words) - 1)
    )
    return total / (len(words) - 1)
