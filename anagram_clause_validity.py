"""Deterministic clause and noun-phrase validity checks for ranking.

The core grammar scorer is intentionally permissive around POS ambiguity. This
module adds a narrow validity layer for cases where WordNet's overlapping noun,
verb and function-word labels can otherwise make malformed clauses look fully
formed. The checks are symbolic and conservative: no corpus-negative evidence,
learned model or language model is involved.
"""

from __future__ import annotations

from collections.abc import Sequence

import anagram_rerank_core as core

_PRONOUN_CLASSES = frozenset({"PRON", "PRON_12", "PRON_PL", "PRON_SG3"})
_FINITE_AUX_CLASSES = frozenset(
    {"BE_AUX", "HAVE_AUX", "MODAL", "DONT", "DOESNT", "DO_AUX"}
)
_FINITE_BE = frozenset(
    {"am", "is", "are", "was", "were", "isnt", "arent", "wasnt", "werent"}
)
# A few function-word-like forms genuinely work as nominal heads. Keep this
# deliberately narrow so a/the/my cannot become subjects merely because WordNet
# also contains a noun sense for the surface token.
_NOMINAL_FUNCTION_HEADS = frozenset({"one", "more", "less", "most", "least"})

# Structure demotion constants are intentionally bounded rather than hard
# rejection. They encode the validity layer's contract: malformed analyses lose
# enough confidence to stop masquerading as complete clauses, while downstream
# lexical/phrase evidence can still distinguish imperfect but plausible text.
_HARD_NORM_CAP = 0.25
_HARD_COVERAGE_CAP = 0.25
_HARD_VALENCY_CAP = 0.50
_HARD_AGREEMENT_CAP = 0.50
_SOFT_VALENCY_CAP = 0.65
_SOFT_AGREEMENT_CAP = 0.65
_FRAGMENT_COVERAGE_THRESHOLD = 0.60
_SOFT_NORM_BASE_WEIGHT = 0.25
_SOFT_NORM_COVERAGE_WEIGHT = 0.75
_SOFT_NORM_COVERAGE_EXPONENT = 1.4

# Local/surface penalties are likewise deliberately small and high-confidence.
_DETERMINER_AUX_PAIR_PENALTY = 1.75
_ARTICLE_MISMATCH_PAIR_PENALTY = 1.60
_ARTICLE_MISMATCH_SURFACE_PENALTY = 0.12
_DETERMINER_AUX_SURFACE_PENALTY = 0.18

# Orthographic approximations only. They are used as bounded negative evidence,
# never as hard rejection, because English pronunciation delights in exceptions.
_VOWEL_SOUND_CONSONANT_PREFIXES = ("heir", "honest", "honor", "hour")
_CONSONANT_SOUND_VOWEL_PREFIXES = (
    "euro",
    "one",
    "once",
    "uni",
    "use",
    "user",
    "usual",
)


def valid_subject_head(word: str, lex: core.WordNetLexicon) -> bool:
    """Return whether ``word`` may head a subject NP in a finite clause."""
    cls = core.function_class(word)
    if cls in _PRONOUN_CLASSES:
        return True
    if word in _NOMINAL_FUNCTION_HEADS:
        return True
    if cls is not None:
        return False
    return lex.features(word).noun


def lexical_finite_form(word: str, lex: core.WordNetLexicon) -> bool:
    """Return whether a lexical token has an ordinary finite-verb analysis.

    Bare present participles are deliberately excluded. A token may still be
    ambiguous between a participle and a finite form; in that case the explicit
    finite morphology wins and the existing agreement scorer decides whether it
    fits the subject.
    """
    if core.function_class(word) is not None:
        return False
    features = lex.features(word)
    return features.verb_base or features.verb_3sg or features.verb_past


def indefinite_article_mismatch(article: str, following: str) -> bool:
    """Return a high-confidence orthographic a/an mismatch."""
    if article not in {"a", "an"}:
        return False
    word = core.norm_token(following)
    if not word:
        return False

    if word.startswith(_VOWEL_SOUND_CONSONANT_PREFIXES):
        expected = "an"
    elif word.startswith(_CONSONANT_SOUND_VOWEL_PREFIXES):
        expected = "a"
    else:
        expected = "an" if word[0] in "aeiou" else "a"
    return article != expected


def _subject_span_before(
    words: Sequence[str],
    verb_idx: int,
    lex: core.WordNetLexicon,
) -> tuple[int, int, float] | None:
    """Locate a compact subject NP before a lexical finite verb."""
    subj_head_idx = verb_idx - 1
    while subj_head_idx >= 0:
        features = lex.features(words[subj_head_idx])
        cls = core.function_class(words[subj_head_idx])
        if cls == "NEG" or (features.adv and cls not in {"PREP", "CONJ"}):
            subj_head_idx -= 1
            continue
        break

    if subj_head_idx < 0 or not valid_subject_head(words[subj_head_idx], lex):
        return None

    subject_span = core._np_span_ending_at(words, subj_head_idx, lex)
    if subject_span is None:
        return None
    subj_start, subj_coh = subject_span
    return subj_start, subj_head_idx, subj_coh


def _explicit_aux_subject_state(
    words: Sequence[str],
    lex: core.WordNetLexicon,
) -> tuple[bool, bool]:
    """Return (has finite explicit auxiliary, has one with a valid subject)."""
    found = False
    for aux_idx, token in enumerate(words):
        cls = core.function_class(token)
        if aux_idx == 0 or cls not in _FINITE_AUX_CLASSES:
            continue
        if cls == "BE_AUX" and token not in _FINITE_BE:
            continue
        found = True
        subj_head_idx = aux_idx - 1
        if not valid_subject_head(words[subj_head_idx], lex):
            continue
        if core._np_span_ending_at(words, subj_head_idx, lex) is not None:
            return True, True
    return found, False


def _is_subjectless_do_imperative(
    words: Sequence[str],
    lex: core.WordNetLexicon,
) -> bool:
    """Preserve the core's valid leading ``do``/``don't`` imperative analysis."""
    if len(words) < 2 or words[0] not in {"do", "dont"}:
        return False
    return lex.features(words[1]).verb_base


def best_valid_lexical_clause_coverage(
    words: Sequence[str],
    lex: core.WordNetLexicon,
) -> float:
    """Return the best coverage achievable by a genuinely finite lexical clause."""
    words = tuple(words)
    n = len(words)
    if n < 2:
        return 0.0

    best = 0.0
    for verb_idx, token in enumerate(words):
        if not lexical_finite_form(token, lex):
            continue

        subject = _subject_span_before(words, verb_idx, lex)
        if subject is None:
            continue
        subj_start, subj_head_idx, _subj_coh = subject

        _valency, tail_consumed = core._valency_for_tail(
            token, words[verb_idx + 1 :], lex
        )

        subject_tokens = subj_head_idx - subj_start + 1
        intervening = verb_idx - subj_head_idx - 1
        consumed = subject_tokens + intervening + 1 + tail_consumed
        coverage = min(1.0, consumed / n)
        if subj_start > 0:
            coverage *= 0.90
        best = max(best, coverage)

    return best


def _demote_structure(
    result: core.StructureResult,
    *,
    coverage: float,
    hard: bool,
) -> core.StructureResult:
    """Return a bounded demotion while preserving the scorer's result schema."""
    coverage = max(0.0, min(result.coverage, coverage))
    if hard or coverage <= 0.0:
        norm = min(result.norm, _HARD_NORM_CAP)
        coverage = min(result.coverage, _HARD_COVERAGE_CAP)
        valency = min(result.valency, _HARD_VALENCY_CAP)
        agreement = min(result.agreement, _HARD_AGREEMENT_CAP)
        kind = "fragment"
    else:
        ratio = max(0.0, min(1.0, coverage / result.coverage))
        scale = _SOFT_NORM_BASE_WEIGHT + _SOFT_NORM_COVERAGE_WEIGHT * (
            ratio**_SOFT_NORM_COVERAGE_EXPONENT
        )
        norm = min(result.norm, result.norm * scale)
        valency = min(result.valency, _SOFT_VALENCY_CAP)
        agreement = min(result.agreement, _SOFT_AGREEMENT_CAP)
        kind = (
            "fragment"
            if coverage < _FRAGMENT_COVERAGE_THRESHOLD
            else result.kind
        )

    norm = max(0.0, norm)
    return core.StructureResult(norm, valency, coverage, agreement, kind, 4.0 * norm)


def adjust_base_clause_structure(
    words: Sequence[str],
    lex: core.WordNetLexicon,
    result: core.StructureResult,
) -> core.StructureResult:
    """Demote finite-clause scores that depend on invalid subject/morphology.

    Two concrete WordNet ambiguity failures motivate this layer:

    * ``a am sitting managers`` can treat the article ``a`` as a noun subject.
    * ``an game starting aims`` can treat bare ``starting`` as a finite verb.

    Valid explicit auxiliaries are left to the existing auxiliary/copular logic.
    The core's leading ``do``/``don't`` imperative form is also preserved: it is
    intentionally subjectless, so requiring an explicit NP there would create a
    false negative. For lexical clauses, reported coverage is capped to what a
    genuinely finite lexical predicate can cover.
    """
    if result.kind not in {"clause", "copula"} or result.coverage <= 0.0:
        return result
    if result.kind == "clause" and _is_subjectless_do_imperative(words, lex):
        return result

    has_aux, valid_aux_subject = _explicit_aux_subject_state(words, lex)
    if has_aux:
        if valid_aux_subject:
            return result
        return _demote_structure(result, coverage=_HARD_COVERAGE_CAP, hard=True)

    if result.kind == "copula":
        return result

    valid_coverage = best_valid_lexical_clause_coverage(words, lex)
    if valid_coverage + 0.05 >= result.coverage:
        return result
    return _demote_structure(
        result,
        coverage=valid_coverage,
        hard=valid_coverage <= 0.0,
    )


def pair_validity_adjustment(
    left: str,
    right: str,
    lex: core.WordNetLexicon,
) -> float:
    """Return bounded local evidence against clearly malformed transitions."""
    del lex
    right_cls = core.function_class(right)
    score = 0.0

    # A determiner cannot itself be the subject head immediately before a
    # finite auxiliary. Core pair_grammar may otherwise recover some positive
    # noun evidence from WordNet's alternate sense of tokens such as ``a``.
    # Explicit nominal function heads such as ``one`` are the narrow exception.
    if (
        left not in _NOMINAL_FUNCTION_HEADS
        and core._det_class(left) is not None
        and right_cls in _FINITE_AUX_CLASSES
    ):
        score -= _DETERMINER_AUX_PAIR_PENALTY

    # Do not penalize generic noun -> V-ing pairs locally: English noun phrases
    # such as "silver lining" make that adjacency legitimately common. Bare
    # V-ing pretending to be a finite predicate is handled at whole-clause level.
    if indefinite_article_mismatch(left, right):
        score -= _ARTICLE_MISMATCH_PAIR_PENALTY

    return score


def apply_surface_structure_penalties(
    words: Sequence[str],
    lex: core.WordNetLexicon,
    result: core.StructureResult,
) -> core.StructureResult:
    """Apply small whole-phrase penalties for high-confidence surface errors."""
    words = tuple(words)
    article_mismatches = sum(
        1
        for left, right in zip(words, words[1:])
        if indefinite_article_mismatch(left, right)
    )
    determiner_aux = sum(
        1
        for left, right in zip(words, words[1:])
        if left not in _NOMINAL_FUNCTION_HEADS
        and core._det_class(left) is not None
        and core.function_class(right) in _FINITE_AUX_CLASSES
    )
    penalty = (
        _ARTICLE_MISMATCH_SURFACE_PENALTY * article_mismatches
        + _DETERMINER_AUX_SURFACE_PENALTY * determiner_aux
    )
    if penalty <= 0.0:
        return result

    norm = max(0.0, result.norm - penalty)
    return core.StructureResult(
        norm,
        result.valency,
        result.coverage,
        result.agreement,
        result.kind,
        4.0 * norm,
    )
