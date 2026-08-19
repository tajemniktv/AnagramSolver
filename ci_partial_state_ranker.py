#!/usr/bin/env python3
"""Leakage-safe research A/B for a learned partial-bag beam scorer.

The complete-order ranker has useful evidence that does not exist yet while a
word bag is still being assembled. This experiment deliberately stays earlier in
the pipeline: it learns a tiny linear pairwise scorer from cheap lexical,
collocation, progress, and candidate-suffix features only.

Training/evaluation is grouped by whole benchmark puzzle. No partial state from a
held-out answer may appear in that fold's training data.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anagram_generate as generator
import anagram_user_lexicon as lexicon
import anagram_user_search as search

BENCHMARK_PATH = Path("anagram_benchmarks.json")
FOLDS = 5
DATA_BEAM_WIDTH = 128
EVAL_BEAM_WIDTH = 192
NEGATIVES_PER_DEPTH = 48
EPOCHS = 35
LEARNING_RATE = 0.06
L2 = 0.004

FEATURE_NAMES = (
    "lexical",
    "optimistic_lexical",
    "avg_zipf",
    "min_zipf",
    "last_zipf",
    "pair_coverage",
    "pair_strength",
    "duplicate_fraction",
    "chosen_fraction",
    "remaining_letter_fraction",
    "next_start_fraction",
    "remaining_avg_length",
)
FeatureVector = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreparedCase:
    key: str
    answer: str
    target: tuple[int, ...]
    target_indices: tuple[int, ...]
    candidates: list[generator.Candidate]
    sparse_signatures: list[tuple[tuple[int, int], ...]]
    by_signature: dict[tuple[int, ...], list[int]]
    min_candidate_len: int
    max_candidate_len: int


@dataclass(frozen=True, slots=True)
class TrainingGroup:
    case_key: str
    depth: int
    positive: FeatureVector
    negatives: tuple[FeatureVector, ...]


@dataclass(frozen=True, slots=True)
class LinearModel:
    weights: FeatureVector

    def score(self, features: FeatureVector) -> float:
        return sum(
            weight * value
            for weight, value in zip(self.weights, features, strict=True)
        )


@dataclass(frozen=True, slots=True)
class TraceResult:
    survived: bool
    drop_depth: int | None
    expansions: int
    seconds: float


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _fold(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _features(
    expansion: search.BeamExpansion,
    *,
    word_count: int,
    words_left_after: int,
    total_letters: int,
    candidates: list[generator.Candidate],
    max_candidate_len: int,
    pair_priority: tuple[float, float, float],
) -> FeatureVector:
    chosen = expansion.chosen
    zipfs = [candidates[index].zipf for index in chosen]
    lexical = search._lexical_score(chosen, candidates)
    if words_left_after:
        future_zipf = candidates[expansion.next_start].zipf
        optimistic = search._optimistic_score(
            chosen,
            words_left_after,
            future_zipf,
            candidates,
        )
    else:
        optimistic = lexical

    average_zipf = sum(zipfs) / len(zipfs)
    minimum_zipf = min(zipfs)
    last_zipf = candidates[chosen[-1]].zipf
    duplicate_fraction = (len(chosen) - len(set(chosen))) / len(chosen)
    remaining_word_count = max(words_left_after, 1)
    remaining_avg_length = expansion.remaining_len / remaining_word_count
    pair_coverage, pair_raw, _pair_lexical = pair_priority

    values = (
        _clip(lexical / 8.0),
        _clip(optimistic / 8.0),
        _clip(average_zipf / 8.0),
        _clip(minimum_zipf / 8.0),
        _clip(last_zipf / 8.0),
        _clip(pair_coverage),
        math.tanh(pair_raw / 8.0),
        _clip(duplicate_fraction),
        _clip(len(chosen) / max(word_count, 1)),
        _clip(expansion.remaining_len / max(total_letters, 1)),
        _clip(expansion.next_start / max(len(candidates), 1)),
        _clip(remaining_avg_length / max(max_candidate_len, 1)),
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise RuntimeError("partial-state feature schema produced invalid values")
    return values


def _sigmoid_negative_margin(margin: float) -> float:
    if margin >= 0.0:
        exp_neg = math.exp(-min(margin, 700.0))
        return exp_neg / (1.0 + exp_neg)
    exp_pos = math.exp(max(margin, -700.0))
    return 1.0 / (1.0 + exp_pos)


def _train(groups: list[TrainingGroup]) -> LinearModel:
    weights = [0.0] * len(FEATURE_NAMES)
    ordered = sorted(groups, key=lambda group: (group.case_key, group.depth))
    step = 0
    for _ in range(EPOCHS):
        for group in ordered:
            for negative in group.negatives:
                diff = [
                    positive - negative_value
                    for positive, negative_value in zip(
                        group.positive,
                        negative,
                        strict=True,
                    )
                ]
                margin = sum(
                    weight * delta
                    for weight, delta in zip(weights, diff, strict=True)
                )
                probability = _sigmoid_negative_margin(margin)
                step += 1
                rate = LEARNING_RATE / math.sqrt(1.0 + 0.0005 * step)
                for index, delta in enumerate(diff):
                    gradient = probability * delta - L2 * weights[index]
                    weights[index] += rate * gradient
    return LinearModel(tuple(weights))


def _target_indices(
    answer_words: tuple[str, ...],
    candidates: list[generator.Candidate],
) -> tuple[int, ...] | None:
    by_word = {candidate.word: index for index, candidate in enumerate(candidates)}
    try:
        return tuple(sorted(by_word[word] for word in answer_words))
    except KeyError:
        return None


def _prepare_cases() -> tuple[list[PreparedCase], generator.BigramModel, list[str]]:
    payload = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise RuntimeError("benchmark file has no cases list")

    user_lexicon = lexicon.ensure_user_lexicon()
    ngram_dir = Path(generator.DEFAULT_NGRAM_DIR)
    one_path, two_path = generator.ensure_ngram_data(
        ngram_dir,
        refresh=False,
        need_bigrams=True,
    )
    if two_path is None:
        raise RuntimeError("bigram corpus unavailable for partial-state ranker")
    unigrams = generator.load_unigram_model(one_path)
    short_words = set(generator.DEFAULT_SHORT_WORDS)
    short_words.update(user_lexicon.extra_short_words)

    pending: list[
        tuple[
            str,
            str,
            tuple[int, ...],
            tuple[int, ...],
            list[generator.Candidate],
        ]
    ] = []
    skipped: list[str] = []
    vocabulary: set[str] = set()

    for raw in raw_cases:
        if not isinstance(raw, dict):
            continue
        key = raw.get("id")
        answer = raw.get("answer")
        if not isinstance(key, str) or not isinstance(answer, str):
            continue
        answer_words = tuple(generator.tokenize_words(answer))
        if not 2 <= len(answer_words) <= 6:
            skipped.append(f"{key}:word-count")
            continue
        target = generator.counts(answer)
        min_zipf_raw = raw.get("min_zipf", 2.7)
        min_zipf = float(min_zipf_raw) if isinstance(min_zipf_raw, (int, float)) else 2.7
        candidates = generator.load_words(
            user_lexicon.dictionary,
            target,
            min_len=2,
            max_len=sum(target),
            excluded_words=set(),
            exclude_regexes=[],
            forbid_chars=set(),
            min_zipf=min_zipf,
            short_policy="common",
            short_whitelist=short_words,
            forced_words=set(),
            unigrams=unigrams,
        )
        indices = _target_indices(answer_words, candidates)
        if indices is None:
            missing = sorted(set(answer_words) - {candidate.word for candidate in candidates})
            skipped.append(f"{key}:lexical:{','.join(missing)}")
            continue
        pending.append((key, answer, target, indices, candidates))
        vocabulary.update(candidate.word for candidate in candidates)

    bigrams = generator.load_bigram_model(two_path, unigrams, vocabulary)
    prepared: list[PreparedCase] = []
    for key, answer, target, indices, candidates in pending:
        sparse = [search._sparse_signature(candidate.sig) for candidate in candidates]
        by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, candidate in enumerate(candidates):
            by_signature[candidate.sig].append(index)
        prepared.append(
            PreparedCase(
                key=key,
                answer=answer,
                target=target,
                target_indices=indices,
                candidates=candidates,
                sparse_signatures=sparse,
                by_signature=dict(by_signature),
                min_candidate_len=min(candidate.length for candidate in candidates),
                max_candidate_len=max(candidate.length for candidate in candidates),
            )
        )
    return prepared, bigrams, skipped


def _expand_states(
    case: PreparedCase,
    states: list[search.BeamState],
    *,
    depth: int,
    bigrams: generator.BigramModel,
    width: int,
    model: LinearModel | None,
) -> tuple[list[search.BeamState], int]:
    word_count = len(case.target_indices)
    words_left_after = word_count - depth - 1
    total_letters = sum(case.target)
    branch_limit = search._branch_width(word_count)
    lexical_heap: list[search.ScoredItem] = []
    pair_heap: list[search.PairItem] = []
    anchor_heaps: dict[int, list[search.ScoredItem]] = defaultdict(list)
    serial = 0
    expansions = 0

    for rem, rem_len, start, chosen in states:
        for expansion in search._iter_state_expansions(
            rem,
            rem_len,
            start,
            chosen,
            candidates=case.candidates,
            sparse_signatures=case.sparse_signatures,
            by_signature=case.by_signature,
            min_candidate_len=case.min_candidate_len,
            max_candidate_len=case.max_candidate_len,
            words_left_after=words_left_after,
            allow_repeat=True,
            branch_limit=branch_limit,
        ):
            pair_priority = search._pair_priority(expansion.chosen, case.candidates, bigrams)
            if model is None:
                if words_left_after:
                    future_zipf = case.candidates[expansion.next_start].zipf
                    priority = search._optimistic_score(
                        expansion.chosen,
                        words_left_after,
                        future_zipf,
                        case.candidates,
                    )
                else:
                    priority = search._lexical_score(expansion.chosen, case.candidates)
            else:
                priority = model.score(
                    _features(
                        expansion,
                        word_count=word_count,
                        words_left_after=words_left_after,
                        total_letters=total_letters,
                        candidates=case.candidates,
                        max_candidate_len=case.max_candidate_len,
                        pair_priority=pair_priority,
                    )
                )

            payload: tuple[Any, ...] = (
                expansion.remaining,
                expansion.remaining_len,
                expansion.next_start,
                expansion.chosen,
            )
            lexical_item: search.ScoredItem = (priority, -serial, payload)
            pair_item: search.PairItem = (*pair_priority, -serial, payload)
            serial += 1
            expansions += 1
            search._push_bounded(lexical_heap, lexical_item, width)
            search._push_pair_bounded(pair_heap, pair_item, width)
            search._push_bounded(
                anchor_heaps[expansion.chosen[-1]],
                lexical_item,
                search.ANCHOR_CHAMPIONS_PER_WORD,
            )

    selected = search._select_multi_view(lexical_heap, pair_heap, anchor_heaps, width)
    return [
        (payload[0], payload[1], payload[2], payload[3])
        for payload in selected
    ], expansions


def _target_expansion(
    case: PreparedCase,
    target_state: search.BeamState,
    *,
    depth: int,
) -> search.BeamExpansion | None:
    word_count = len(case.target_indices)
    target_index = case.target_indices[depth]
    words_left_after = word_count - depth - 1
    rem, rem_len, start, chosen = target_state
    for expansion in search._iter_state_expansions(
        rem,
        rem_len,
        start,
        chosen,
        candidates=case.candidates,
        sparse_signatures=case.sparse_signatures,
        by_signature=case.by_signature,
        min_candidate_len=case.min_candidate_len,
        max_candidate_len=case.max_candidate_len,
        words_left_after=words_left_after,
        allow_repeat=True,
        branch_limit=search._branch_width(word_count),
    ):
        if expansion.index == target_index:
            return expansion
    return None


def _training_groups(
    case: PreparedCase,
    bigrams: generator.BigramModel,
) -> tuple[list[TrainingGroup], bool]:
    word_count = len(case.target_indices)
    total_letters = sum(case.target)
    states: list[search.BeamState] = [(case.target, total_letters, 0, ())]
    target_state = states[0]
    groups: list[TrainingGroup] = []

    for depth in range(max(0, word_count - 1)):
        next_states, _ = _expand_states(
            case,
            states,
            depth=depth,
            bigrams=bigrams,
            width=DATA_BEAM_WIDTH,
            model=None,
        )
        positive_expansion = _target_expansion(case, target_state, depth=depth)
        if positive_expansion is None:
            return groups, False
        words_left_after = word_count - depth - 1
        positive_pair = search._pair_priority(
            positive_expansion.chosen,
            case.candidates,
            bigrams,
        )
        positive_features = _features(
            positive_expansion,
            word_count=word_count,
            words_left_after=words_left_after,
            total_letters=total_letters,
            candidates=case.candidates,
            max_candidate_len=case.max_candidate_len,
            pair_priority=positive_pair,
        )

        negatives: list[FeatureVector] = []
        seen: set[tuple[int, ...]] = {positive_expansion.chosen}
        for rem, rem_len, start, chosen in next_states:
            if chosen in seen:
                continue
            seen.add(chosen)
            pseudo = search.BeamExpansion(
                index=chosen[-1],
                remaining=rem,
                remaining_len=rem_len,
                next_start=start,
                chosen=chosen,
            )
            pair_priority = search._pair_priority(chosen, case.candidates, bigrams)
            negatives.append(
                _features(
                    pseudo,
                    word_count=word_count,
                    words_left_after=words_left_after,
                    total_letters=total_letters,
                    candidates=case.candidates,
                    max_candidate_len=case.max_candidate_len,
                    pair_priority=pair_priority,
                )
            )
            if len(negatives) >= NEGATIVES_PER_DEPTH:
                break
        if negatives:
            groups.append(
                TrainingGroup(
                    case_key=case.key,
                    depth=depth + 1,
                    positive=positive_features,
                    negatives=tuple(negatives),
                )
            )

        states = next_states
        target_state = (
            positive_expansion.remaining,
            positive_expansion.remaining_len,
            positive_expansion.next_start,
            positive_expansion.chosen,
        )
    return groups, True


def _trace(
    case: PreparedCase,
    bigrams: generator.BigramModel,
    model: LinearModel | None,
) -> TraceResult:
    started = time.perf_counter()
    word_count = len(case.target_indices)
    states: list[search.BeamState] = [(case.target, sum(case.target), 0, ())]
    expansions = 0

    for depth in range(max(0, word_count - 1)):
        states, expanded = _expand_states(
            case,
            states,
            depth=depth,
            bigrams=bigrams,
            width=EVAL_BEAM_WIDTH,
            model=model,
        )
        expansions += expanded
        target_prefix = case.target_indices[: depth + 1]
        if not any(chosen == target_prefix for _, _, _, chosen in states):
            return TraceResult(False, depth + 1, expansions, time.perf_counter() - started)

    if word_count == 1:
        survived = True
    else:
        prefix = case.target_indices[:-1]
        final_index = case.target_indices[-1]
        survived = any(
            chosen == prefix
            and final_index >= start
            and case.candidates[final_index].sig == rem
            for rem, _rem_len, start, chosen in states
        )
    return TraceResult(
        survived,
        None if survived else word_count,
        expansions,
        time.perf_counter() - started,
    )


def main() -> int:
    prepared, bigrams, skipped = _prepare_cases()
    groups: list[TrainingGroup] = []
    branch_misses: list[str] = []
    usable_cases: list[PreparedCase] = []
    for case in prepared:
        case_groups, branch_complete = _training_groups(case, bigrams)
        if not case_groups:
            skipped.append(f"{case.key}:no-training-groups")
            continue
        groups.extend(case_groups)
        usable_cases.append(case)
        if not branch_complete:
            branch_misses.append(case.key)

    print(
        f"partial-ranker dataset: usable_cases={len(usable_cases)}, "
        f"groups={len(groups)}, pairwise_examples={sum(len(group.negatives) for group in groups)}, "
        f"skipped={len(skipped)}, branch_misses={len(branch_misses)}",
        flush=True,
    )
    if skipped:
        print("skipped sample: " + "; ".join(skipped[:10]), flush=True)
    if branch_misses:
        print("target outside local branch cutoff: " + ", ".join(branch_misses), flush=True)
    if len(usable_cases) < 12:
        raise RuntimeError("too few lexically valid benchmark cases for grouped ranker research")

    baseline_hits = 0
    learned_hits = 0
    improved: list[str] = []
    regressed: list[str] = []
    baseline_seconds = 0.0
    learned_seconds = 0.0
    held_out_count = 0
    fold_models: list[LinearModel] = []

    for fold in range(FOLDS):
        training = [group for group in groups if _fold(group.case_key) != fold]
        held_out = [case for case in usable_cases if _fold(case.key) == fold]
        if not training or not held_out:
            continue
        model = _train(training)
        fold_models.append(model)
        for case in held_out:
            baseline = _trace(case, bigrams, None)
            learned = _trace(case, bigrams, model)
            held_out_count += 1
            baseline_hits += int(baseline.survived)
            learned_hits += int(learned.survived)
            baseline_seconds += baseline.seconds
            learned_seconds += learned.seconds
            if learned.survived and not baseline.survived:
                improved.append(case.key)
            elif baseline.survived and not learned.survived:
                regressed.append(case.key)
            print(
                f"fold={fold} {case.key:<36} baseline={baseline.survived!s:<5} "
                f"learned={learned.survived!s:<5} "
                f"drop={baseline.drop_depth}/{learned.drop_depth} "
                f"seconds={baseline.seconds:.3f}/{learned.seconds:.3f}",
                flush=True,
            )

    if held_out_count == 0:
        raise RuntimeError("grouped cross-validation produced no held-out cases")

    print(flush=True)
    print(
        f"held-out target-path survival @ beam {EVAL_BEAM_WIDTH}: "
        f"baseline={baseline_hits}/{held_out_count}; learned={learned_hits}/{held_out_count}; "
        f"delta={learned_hits - baseline_hits:+d}",
        flush=True,
    )
    print(
        f"runtime seconds baseline/learned={baseline_seconds:.3f}/{learned_seconds:.3f}; "
        f"improved={improved or 'none'}; regressed={regressed or 'none'}",
        flush=True,
    )
    if fold_models:
        mean_weights = tuple(
            sum(model.weights[index] for model in fold_models) / len(fold_models)
            for index in range(len(FEATURE_NAMES))
        )
        ranked_weights = sorted(
            zip(FEATURE_NAMES, mean_weights, strict=True),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        print(
            "mean absolute-weight order: "
            + ", ".join(f"{name}={weight:+.3f}" for name, weight in ranked_weights),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
