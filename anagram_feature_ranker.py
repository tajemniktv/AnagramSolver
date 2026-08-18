"""Dependency-free learning-to-rank over explicit AnagramSolver features.

This is deliberately not a language model. It learns a small linear pairwise
ranking function over grammar, structure, phrase and corpus-cohesion features
that the deterministic solver already exposes. Training and cross-validation
are grouped by unordered word bag so permutations of the same answer can never
leak across train/test folds.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MODEL_SCHEMA = "anagram-explicit-ranker-1"
FEATURE_NAMES = (
    "grammar_norm",
    "structure_norm",
    "valency_norm",
    "syntax_coverage",
    "objective",
    "phrase_score",
    "phrase_exact",
    "phrase_longer",
    "phrase_bigram_coverage",
    "cohesion",
    "cohesion_coverage",
    "cohesion_longest",
    "cohesion_compactness",
    "cohesion_frequency",
    "cohesion_splice_clean",
    "grammar_x_phrase",
    "structure_x_cohesion",
    "word_count_scaled",
)

FeatureVector = tuple[float, ...]


def _bounded(value: object, default: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return default
    number = float(value)
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


def _validate_feature_vector(features: Sequence[float]) -> None:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError("feature vector does not match ranker schema")
    if not all(math.isfinite(float(value)) for value in features):
        raise ValueError("ranker features must be finite")


def explicit_order_features(
    *,
    grammar_norm: float,
    structure_norm: float,
    valency_norm: float,
    syntax_coverage: float,
    objective: float,
    phrase_score: float = 0.0,
    phrase_details: Mapping[str, float] | None = None,
    word_count: int,
) -> FeatureVector:
    """Build the stable runtime/training feature vector for one complete order."""
    details = phrase_details or {}
    phrase = _bounded(phrase_score)
    grammar = _bounded(grammar_norm)
    structure = _bounded(structure_norm)
    valency = _bounded(valency_norm)
    coverage = _bounded(syntax_coverage)
    objective_norm = _bounded(objective)
    cohesion = _bounded(details.get("cohesion", 0.0))
    cohesion_coverage = _bounded(details.get("cohesion_coverage", 0.0))
    cohesion_longest = _bounded(details.get("cohesion_longest_fraction", 0.0))
    cohesion_frequency = _bounded(details.get("cohesion_frequency", 0.0))
    splice_penalty = _bounded(details.get("cohesion_splice_penalty", 0.0))
    segments_raw = details.get("cohesion_segments", 0.0)
    segments = float(segments_raw) if isinstance(segments_raw, (int, float)) else 0.0
    compactness = 1.0 / segments if segments > 0.0 else 0.0
    exact = 1.0 if float(details.get("whole_count", 0.0)) > 0.0 else 0.0
    longer = _bounded(details.get("longer", 0.0))
    bigram = _bounded(details.get("bigram_coverage", 0.0))
    wc = max(0.0, min(1.0, float(word_count) / 8.0))

    return (
        grammar,
        structure,
        valency,
        coverage,
        objective_norm,
        phrase,
        exact,
        longer,
        bigram,
        cohesion,
        cohesion_coverage,
        cohesion_longest,
        compactness,
        cohesion_frequency,
        1.0 - splice_penalty if cohesion > 0.0 else 0.0,
        grammar * phrase,
        structure * cohesion,
        wc,
    )


@dataclass(slots=True, frozen=True)
class RankItem:
    """One candidate order inside a word-bag ranking group."""

    key: str
    features: FeatureVector
    positive: bool
    baseline_score: float

    def __post_init__(self) -> None:
        _validate_feature_vector(self.features)
        if not math.isfinite(self.baseline_score):
            raise ValueError("rank item baseline score must be finite")


@dataclass(slots=True, frozen=True)
class RankGroup:
    """Candidates that compete only with permutations of the same word bag."""

    key: str
    items: tuple[RankItem, ...]


@dataclass(slots=True, frozen=True)
class RankMetrics:
    groups: int
    recall1: float
    mrr: float


@dataclass(slots=True, frozen=True)
class LinearRankModel:
    """Small inspectable scoring model over FEATURE_NAMES."""

    weights: FeatureVector

    def __post_init__(self) -> None:
        if len(self.weights) != len(FEATURE_NAMES):
            raise ValueError("ranker weight count does not match feature schema")
        if not all(math.isfinite(weight) for weight in self.weights):
            raise ValueError("ranker weights must be finite")

    def score(self, features: Sequence[float]) -> float:
        _validate_feature_vector(features)
        return sum(
            weight * float(value)
            for weight, value in zip(self.weights, features, strict=True)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "features": list(FEATURE_NAMES),
            "weights": list(self.weights),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> LinearRankModel:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported explicit-ranker model schema")
        if payload.get("features") != list(FEATURE_NAMES):
            raise ValueError("ranker feature schema does not match this solver")
        raw = payload.get("weights")
        if not isinstance(raw, list) or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in raw
        ):
            raise ValueError("ranker weights must be a numeric list")
        return cls(tuple(float(value) for value in raw))


def _sigmoid_negative_margin(margin: float) -> float:
    """Return sigmoid(-margin) without overflowing on confident pairs."""
    if margin >= 0.0:
        exp_neg = math.exp(-min(margin, 700.0))
        return exp_neg / (1.0 + exp_neg)
    exp_pos = math.exp(max(margin, -700.0))
    return 1.0 / (1.0 + exp_pos)


def train_pairwise_ranker(
    groups: Sequence[RankGroup],
    *,
    epochs: int = 80,
    learning_rate: float = 0.08,
    l2: float = 0.002,
) -> LinearRankModel:
    """Fit deterministic pairwise logistic ranking with L2 regularization."""
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and > 0")
    if not math.isfinite(l2) or l2 < 0.0:
        raise ValueError("l2 must be finite and >= 0")

    weights = [0.0] * len(FEATURE_NAMES)
    ordered_groups = sorted(groups, key=lambda group: group.key)
    step = 0

    for _ in range(epochs):
        for group in ordered_groups:
            positives = [item for item in group.items if item.positive]
            negatives = [item for item in group.items if not item.positive]
            if not positives or not negatives:
                continue
            for positive in positives:
                for negative in negatives:
                    _validate_feature_vector(positive.features)
                    _validate_feature_vector(negative.features)
                    diff = [
                        p - n
                        for p, n in zip(
                            positive.features,
                            negative.features,
                            strict=True,
                        )
                    ]
                    margin = sum(
                        weight * delta
                        for weight, delta in zip(weights, diff, strict=True)
                    )
                    probability = _sigmoid_negative_margin(margin)
                    step += 1
                    rate = learning_rate / math.sqrt(1.0 + 0.0005 * step)
                    for index, delta in enumerate(diff):
                        gradient = probability * delta - l2 * weights[index]
                        weights[index] += rate * gradient

    return LinearRankModel(tuple(weights))


def _rank(items: Sequence[RankItem], model: LinearRankModel | None) -> list[RankItem]:
    if model is None:
        return sorted(items, key=lambda item: (-item.baseline_score, item.key))
    return sorted(
        items,
        key=lambda item: (-model.score(item.features), -item.baseline_score, item.key),
    )


def evaluate_groups(
    groups: Sequence[RankGroup],
    model: LinearRankModel | None,
) -> RankMetrics:
    """Evaluate only groups that contain at least one positive candidate."""
    reciprocal: list[float] = []
    top1 = 0
    for group in groups:
        ranked = _rank(group.items, model)
        positive_ranks = [
            index
            for index, item in enumerate(ranked, 1)
            if item.positive
        ]
        if not positive_ranks:
            continue
        best = min(positive_ranks)
        reciprocal.append(1.0 / best)
        top1 += int(best == 1)
    if not reciprocal:
        return RankMetrics(0, 0.0, 0.0)
    count = len(reciprocal)
    return RankMetrics(count, top1 / count, sum(reciprocal) / count)


def fold_for_group(key: str, folds: int) -> int:
    if folds < 2:
        raise ValueError("folds must be >= 2")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def cross_validate_pairwise_ranker(
    groups: Sequence[RankGroup],
    *,
    folds: int = 5,
    epochs: int = 80,
    learning_rate: float = 0.08,
    l2: float = 0.002,
) -> tuple[RankMetrics, RankMetrics]:
    """Return pooled held-out baseline/model metrics with bag-wise split isolation."""
    if folds < 2:
        raise ValueError("folds must be >= 2")

    baseline_reciprocal: list[float] = []
    model_reciprocal: list[float] = []
    baseline_top1 = 0
    model_top1 = 0

    for fold in range(folds):
        training = [group for group in groups if fold_for_group(group.key, folds) != fold]
        held_out = [group for group in groups if fold_for_group(group.key, folds) == fold]
        if not training or not held_out:
            continue
        model = train_pairwise_ranker(
            training,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
        )
        for group in held_out:
            baseline_ranked = _rank(group.items, None)
            model_ranked = _rank(group.items, model)
            base_ranks = [i for i, item in enumerate(baseline_ranked, 1) if item.positive]
            learned_ranks = [i for i, item in enumerate(model_ranked, 1) if item.positive]
            if not base_ranks or not learned_ranks:
                continue
            base = min(base_ranks)
            learned = min(learned_ranks)
            baseline_reciprocal.append(1.0 / base)
            model_reciprocal.append(1.0 / learned)
            baseline_top1 += int(base == 1)
            model_top1 += int(learned == 1)

    count = len(model_reciprocal)
    if count == 0:
        empty = RankMetrics(0, 0.0, 0.0)
        return empty, empty
    return (
        RankMetrics(count, baseline_top1 / count, sum(baseline_reciprocal) / count),
        RankMetrics(count, model_top1 / count, sum(model_reciprocal) / count),
    )
