from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from anagram_feature_ranker import (
    FEATURE_NAMES,
    LinearRankModel,
    RankGroup,
    RankItem,
    cross_validate_pairwise_ranker,
    evaluate_groups,
    explicit_order_features,
    fold_for_group,
    train_pairwise_ranker,
)


def _features(signal: float, *, phrase: float = 0.0) -> tuple[float, ...]:
    return explicit_order_features(
        grammar_norm=signal,
        structure_norm=signal,
        valency_norm=signal,
        syntax_coverage=signal,
        objective=signal,
        phrase_score=phrase,
        phrase_details={
            "whole_count": 1.0 if phrase else 0.0,
            "longer": phrase,
            "bigram_coverage": phrase,
            "cohesion": phrase,
            "cohesion_coverage": phrase,
            "cohesion_longest_fraction": phrase,
            "cohesion_segments": 1.0 if phrase else 0.0,
            "cohesion_frequency": phrase,
            "cohesion_splice_penalty": 0.0,
        },
        word_count=4,
    )


def _synthetic_groups(count: int = 20) -> list[RankGroup]:
    groups: list[RankGroup] = []
    for index in range(count):
        # Deliberately make the baseline prefer the negative candidate while the
        # explicit feature signal consistently identifies the positive one.
        positive = RankItem(
            key=f"good-{index}",
            features=_features(0.85, phrase=0.9),
            positive=True,
            baseline_score=0.2,
        )
        negative = RankItem(
            key=f"bad-{index}",
            features=_features(0.15, phrase=0.1),
            positive=False,
            baseline_score=0.8,
        )
        groups.append(RankGroup(key=f"bag-{index}", items=(positive, negative)))
    return groups


class ExplicitFeatureRankerTests(unittest.TestCase):
    def test_feature_vector_has_stable_schema_and_bounds(self) -> None:
        features = explicit_order_features(
            grammar_norm=1.4,
            structure_norm=-1.0,
            valency_norm=0.8,
            syntax_coverage=0.6,
            objective=0.75,
            phrase_score=0.9,
            phrase_details={
                "whole_count": 12.0,
                "longer": 0.7,
                "bigram_coverage": 0.5,
                "cohesion": 0.8,
                "cohesion_coverage": 1.0,
                "cohesion_longest_fraction": 0.75,
                "cohesion_segments": 2.0,
                "cohesion_frequency": 0.4,
                "cohesion_splice_penalty": 0.25,
            },
            word_count=5,
        )

        self.assertEqual(len(features), len(FEATURE_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in features))
        self.assertEqual(features[0], 1.0)
        self.assertEqual(features[1], 0.0)
        self.assertEqual(features[6], 1.0)
        self.assertEqual(features[12], 0.5)

    def test_model_json_round_trip_and_schema_guard(self) -> None:
        model = LinearRankModel(tuple(index / 10 for index in range(len(FEATURE_NAMES))))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranker.json"
            model.save(path)
            self.assertEqual(LinearRankModel.load(path), model)

            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["features"] = list(reversed(payload["features"]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "feature schema"):
                LinearRankModel.load(path)

    def test_pairwise_training_learns_explicit_signal(self) -> None:
        groups = _synthetic_groups()
        baseline = evaluate_groups(groups, None)
        model = train_pairwise_ranker(groups, epochs=20)
        learned = evaluate_groups(groups, model)

        self.assertEqual(baseline.recall1, 0.0)
        self.assertEqual(learned.recall1, 1.0)
        self.assertEqual(learned.mrr, 1.0)

    def test_group_fold_is_stable_and_keeps_bag_whole(self) -> None:
        first = fold_for_group("a b c", 5)
        self.assertEqual(first, fold_for_group("a b c", 5))
        self.assertTrue(0 <= first < 5)
        self.assertNotEqual(
            {fold_for_group(f"bag-{index}", 5) for index in range(20)},
            set(),
        )

    def test_cross_validation_is_held_out_by_group(self) -> None:
        baseline, learned = cross_validate_pairwise_ranker(
            _synthetic_groups(40),
            folds=5,
            epochs=20,
        )

        self.assertEqual(baseline.groups, 40)
        self.assertEqual(learned.groups, 40)
        self.assertEqual(baseline.recall1, 0.0)
        self.assertEqual(learned.recall1, 1.0)
        self.assertEqual(learned.mrr, 1.0)


if __name__ == "__main__":
    unittest.main()
