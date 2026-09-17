"""Differential numeric and tie-order tests for native scoring primitives."""
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import anagram_generate as reference


def close(got, expected):
    if isinstance(expected, float):
        assert math.isclose(got, expected, abs_tol=1e-12, rel_tol=1e-12), (got, expected)
    elif isinstance(expected, dict):
        assert got.keys() == expected.keys()
        for key in expected:
            close(got[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(got) == len(expected)
        for a, b in zip(got, expected):
            close(a, b)
    else:
        assert got == expected, (got, expected)


def main():
    rng = random.Random(1709)
    vocabulary = ["a", "is", "cat", "dog", "the", "runs", "dont", "missing",
                  "run", "cats", "dogged", "running", "studies", "study", "studie", "does"]
    cases, expected = [], []
    for index in range(200):
        words = rng.choices(vocabulary, k=rng.randint(0, 7))
        counts = {word: rng.randint(1, 1000) for word in vocabulary[:-1]}
        # Include all-tied fixtures to prove reference traversal tie behavior.
        if index % 10 == 0:
            counts = {word: 10 for word in vocabulary}
        pairs = {(a, b): rng.randint(1, 10) for a in vocabulary for b in vocabulary
                 if rng.random() < 0.2 and index % 10 != 0}
        values = [float(rng.randint(0, 3)) for _ in range(rng.randint(0, 12))]
        unigram = reference.UnigramModel(counts, sum(counts.values()))
        bigram = reference.BigramModel(unigram, pairs)
        hints = set(rng.sample(vocabulary, rng.randint(0, 3)))
        bags = [rng.choices(vocabulary, k=rng.randint(1, 5)) for _ in range(12)]
        bags = [bag for bag in bags if not hints or hints.intersection(bag)]
        records = reference.build_records(bags, [], hints, "any", unigram, bigram, set(vocabulary), {"a", "is"})
        record_fields = ["words", "matched_hints", "family", "family_best_lex", "family_size",
                         "hint_info", "pair_raw", "pair_coverage", "lex_pct", "family_pct", "pair_pct", "pre_score"]
        expected_records = []
        for record in records:
            item = {field: getattr(record, field) for field in record_fields}
            item["lexical"] = {field: getattr(record, field) for field in ["lex_raw", "avg_zipf", "min_zipf", "junk_penalty"]}
            expected_records.append(item)
        cases.append(dict(words=words, bags=bags, hints=sorted(hints), short=["a", "is"], vocabulary=vocabulary,
                          unigrams="".join(f"{w}\t{c}\n" for w, c in counts.items()),
                          bigrams="".join(f"{a} {b}\t{c}\n" for (a, b), c in pairs.items()), values=values))
        expected.append(dict(lexical=reference.lexical_details(words, unigram, {"a", "is"}),
                             pair=reference.bigram_pair_potential(words, bigram),
                             order=reference.best_word_order(words, bigram),
                             percentiles=reference.percentile_map(values), records=expected_records))
    subprocess.run(["cargo", "build", "--locked", "--example", "scoring_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/debug/examples" / ("scoring_probe.exe" if sys.platform == "win32" else "scoring_probe")
    result = subprocess.run([str(executable)], input="\n".join(map(json.dumps, cases)),
                            text=True, capture_output=True, check=True, timeout=60)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        try:
            close(got, want)
        except AssertionError as error:
            raise AssertionError((index, cases[index], error)) from error
    print(f"Scoring component parity passed: {len(cases)} seeded cases")


if __name__ == "__main__":
    main()
