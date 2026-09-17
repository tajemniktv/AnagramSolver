"""Seeded ordered-generation differential gate; no corpora or downloads needed."""
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import anagram_generate as reference


def main():
    rng = random.Random(20260917)
    vocabulary = ["".join(word) for size in range(1, 4)
                  for word in itertools.product("abc", repeat=size)]
    cases = []
    expected = []
    for _ in range(500):
        words = rng.sample(vocabulary, rng.randint(1, 16))
        case = dict(text="".join(rng.choices("abc", k=rng.randint(1, 9))),
                    candidates=words, min_words=1, max_words=rng.randint(1, 6),
                    cap=rng.choice([0, 1, 2, 5, 20]), repeat=rng.choice([True, False]),
                    clues=rng.sample(words, rng.randint(0, min(3, len(words)))),
                    initial=[], mode=rng.choice(["any", "exactly-one"]),
                    strategy=rng.choice(["prefix", "diverse"]))
        if case["clues"] and rng.random() < 0.3:
            case["initial"] = [rng.choice(case["clues"])]
        stats = reference.SearchStats()
        candidates = [reference.Candidate(w, reference.counts(w), len(w), 0) for w in words]
        bags = list(reference.search_solutions(reference.counts(case["text"]), candidates,
                    case["min_words"], case["max_words"], case["cap"], case["repeat"],
                    clue_words=set(case["clues"]), initial_clue_words=set(case["initial"]),
                    hint_mode=case["mode"], strategy=case["strategy"], stats=stats))
        cases.append(case)
        expected.append(dict(bags=[list(b) for b in bags], truncated=stats.truncated))
    subprocess.run(["cargo", "build", "--locked", "--example", "generation_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/debug/examples" / ("generation_probe.exe" if sys.platform == "win32" else "generation_probe")
    output = subprocess.run([str(executable)], input="\n".join(map(json.dumps, cases)),
                            text=True, capture_output=True, check=True, timeout=60)
    actual = [json.loads(line) for line in output.stdout.splitlines()]
    assert len(actual) == len(expected)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got == want, (index, cases[index], got, want)
    print(f"Ordered generation parity passed: {len(cases)} seeded cases")


if __name__ == "__main__":
    main()
