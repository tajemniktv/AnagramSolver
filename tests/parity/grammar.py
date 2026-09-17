"""Local grammar parity against Python's actual WordNet-backed scorer."""
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import anagram_rerank_core as reference
from scoring import close


def main():
    directory = ROOT / ".anagram_data/wordnet31/dict"
    lex = reference.WordNetLexicon.load(directory)
    rng = random.Random(917)
    vocabulary = sorted(reference.FUNCTION_WORDS | {"dog", "dogs", "cat", "cats", "run", "runs", "deep", "red", "turn", "give", "knowledge", "power", "quickly", "xyznonword"})
    cases = [rng.choices(vocabulary, k=rng.randint(0, 8)) for _ in range(250)]
    # Every function-word pair exercises class priority and collision rules.
    cases.append(vocabulary)
    cases.extend(phrase.split() for phrase in ["the young", "a professional", "the light faded",
                 "the repaired engine", "the mechanic repaired engines", "birds of a feather",
                 "mother of invention", "these old sheep", "a dogs", "school bus", "less is more"])
    subprocess.run(["cargo", "build", "--locked", "--example", "grammar_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/debug/examples" / ("grammar_probe.exe" if sys.platform == "win32" else "grammar_probe")
    result = subprocess.run([str(executable), str(directory)], input="\n".join(map(json.dumps, cases)),
                            text=True, capture_output=True, check=True, timeout=60)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(cases)
    for words, got in zip(cases, actual):
        expected = dict(classes=[reference.function_class(w) for w in words],
                        pairs=[[reference.pair_grammar(a,b,lex) for b in words] for a in words],
                        start=[reference.start_score(w,lex) for w in words],
                        end=[reference.end_score(w,lex) for w in words],
                        local=reference.local_grammar_raw(words,lex),
                        potential=reference.grammar_potential(words,lex), coverage=reference.content_coverage(words,lex),
                        np_start=[reference._np_span_starting_at(words,i,lex) for i in range(len(words))],
                        np_end=[reference._np_span_ending_at(words,i,lex) for i in range(len(words))])
        try:
            close(got, expected)
        except AssertionError as error:
            raise AssertionError((words, error)) from error
    print(f"Local grammar parity passed: {len(cases)} cases including all function-word pairs")


if __name__ == "__main__":
    main()
