"""Real WordNet lexical-feature/frame differential gate."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import anagram_rerank_core as reference


def main():
    directory = ROOT / ".anagram_data/wordnet31/dict"
    lexicon = reference.WordNetLexicon.load(directory)
    rng = random.Random(1709)
    vocabulary = sorted(lexicon.nouns | lexicon.verbs | lexicon.adjs | lexicon.advs)
    words = set(rng.sample(vocabulary, 1000)) | reference.FUNCTION_WORDS
    for word in rng.sample(sorted(lexicon.verbs), 300):
        words.update((word + "s", word + "ed", word + "ing"))
    words.update(lexicon.noun_exc)
    words.update(lexicon.verb_exc)
    words.update(["DON’T", "école", "", "xyznonword", "stopped", "planned", "studies", "lies"])
    words = sorted(words)
    methods = [lexicon.allows_object, lexicon.allows_intransitive, lexicon.allows_pp,
               lexicon.allows_predicative, lexicon.allows_object_predicative,
               lexicon.allows_infinitive_or_gerund, lexicon.allows_clausal]
    subprocess.run(["cargo", "build", "--locked", "--example", "wordnet_probe"], cwd=ROOT, check=True)
    executable = ROOT / "target/debug/examples" / ("wordnet_probe.exe" if sys.platform == "win32" else "wordnet_probe")
    result = subprocess.run([str(executable), str(directory)], input="\n".join(words) + "\n",
                            encoding="utf-8", capture_output=True, check=True, timeout=60)
    actual = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(actual) == len(words)
    for word, got in zip(words, actual):
        expected = dict(features=asdict(lexicon.features(word)),
                        bases=sorted(lexicon.verb_base_lemmas(word)), frames=sorted(lexicon.frames_for(word)),
                        allows=[method(word) for method in methods])
        assert got == expected, (word, got, expected)
    print(f"WordNet parity passed: {len(words)} words, morphology, frames and valency categories")


if __name__ == "__main__":
    main()
