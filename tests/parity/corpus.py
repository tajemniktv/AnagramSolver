"""Compare native dictionary admission and generated prefixes using real corpora."""
import json
from pathlib import Path
import subprocess
import sys
from itertools import product

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "archive"))
import anagram_generate as reference


def main():
    dictionary = ROOT / ".anagram_data/dictionary/large.txt"
    frequency = ROOT / ".anagram_data/ngrams/count_1w.txt"
    unigrams = reference.load_unigram_model(frequency)
    executable = ROOT / "target/debug" / ("anagram-cli.exe" if sys.platform == "win32" else "anagram-cli")
    for text in ("knowledgeispower", "thesehipsdontlie", "testing", "ateate"):
        for strategy, short_policy in product(("prefix", "diverse"), ("none", "common", "all")):
            request = dict(schema_version=1, text=text, required=[], hints=[], excluded=[],
                           min_words=1, max_words=4, min_word_length=3, max_word_length=30,
                           min_zipf=2.7, candidate_budget=20, allow_repeat=True,
                           strategy=strategy, hint_mode="any", short_policy=short_policy,
                           extra_short_words=["té"])
            candidates = reference.load_words(dictionary, reference.counts(text), 3, 30,
                         set(), [], set(), 2.7, short_policy, reference.DEFAULT_SHORT_WORDS | {"te"}, set(), unigrams)
            stats = reference.SearchStats()
            expected = list(reference.search_solutions(reference.counts(text), candidates,
                            1, 4, 20, True, strategy=strategy, stats=stats))
            result = subprocess.run([str(executable), "generate", str(dictionary), str(frequency)],
                                    input=json.dumps(request), text=True, capture_output=True,
                                    check=True, timeout=60)
            actual = json.loads(result.stdout)
            assert actual["vocabulary_size"] == len(candidates), (text, strategy, "vocabulary")
            assert actual["bags"] == [list(bag) for bag in expected], (text, strategy, "bags")
            assert (actual["stop"] == "candidate_cap") == stats.truncated
    print("Real-corpus admission and prefix parity passed: 24 short-policy cases")


if __name__ == "__main__":
    main()
