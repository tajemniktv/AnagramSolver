"""Capture/check small deterministic outputs of the pinned Python oracle."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from capture_reference import REFERENCE


def verify_sources():
    manifest = json.loads((ROOT / "tests/reference/manifest.json").read_text())
    assert manifest["reference_commit"] == REFERENCE
    subprocess.run(["git", "diff", "--exit-code", REFERENCE, "--",
                    *[f["path"] for f in manifest["source_files"]]],
                   cwd=ROOT, check=True, capture_output=True)
    for item in manifest["corpus_files"]:
        path = ROOT / item["path"]
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], path


def capture():
    import anagram_generate as g
    import anagram_solver as cli
    from anagram_feature_ranker import FEATURE_NAMES, LinearRankModel, RankItem, _rank
    from anagram_rerank_core import PhraseIndex
    import sqlite3

    normalized = [{"input": s, "letters": g.normalize_letters(s)} for s in
                  ["", "!!!", "A té—A", "Straße", "Æ Œ ø Ł", "ﬃ ＡＢＣ", "e\u0301", "猫🙂"]]
    generation = []
    for text, required, hints, excluded in [
        ("ate", [], [], []), ("ateateate", ["eat", "eat"], [], []),
        ("ateate", [], ["ate", "eat"], []), ("ate", [], [], ["eat"]),
        ("zzz", [], [], []), ("ate", [], ["zzz"], []),
    ]:
        remaining = g.counts(text)
        for word in required:
            remaining = g.subtract_counts(remaining, g.counts(word))
        words = [w for w in ["ate", "eat", "tea"] if w not in excluded]
        candidates = [g.Candidate(w, g.counts(w), len(w), 0) for w in words]
        for strategy in ["prefix", "diverse"]:
            for cap in [0, 1, 3, 10]:
                for mode in ["any", "exactly-one"]:
                    stats = g.SearchStats()
                    bags = list(g.search_solutions(remaining, candidates, 1, 3, cap, True,
                                clue_words=set(hints), initial_clue_words=set(required) & set(hints),
                                hint_mode=mode, strategy=strategy, stats=stats))
                    generation.append(dict(input=text, required=required, hints=hints, excluded=excluded,
                        vocabulary=words, min_residual_words=1, max_residual_words=3,
                        repeat=True, strategy=strategy, cap=cap, hint_mode=mode,
                        bags=[required + list(b) for b in bags], truncated=stats.truncated))
    validation = []
    for argv in [["!!!"], ["ate", "--exhaustive", "--max-results", "1"],
                 ["ate", "--require", "eat"], ["ateate", "--require", "eat,eat"],
                 ["ateate", "--require", "eat", "--words", "1"],
                 ["ate", "--min-words", "3", "--max-words", "2"],
                 ["ate", "--min-zipf", "nan"], ["ate", "--exhaustive"],
                 ["ate", "--quick"], ["ate", "--max-results", "3"]]:
        args = cli.build_parser().parse_args(argv)
        try:
            cli._validate_args(args)
            value = dict(mode=cli._generation_mode(args), cap=cli._generation_cap(args),
                         residual_bounds=cli._residual_word_limits(args))
        except SystemExit as error:
            value = dict(error=str(error))
        validation.append(dict(argv=argv, output=value))
    phrase_rows = [("the dog", 2, 100), ("dog runs", 2, 50), ("the dog runs", 3, 20)]
    with tempfile.TemporaryDirectory(dir=ROOT / ".codex/temp", prefix="reference-phrase-") as temp:
        path = Path(temp) / "phrases.sqlite"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE ngrams(text TEXT PRIMARY KEY, n INTEGER, count INTEGER)")
        connection.executemany("INSERT INTO ngrams VALUES (?, ?, ?)", phrase_rows)
        connection.commit()
        connection.close()
        index = PhraseIndex.open(path)
        try:
            phrase = [dict(words=words, score=index.score(words)) for words in
                      [("the", "dog", "runs"), ("runs", "dog", "the"), ("missing", "evidence")]]
        finally:
            index.connection.close()
    model = LinearRankModel(tuple(0.25 if i == 0 else 0.0 for i in range(len(FEATURE_NAMES))))
    items = [RankItem(key, tuple([value] + [0.0] * 17), False, baseline)
             for key, value, baseline in [("b", 1.0, 0.2), ("a", 1.0, 0.2), ("c", 0.0, 0.9)]]
    return dict(reference_commit=REFERENCE, normalization=normalized, generation=generation,
                frontend_validation=validation,
                optional_phrase=dict(origin="project-authored synthetic fixture", rows=phrase_rows, outputs=phrase),
                optional_model=dict(origin="project-authored synthetic fixture", model=model.to_dict(),
                    items=[asdict(item) for item in items],
                    scores=[model.score(item.features) for item in items],
                    ranked=[item.key for item in _rank(items, model)],
                    disabled=[item.key for item in _rank(items, None)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    verify_sources()
    value = json.dumps(capture(), indent=2, ensure_ascii=True) + "\n"
    path = ROOT / "tests/reference/behavior.json"
    if args.check:
        assert path.read_text(encoding="utf-8") == value, "Frozen behavior drift"
    else:
        path.write_text(value, encoding="utf-8")
    print("Pinned oracle behavior fixtures " + ("verified" if args.check else "captured"))


if __name__ == "__main__":
    main()
