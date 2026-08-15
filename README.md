# AnagramSolver

A multi-word exact anagram solver that combines exact letter matching with lexical frequency, WordNet grammar/valency, retained word-order candidates, positive bigram evidence, and optional Wikimedia phrase evidence.

The normal user-facing entry point is `anagram_solver.py`. The lower-level generator, reranker, corpus builder, and benchmark scripts remain available for research and debugging.

## Quick start

Python 3.13 or newer is recommended.

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT"
```

On the first run the solver may download/cache its dictionary, WordNet data, and word-frequency data. Later runs reuse those caches.

Normal use runs a **balanced** search capped at 100,000 generated word bags. It is much more responsive than unlimited 2–6-word enumeration, but the cap means it **can miss the answer** if the correct bag occurs later in generation order.

For a faster exploratory pass:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --quick
```

`--quick` caps generation at 20,000 candidate word bags and can miss the answer.

For unlimited **candidate generation** with no generation cap:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --exhaustive
```

`--exhaustive` exhaustively generates matching word bags, but the user-facing reranker still deep-analyzes a bounded shortlist and only those deep-ranked rows are displayed. In other words, it removes **generation** truncation; it is not a promise that every generated bag receives full grammar/phrase analysis. It can become much slower when the word count is unknown or many short/common words fit the letter multiset. Supplying clues or an exact word count can reduce that search space dramatically.

## Common options

If you know the answer contains four words:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --words 4
```

Add clue words. At least one supplied hint must occur in the answer:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --hint dont --hint phone
```

Exclude known-bad words:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --exclude hit,oldie,lois
```

Require a known word:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --require hips
```

Show more results per word-count bucket:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --top 25
```

Admit rarer vocabulary by lowering the frequency cutoff:

```powershell
python anagram_solver.py "tommarvoloriddle" --hint voldemort --words 4 --min-zipf 0
```

The default `--min-zipf 2.7` is a broad normal-English filter. Set it to `0` when rare names/terms matter, with the understanding that the search space can grow substantially.

For scripts or other programs, JSON output is available:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --words 4 --json
```

Use `--verbose` to expose the underlying generator/reranker diagnostics.

## Optional Wikimedia phrase evidence

The solver works without a phrase database and still uses positive-only observed bigrams. A Wikimedia phrase database can provide stronger late-stage evidence for known titles, names, sayings, and other attested phrases.

Build a Wiktionary index:

```powershell
python build_wikimedia_phrase_index.py --output wikimedia_phrases.db --rebuild
```

Or include Wikipedia titles as well:

```powershell
python build_wikimedia_phrase_index.py --include-wikipedia --output wikimedia_phrases.db --rebuild
```

Then use it while solving:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --phrase-db wikimedia_phrases.db
```

The combined Wikipedia build is large and intentionally not performed automatically by the normal solver.

## Cached runs

The user-facing frontend stores intermediate candidate/reranked exports under:

```text
~/.multi_anagram/solver_runs/
```

The cache key includes the generation constraints, generation mode/cap, and generator source hash, so repeating the same search can skip candidate generation while changed constraints/source code create a new cache entry. Use `--rebuild` to force regeneration or `--work-root` to choose a different location.

## Research / low-level tools

`anagram_generate.py` generates exact canonical word bags and applies lexical/clue filtering.

`anagram_rerank.py` consumes a `candidates.txt` export and performs WordNet grammar/valency analysis, retained top-K ordering, positive bigram scoring, and optional phrase-index rescoring.

`anagram_benchmark.py` runs the ordering or end-to-end benchmark suite.

`build_wikimedia_phrase_index.py` builds the optional SQLite phrase index.

For example, the low-level exhaustive path remains available:

```powershell
python anagram_generate.py "ODITIHNSLSHEEEPT" --all-results --min-zipf 2.7 --export candidates.txt
python anagram_rerank.py candidates.txt --export reranked.txt
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Pull requests also run the fast phrase-order A/B benchmark. The expensive full corpus matrix is kept as an explicit manual workflow rather than running after every merge.
