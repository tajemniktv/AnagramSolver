# AnagramSolver

A multi-word exact anagram solver that combines exact letter matching with lexical frequency, WordNet grammar/valency, retained word-order candidates, positive bigram evidence, and optional Wikimedia phrase evidence.

The normal user-facing entry point is `anagram_solver.py`. The lower-level generator, reranker, corpus builder, and benchmark scripts remain available for research and debugging.

## Quick start

Python 3.13 or newer is recommended.

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT"
```

On the first run the solver may download/cache its dictionary, WordNet data, and word-frequency data. Runtime data lives under `.anagram_data/` next to the scripts, so copying the project directory also carries its caches. The directory is git-ignored. Later runs reuse those caches.

Normal use runs a **balanced** search capped at 100,000 generated word bags. It is much more responsive than unlimited 2–6-word enumeration, but the cap means it **can miss the answer** if the correct bag occurs later in generation order.

Use `--search-strategy diverse` to distribute a bounded budget round-robin across feasible word counts and supplied alternative clues. Exhausted groups give their unused share to the remaining groups, and duplicate bags never consume the budget twice. This improves coverage of later word counts; it does not guarantee better answer recall for every puzzle. The default remains `--search-strategy prefix`: broader coverage can make deep ranking much slower because it admits more long word bags. Unlimited generation retains its historical enumeration order.

The final output reports generated bags, deep-analyzed bags, and whether more matches actually exist beyond the generation cap. The solver probes one additional unique bag to distinguish reaching the cap from exhausting the search at exactly that count. This probe can add search time. A no-match result explains whether any usable vocabulary remained; contradictory required/excluded words and impossible clues produce explicit errors rather than silently relaxing constraints.

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

For a custom bounded search budget:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --max-results 5000
```

`--max-results` must be positive and cannot be combined with `--exhaustive`. It overrides the balanced/quick generation cap, not the deep-ranking shortlist size. JSON output includes a `search` object with the cap, strategy, generated/deep-analyzed counts, truncation, vocabulary size, generation duration (`seconds`), and cache-hit flags. When candidates are reused, `seconds` describes their original generation, not the current invocation's latency.

## Optional Wikimedia phrase evidence

The solver works without a phrase database and still uses positive-only observed bigrams. A Wikimedia phrase database can provide stronger late-stage evidence for known titles, names, sayings, and other attested phrases.

Build a Wiktionary index:

```powershell
python build_wikimedia_phrase_index.py --rebuild
```

Or include Wikipedia titles as well:

```powershell
python build_wikimedia_phrase_index.py --include-wikipedia --rebuild
```

Then use it while solving:

```powershell
python anagram_solver.py "ODITIHNSLSHEEEPT" --phrase-db .anagram_data/phrase_indexes/wikimedia_phrases.db
```

The combined Wikipedia build is large and intentionally not performed automatically by the normal solver.

## Cached runs

The user-facing frontend stores intermediate candidate/reranked exports under:

```text
.anagram_data/solver_runs/
```

All default generated/cache directories are children of `.anagram_data/`: solver runs, dictionaries, n-grams, WordNet, prepared rows, Wikimedia title downloads, phrase indexes, and benchmark artifacts. The cache key includes the generation constraints, generation mode/cap, and generator source hash, so repeating the same search can skip candidate generation while changed constraints/source code create a new cache entry. Use `--rebuild` to force regeneration or `--work-root` to choose a different location.

Completed rankings are also reused. Their separate cache identity includes candidate content, ranking options, engine source, and runtime corpus identity/size/modification times (including an optional phrase database and its WAL). Changing `--top` only changes display, so it can reuse the same ranking. Normal frontend runs keep their prepared-row caches inside the run directory; low-level reranker defaults remain unchanged.

Candidate and ranking caches are integrity-checked and regenerated if damaged. Concurrent runs write private temporary files before atomic publication. Interrupted downloads similarly preserve previous corpus files. `--rebuild` bypasses cached generation, prepared rows, and final rankings. Corpus files edited in place should retain normal modification-time updates; if a tool deliberately preserves both size and timestamp, use `--rebuild`.

## Performance and integration checks

The hot-path probe remains available with `python ci_performance_probe.py`. For actual CLI latency, memory, candidate coverage, and cold/warm comparison:

```powershell
python -m pip install psutil==7.2.2
python benchmark_user_runs.py --case phone_charge --case user_testing_anagrams --cap 2000 --samples 3
```

The default case selection is the registry's `performance` suite. Both search strategies use the same bag cap and per-invocation timeout. The JSON report is written to `.anagram_data/benchmarks/user_performance.json` and includes source/platform identity, individual samples, median latency, memory, and expected-bag recall. These are observational measurements, not hard cross-machine speed thresholds.

Here **cold** means empty candidate, prepared-row, and ranking caches for the puzzle. Runtime corpora are provisioned beforehand; download latency and OS page-cache clearing are deliberately excluded. Memory is the sampled sum of solver/descendant RSS, so short peaks can be missed and shared pages can be counted more than once.

After provisioning runtime corpora, enable real-subprocess integration tests:

```powershell
$env:ANAGRAM_INTEGRATION = "1"
python -m unittest discover -s tests -p test_cli_integration.py -v
```

CI runs unit tests on Windows and Linux with Python 3.13/3.14, and real CLI/cache-recovery tests on both operating systems. Integration tests use temporary puzzle caches and do not remove existing user caches.

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

## Tests and shared case registry

```powershell
python -m unittest discover -s tests -v
```

All **shared real-world test/CI cases** live in one case-centric registry: `anagram_benchmarks.json`. The Python API in `anagram_suite.py` exposes `cases_for(...)`, and the scenario-driven CI runners select their workloads from it:

- `ordering` — blocking ordering regression benchmark;
- `phrase_ordering` — phrase/corpus ordering A/B;
- `normal_user_cli` — real `anagram_solver.py` invocations;
- `full` — full generation + reranking matrix;
- `performance` — repeatable ordering/deep-analysis workload;
- `refinement` — forced-beam k-opt experiment;
- `feature_ranker` — grouped ranker training/evaluation.

A case with **no `suites` field defaults to all suites**. Existing cases use explicit `core`, `full`, `performance`, or CLI memberships where needed to preserve the intended runtime. `core` expands to ordering, phrase ordering, refinement, and feature-ranker evaluation; `all` expands to every suite.

A case can own common solver options plus suite-specific overrides. For example:

```json
{
  "id": "my_new_anagram",
  "target": "IAMTESTINGANAGRAMS",
  "answer": "i am testing anagrams",
  "solver": {
    "hints": ["testing"],
    "words": 4,
    "min_zipf": 2.7
  },
  "normal_user_cli": {
    "verbose": true,
    "timeout_seconds": 120
  },
  "ordering": {
    "max_rank": 10
  }
}
```

Because `suites` is omitted, that single object participates in every registry-backed CI suite. Add `"suites": ["core", "normal_user_cli"]` (or any explicit suite list) to restrict it. Set `"enabled": false` to temporarily disable a case without deleting it.

There is also a small management CLI, so routine edits do not require hand-editing JSON:

```powershell
python anagram_suite.py list
python anagram_suite.py list --suite normal_user_cli
python anagram_suite.py add my_new_anagram --target IAMTESTINGANAGRAMS --answer "i am testing anagrams" --hint testing --words 4 --verbose
python anagram_suite.py disable my_new_anagram
python anagram_suite.py enable my_new_anagram
python anagram_suite.py remove my_new_anagram
python anagram_suite.py validate
```

`add` defaults to **all suites**. Use repeated `--suite` flags to restrict a new case, for example `--suite core --suite normal_user_cli`.

Small synthetic fixtures that exist only to exercise one function or subsystem should still stay beside that subsystem's unit tests. Test modules are grouped by behavior rather than by the PR/review that introduced a regression.

Pull requests also run the fast phrase-order A/B benchmark. The expensive full corpus matrix is kept as an explicit manual workflow rather than running after every merge.
