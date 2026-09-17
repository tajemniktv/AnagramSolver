# Native tools and remaining compatibility boundaries

Build `cargo build -p anagram-cli --release` to obtain `target/release/anagram-cli.exe`.
These are optional maintenance/experiment tools; the desktop solver never launches
the CLI and does not need Python. Tool output is JSON. `--timeout-ms N` also applies
to these commands. Existing model/dictionary/database outputs are never overwritten:
choose a new path and select it in desktop Settings after successful preparation.

## Learned order ranking

1. Prepare a JSON array of cases, for example:
   `[ {"answer":"we are home","acceptable_orders":["we are home"]}, ... ]`.
   Use multiple distinct bags. All orders of a bag stay in the same validation fold.
2. `anagram-cli ranker-build WORDNET_DICT [PHRASE_DB]` reads that array from stdin
   and writes `{groups, skipped}`. It enumerates exact orders for 2–6 words and
   retains up to 56. Missing acceptable orders are reported as skipped, never
   inserted into the pool as an artificial positive. Combine multiple acceptable
   orders into one case rather than repeating the same bag.
3. `anagram-cli ranker-train NEW_MODEL.json` reads that output from stdin. Optional
   top-level `options` contains `folds`, `epochs`, `learning_rate`, `l2` (defaults
   5, 80, 0.08, 0.002). It writes the model file and reports baseline and held-out
   recall@1/MRR. At least two occupied hash folds are required. The final saved
   all-group fit is **not** used to calculate held-out metrics.
4. Select the model in desktop Settings → Experimental, or pass it to the native solver:
   `anagram-cli solve DICTIONARY UNIGRAMS BIGRAMS WORDNET - MODEL.json`.
   Replace `-` with a phrase database if desired. Solve requests accept optional
   `workers` (0 for automatic, 1–32 explicit, default 1) and `refine` (default false).
5. For offline experiments, `anagram-cli ranker-rank MODEL.json` reads an array
   of `{key, features, baseline_score}` from stdin and returns ranked indices and
   loaded-model identity. The 18-feature schema is in `learned.rs`.

Alternatively, save the `ranker-build` output as JSON and use Settings → Experimental
to train locally. The desktop limits input to 4 MiB and 1,000 groups, with one worker
and a 60-second deadline. Epochs (1–200), folds (2–10), learning rate and L2 are
explicit controls. The resulting quality report records those options.
Cancel and app shutdown stop the worker. Successful models get unique files in
`user-data/models` beside the installed app; they are not automatically enabled.
Select the model and save settings to apply it. No source downloads or dataset
preparation run automatically. The Experimental tab can also build datasets from
the labelled case JSON described in step 1 and reopen saved `.report.json` quality
reports. Authoring the correct answers remains a user-owned step.
The native builder additionally accepts `{ "cases": [...], "options": {
"retained_orders": 56, "phrase_bonus_max": 10.0 } }`; the desktop exposes both
options. Plain case arrays retain the historical defaults. Standalone `ranker-rank`
is available from Experimental without training or model activation.

Settings → Data exposes native downloads, offline preparation, non-destructive
updates and validation. Settings → Diagnostics exposes last-solve provenance,
timings and cache information. The desktop links the same maintenance library as
the CLI; it does not invoke a CLI executable or Python process.

Desktop learned selection operates on the retained-order corpus shortlist. It does
not imply a quality improvement; evaluate a model on representative held-out cases.
Standard final scores are retained so their existing interpretation does not change.

## Corpus preparation and updates

Phrase building automatically uses up to eight workers for normalization and
phrase extraction. Bounded batches combine duplicate phrases before sorted SQLite
writes; a 64-MiB page cache and `WITHOUT ROWID` table reduce storage and random I/O.
Decompression and database writes remain serial. Output includes `workers` and
`database_updates`; phrase/count semantics and no-overwrite publication are unchanged.

- `anagram-cli prepare-dictionary BASE UNIGRAMS NEW_DICTIONARY` normalizes and
  deduplicates the base vocabulary, adds the established contraction spellings,
  and reports corpus-common two-letter words (Zipf >= 5) for the extra-short list.
- `anagram-cli build-phrases NEW_DB TITLE_FILE[.gz] ...` ingests one or more local
  Wikimedia-style title dumps. It accepts 2–8-word titles, stores whole titles
  and 2–5-word subphrases, and aggregates duplicate evidence into unique SQLite
  text keys. Inputs may be plain text or gzip. Outputs are published only after
  successful completion; cancelled/failed builds never publish a partial database.
- `pwsh -File tools/prepare-native-data.ps1 -NativeCli PATH_TO_CLI -OutputDirectory NEW_DIRECTORY`
  explicitly downloads the dictionary, Norvig counts and WordNet 3.1, prepares the
  dictionary, and records file hashes. Add `-IncludeWiktionary` and/or
  `-IncludeWikipedia` to download title dumps and build a combined phrase database.
  It never changes the active desktop settings or overwrites an existing corpus
  directory. Source downloads stay in project `.codex/temp` for inspection/retry.
  `-SourceDirectory DIR` uses already-downloaded `base.txt`, `count_1w.txt`,
  `count_2w.txt`, `wordnet.tar.gz`, and optional `enwiktionary.gz`/`enwiki.gz`
  instead of network access. Dataset licensing still governs redistribution.

Use a new versioned folder beneath the app's `corpora` directory for an update,
then choose its files in Settings. The previous corpus is preserved for rollback.
Automatic background updates/downloads are deliberately not enabled.

## Deliberately retained differences

- Rust regex excludes backreferences and look-around; invalid patterns are errors.
- Phrase databases must have a real `ngrams` table with unique text keys.
- Python pickle caches and historical command-line flag syntax are not imported.
- Extended desktop mode still respects engine correctness limits (10-word ordering),
  but no longer caps result counts. Pagination is not truncation. Standard bounded
  mode remains the safe default; cache and execution limits can be customized.
- The website, multi-platform packaging and full combined-corpus benchmark matrix
  remain deferred. Existing Python scripts remain available; native features need
  not reproduce every experimental script's command-line interface.
