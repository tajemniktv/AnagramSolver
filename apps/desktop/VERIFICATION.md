# Phase 4 verification — completed on this PC

## TajsAnagrams remaining desktop/CLI controls, 2026-09-17

- Generation-only execution loads no ranking corpora, exports every bag and reports
  truthful cancellation/deadline status. A pre-start cancellation race now always
  publishes a terminal status rather than leaving the status absent.
- Independent dictionary/phrase preparation and standalone model ranking are
  available in Settings. Training pool retention/phrase weight and optimizer
  learning rate/L2 are configurable; report files retain optimizer settings.
- Workers=0 selects automatic parallelism. Custom cache path/size and execution
  limits are persisted. Extended UI caps are lifted and result rendering paginated.
- Twelve desktop tests, four native-tools integration tests and ten core progress
  tests pass, including unranked export without WordNet, custom zero-timeout,
  custom cache reuse/clear, >100-row Extended admission, automatic-worker parity,
  phrase-weight effects, independent preparation, standalone ranking and legacy
  report path relocation. Strict Clippy, Svelte checks and 16 contract artifacts pass.
- No manual experimental training test or full real-corpus quality benchmark was
  performed. These are functional checks, not a learned-quality improvement claim.
- Normal release build installed and restarted as TajsAnagrams (PID 54936), SHA-256
  `DDB6E4437A1D22946FEE2294D9F0EA8216ACED3AF3C46D553F79F882B476D5DE`.
  Generation-only accepts word counts above the ranker's ten-word limit. Existing
  corpora were preserved. Obsolete `backup/AnagramSolver.exe` was removed after
  verifying the renamed backup; `backup/TajsAnagrams.exe` is the one rotating backup.

## Parallel phrase preparation, 2026-09-17

- Up to eight parsing workers, bounded 8,192-line / 4-MiB batches, duplicate
  aggregation and sorted writes. SQLite uses a 64-MiB cache and `WITHOUT ROWID`;
  journaling/durability defaults and staged no-overwrite publication remain intact.
- Three CLI integration tests and ten desktop tests pass; strict CLI/desktop
  Clippy passes. Coverage includes multi-batch serial/parallel count equality,
  gzip/plain sources, repeats, invalid late input, cancellation and no clobber.
- Release benchmark: 150,000 synthetic titles, 165,004 identical output rows.
  Before: 8.188 s / 11,317,248 bytes. After: 1.606 s / 5,214,208 bytes, 8 workers.
  Every SQLite row was compared. This is not a full Wikimedia benchmark.
- Svelte check and release build passed. After the user confirmed the old job had
  finished, installed and restarted PID 60908 with SHA-256
  `6B67BFFB68B5EFC014A63EA9FDAB522B655CF535B0C4492DF045B653B51DFE3C`.
  Existing corpus sets were not rebuilt or replaced.

## Wikimedia download identification fix, 2026-09-17

- Reproduced the Wiktionary dump HTTP 403 with no User-Agent. Identifying the
  application and its repository satisfies Wikimedia's enforced User-Agent policy:
  both Wiktionary and Wikipedia returned HTTP 200 to HEAD and HTTP 206 to two-byte
  GET range probes. No full dump was downloaded.
- Desktop HTTP client and preparation script now identify AnagramSolver and its
  project URL. The HTTP regression test verifies the actual transmitted header.
- Targeted test and Svelte check pass. Release installed and restarted (PID 48812),
  SHA-256 `432CD4ECF293F0717827C1E464A39A46FE3574B87A4885C9D8D52CA6A2C57D66`.

## Desktop maintenance workflows, 2026-09-17

- Data download/offline preparation creates separate validated sets and manifests;
  selecting a set and saving settings activates it. Previous corpora are retained.
- Experimental builds reusable labelled datasets and saves/reopens held-out reports.
  Diagnostics distinguishes current paths from last-solve provenance/timings/cache.
- Ten desktop tests pass, including staged corpus preparation, failed-stage cleanup,
  previous-set preservation, empty-data rejection, desktop build/train round trip,
  persistent report reload, HTTP status checking and stalled-network cancellation.
  Two shared native CLI maintenance integration tests also pass.
- Strict CLI/desktop Clippy and Svelte checks pass without warnings. Normal release
  build installed and restarted (PID 34424), SHA-256
  `762BDD309963E3006D415B1524D7A756957B06AD79F22DC053CD1D724AB96C5A`.
  One previous release is retained as backup; build cache was explicitly retained.
- Tests use local fixtures and a loopback HTTP server, not production downloads.
  No full-size corpus download or manual training/quality test was performed.

## Experimental training tab, 2026-09-17

- Added explicit local training from prepared groups, held-out metrics and separate
  model activation under Settings → Experimental. No downloads or automatic runs.
- All six desktop backend tests pass, including successful model publication without
  activation and cancellation/join on shutdown. Strict desktop Clippy passes;
  Svelte check reports zero errors and warnings.
- No manual/visual training test or real-data quality assessment was performed,
  as requested. Synthetic automated coverage does not establish a quality gain.
- Normal release build installed and restarted successfully (PID 34972), SHA-256
  `876EB8183E0F08670B9B8E6C77039BFBF859FBD960171C83FA8C49E66A2F5289`.
  The previous release remains the single rotating backup; build cache retained.

## Python-feature follow-up, 2026-09-17

- Final installed executable SHA-256:
  `2CD12946B35D6EB6DD12B050A54A25D53B4DC34EDF51465CACB023E4271FEB86`.
  Single backup: `F225B3CEE9DB9EF2DCEC452932CDC3182BA268D5CDED5D7BD1B0B4107B213CFC`.
  The final rebuild corrects the extended-mode deep-limit label and validates
  worker counts before the serial fast path; installed startup is verified.

- Native/core/desktop tests pass, including serial/parallel equality, monotonic
  progress and cancellation/join, refinement accounting, model-driven order
  selection and changed-model cache misses, extended generation admission,
  recompute and cache clear without modifying settings.
- Native CLI tests build training groups, fit/evaluate a model with bag-isolated
  held-out folds, reload/rank it, ingest gzip titles into a readable canonical
  SQLite corpus and refuse to overwrite outputs. Download/preparation orchestration
  passed with local fixture inputs and the existing WordNet archive. Network
  downloads and full-size corpus rebuilds were not run in this follow-up.
- Strict Clippy, formatting, 16 generated-artifact drift checks, 17 shared
  Python/JavaScript fixtures, TypeScript and Svelte checks pass.
- Installed release `F225B3CEE9DB9EF2DCEC452932CDC3182BA268D5CDED5D7BD1B0B4107B213CFC`
  completed a native UI solve with two workers, refinement and force recompute:
  9,999 generated, 241 analyzed, 21 shown. Settings model/cache controls and new
  Advanced controls were visually inspected. Cache deletion was tested through
  the backend, not clicked through Windows automation.
- Restored one worker, refinement off, empty model, original text/budget and dark
  theme. Extended mode and force recompute are off. No learned-quality or
  parallel-speedup claim is inferred from these correctness checks.

## Original phase-4 acceptance, 2026-09-17

- Normal release build bundles Svelte UI and Rust core and installs/restarts the
  native app without a dev server. Process 47200 opened `AnagramSolver`
  from the per-user installation at the data-migration checkpoint.
- Data-migration checkpoint executable SHA-256:
  `60713A8C025383C58FD22C7FA2682CE7ABE8346391266060A2ABFACD1C4EE881`.
  Its single rollback executable was the preceding release:
  `862B3FE1DA71E64188632679FF94F18E82A5C3F205E0F52DBBA302CD33B8C30E`.
- User completed a real installed-app solve and native text export to
  `D:\TajemnikTV\Downloads\anagrams.txt`. All 21 lines have exactly the input's
  normalized letter multiset (`IAMTESTINGANAGRAMS`). Clipboard and JSON export
  were subsequently verified through native automation, as recorded below.
- User-reported candidate budget defect fixed: integer step 1 replaces step 1000
  anchored at minimum 1. The final installed release accepted a keyboard-submitted
  10,000-value request and completed it with 241 analyzed and 21 shown results.
- `cargo test -p anagram-desktop --quiet`: 4 passing tests. Actual native core
  execution, concurrent-start rejection, cancellation and shutdown join, persisted
  request reload, missing corpus errors, bounded settings read/corruption recovery,
  result expiration and exported phrase text are covered.
- `cargo clippy -p anagram-desktop --all-targets -- -D warnings`: passes.
- Svelte check: zero errors/warnings. Production frontend build passes.
- System-light primary button contrast and stale cancellation-footer text fixed.
- Runtime-data follow-up: ten required corpus files (20.52 MiB) provisioned to
  per-user `Programs/TajemnikTV/AnagramSolver/corpora`; all hashes match source files.
  Repeated provisioning preserved file timestamps. Saved paths migrated to that
  directory; custom path preservation and first-run defaults have a regression
  test. Installed native UI solve using those paths succeeded (9,999 generated,
  241 analyzed, 21 shown). No CLI/Python/Node process is used for solving.
- Settings, results.sqlite and the WebView profile were moved into the install's
  `user-data` folder. The installed app restored the existing puzzle and system
  theme; its live WebView process uses `user-data/EBWebView`. An obsolete WebView
  crash reporter recreated only diagnostics at the old profile path; it is not
  used by the new app. Updates ignore that residue, but reject conflicting saved
  settings or solver caches rather than overwriting them.

## Completed installed-window acceptance

- Copy all populated the clipboard with 21 phrases, each checked for exact input
  letter preservation. The native Save dialog wrote
  `.codex/temp/desktop-qa/export.json`; parsed JSON has succeeded status and 21 rows.
- Entered a nonexistent optional phrase DB path in Settings. The app displayed
  an actionable missing-path error; clearing/saving the path removed it.
- Cancelled an actual 250,000-candidate solve during preparation. The UI showed
  acknowledgement then restored controls and displayed `Solve cancelled` and
  terminal `cancelled`, with no partial result rows. Fixed the misleading failure
  banner discovered during QA, rebuilt and retested in the final release.
- Restart restored the puzzle input and saved light theme. After testing, restored
  and read back the user's original system theme, 9,999 budget and empty optional
  phrase path; main corpora and input were unchanged.
- Compared the generated concept with user screenshots and native captures.
  Dark/light/system appearances are readable. Keyboard submission and selection,
  focus rings, and zoom through 110%, 125%, 150% work; the toolbar reflows at the
  compact breakpoint and panes scroll independently. Reset zoom to 100%.
- User screenshots and usability feedback led to the now-fixed budget bug. The
  app is left open with real results and restored preferences. See DESIGN.md for
  the concept comparison and intentional design differences.
- Reduced-motion support is present in CSS; Windows accessibility settings were
  not changed for QA. Other OS distribution remains deferred.

The attempted WebView2 debug/Playwright attachment was rejected by execution
policy. A subsequent capability check found the separate native Windows Computer
Use tool (`@oai/sky` through node_repl): listing, selecting, activating and capturing
the installed AnagramSolver window all succeeded. The earlier conclusion that all
native automation was unavailable was incorrect. The installed interaction checks
above used this supported tool, without enabling a debug port. The native value
setter was unreliable; observed screenshot coordinates and keyboard input worked.

User screenshots show a successful 9,999-candidate solve, confirming that the
old 1-modulo-1000 input restriction is gone. The dark layout is readable at the
provided maximized size; native capture also shows a readable windowed layout.
The automated checks above separately establish clipboard, JSON and cancellation.

Build-cache removal was also rejected; the latest normal install deliberately used
`-KeepBuildCache` to avoid retrying that denied deletion. The installed current/one
backup retention is verified; project Cargo cache cleanup remains outstanding.

## Completed visual comparison, 2026-09-17

Compared the generated concept with the user's maximized screenshots and native
captures of the installed 1200px window. The graphite rail/canvas, violet action,
Segoe UI typography, restrained borders, spacing and two-pane hierarchy match the
direction. The implemented denser controls and extra word-count column are useful
intentional deviations; result counts and scores are actual engine output.
Light mode retains readable contrast; Follow Windows restores the dark palette on
this PC. Keyboard zoom through 150% increases text and activates the compact
stacked results toolbar without clipping controls horizontally. Independent pane
scrolling keeps long advanced options and results accessible. Focus rings and
keyboard select/submission were exercised. Reduced-motion behavior is provided
by the CSS media rule; Windows accessibility settings were not changed for QA.
