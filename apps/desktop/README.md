# TajsAnagrams desktop

Windows native window, Svelte 5 UI and direct Rust-core integration. No local HTTP
service, website account, Python or Node runtime is needed by the installed app.
Windows WebView2 Runtime is required (present on the development PC).

Use Ctrl+Plus / Ctrl+Minus to adjust text/UI zoom and Ctrl+0 to reset it.

## Build and run

From the repository root: `pwsh -File tools/build-desktop.ps1`.
Successful normal builds install into
`%LOCALAPPDATA%\Programs\TajemnikTV\TajsAnagrams`, rotate one previous executable
into `backup`, add a Start-menu shortcut and restart that installed app gracefully.
`-TestOnly` and CI build without installation/restart. Build scratch uses
`.codex/temp/desktop-build`; normal successful installs remove it after startup
verification. Test-only and failed builds retain scratch for inspection.
`-KeepBuildCache` explicitly retains the build cache without changing installation
or the one-backup retention policy.

On a clean checkout, ignored corpus downloads are not bundled. The app still
installs and starts: use Settings → Data to download/prepare a set, select it,
then Save settings before solving. No large downloads start automatically.
An immediate startup failure restores the verified previous executable when
one exists; settings and corpora are not replaced during rollback.

Development: `pnpm --dir apps/desktop tauri dev`. A browser-only Vite page cannot
solve: desktop IPC deliberately has no mocked or browser-local fallback.

## Data and execution

Advanced options now include Quick/Balanced/Exhaustive presets, automatic (0) or 1–32
ranking workers (default 1), bounded order refinement and force recompute.
Extended mode removes deployment time/candidate/deep caps; generation budget 0
enumerates every matching bag. This can consume substantial memory/CPU and still
only deep-ranks the requested shortlist. Cancellation remains cooperative. Extended
mode is session-only: after restart, explicitly re-enable it for saved unbounded
requests. The engine still supports at most ten ordered words. Extended mode lifts
the UI's numeric deployment maxima, including result counts; results are paginated
in 100-row pages, while Copy/Export include the full result.

**Generate word bags only** skips ranking and needs only a dictionary (plus
unigrams if frequency filtering is enabled). Its output is unranked and bypasses
the ranked-result cache. Text/JSON export and cancellation remain available.

General settings expose the cache path, entry/byte limits and custom execution
limits. Blank custom limits are unlimited; Extended mode bypasses deployment
limits. Existing settings migrate with safe defaults. The TajsAnagrams installation
and saved-report reader relocate legacy app-owned corpus/model/cache paths only
when the corresponding destination exists; external custom paths are preserved.

Settings includes a clear-result-cache action and an Experimental tab for optional
learned-model selection and local training from prepared `ranker-build` JSON.
Training is explicit, cancellable and bounded to one worker, 60 seconds, 4 MiB,
1,000 groups. Models are saved beneath
`user-data/models`; selecting one and saving settings is a separate action.
The model selects retained word orders in the corpus shortlist; the displayed
score remains the standard grammar/corpus score, not a probability or learned
model score. Neither refinement nor a trained model is enabled automatically.
See [native tools](../../NATIVE_TOOLS.md) for Python-free corpus preparation and
model training. Newly trained models require their own quality evaluation.
Learning rate and L2 regularization are editable; saved reports record the chosen
options. Dataset building exposes retained-order count and training phrase weight.
Experimental also supports standalone ranking of supplied feature-record JSON.

### Desktop maintenance

- **Settings → Data:** validate draft corpus/model paths with text sanity counts,
  native loaders and a one-candidate smoke solve. Download/prepare a new corpus set
  from the original dictionary, Norvig and WordNet providers, optionally including
  Wikimedia titles. Or choose an offline source folder. Update by preparing a new
  set, then selecting it and saving settings; active files are never overwritten.
  Reopen a saved set through its `manifest.json`. Prepared sets and source files
  live in `user-data/corpus-sets`; failed stages are removed, older sets retained.
  Independent dictionary-only and phrase-only tools accept arbitrary local source
  files and publish new outputs under `user-data/prepared`, without full downloads.
- **Settings → Experimental:** build datasets from labelled case JSON, review
  skipped cases, train and inspect held-out metrics. Datasets live in
  `user-data/training`; models and persistent `.report.json` reports in
  `user-data/models`. Saved quality reports can be reopened without retraining.
- **Settings → Diagnostics:** inspect configured paths, last-solve SHA-256
  identities, timings, effective budgets and cache flags/storage. Last-solve data
  is explicitly distinguished from validation of the current settings.

Maintenance shares the joined/cancellable worker owner with solving. Phrase builds
use up to eight parsing workers (leaving one logical CPU free where possible),
8,192-title / 4-MiB input batches, aggregated sorted writes and a 64-MiB SQLite
page cache. The `WITHOUT ROWID` database stores the text key only once; journaling
and durability defaults remain enabled. All parsing tasks finish before errors or
cancellation return. Finished manifests report workers and database update counts.
Dataset building
and data preparation have a one-hour deadline; source files are capped at 2 GiB,
with 30-second network read timeouts. Downloads stop promptly on cancellation.
Training retains its smaller 60-second/4-MiB/1,000-group limits. These workflows
link the CLI's native maintenance library; no installed CLI, Python or shell is
required. All actions are explicit and new data/models require separate activation.

Settings offers explicit dictionary, unigram, bigram, WordNet-folder and optional
phrase SQLite paths. Normal local installation provisions the default dictionary,
unigram/bigram counts and required WordNet files into
`%LOCALAPPDATA%\Programs\TajemnikTV\TajsAnagrams\corpora`. These are shared across current
and backup executable releases, not duplicated on each update. Initial copies are
SHA-256 verified; subsequent builds preserve existing corpus data. No CLI executable,
Python runtime, downloaded archives, development tools or old run caches are needed.
The repository copy is retained for development, but the installed app no longer
depends on it. Existing default repository paths migrate to the provisioned files;
custom paths and optional phrase databases remain user-selected. Missing paths produce
actionable errors. Settings, the bounded native cache and WebView profile live in
`user-data` beside the executable. Installation migrates the old app-local profile
without overwriting existing files. This per-user install is writable; updates
rotate only the executable, preserving both data folders. Deleting the installation
folder also deletes its saved state. The system WebView2 runtime remains external.

One active worker, no waiting queue; another start is rejected. The host cancels
and joins its worker on exit. The same engine contracts retain generated,
deep-analyzed and shown counts, exhaustion/cap/deadline outcomes, exact corpus
identities and cache provenance. Completed results survive until the next accepted
job, not across restart. Last submitted options and theme/corpus choices persist.

Standard deployment limits: 40 normalized letters, 250,000 generated candidates, 10,000 deep analyses,
512 beam width, 128 retained orders, 100 displayed rows per word-count group and
120 seconds. Custom settings or Extended mode can change these deployment limits.
UI/backend reject unsupported requests rather than silently clamp.
Regular expression exclusions use bounded Rust syntax. Learned-model/refinement
utilities remain offline; they are not silently added to the default pipeline.

## Verification

`pnpm --dir apps/desktop check`, `cargo test -p tajs-anagrams`, and
`cargo clippy -p tajs-anagrams --all-targets -- -D warnings`.
Installed-window tests must separately exercise real solving, cancel, missing
corpora, saved preferences, copy/export, keyboard/theme/scaling and clean shutdown.
