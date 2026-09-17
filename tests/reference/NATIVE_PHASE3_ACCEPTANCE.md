# Native phase 3 acceptance

## Verified against the current implementation

- Component/ranking parity: 200 scoring, 300 grammar, 118 retained-order,
  160 refinement, 400 phrase/cohesion, 253 learned-model and 12 complete ranking
  pipeline cases pass. Numeric tolerance is 1e-12; classifications, ordering,
  shortlist membership and ties are exact, not tolerance-relaxed.
- All 51 Rust tests pass, including storage, current-policy cache admission,
  cancellation, progress, provenance and executable cache controls.
- `ci_ordering_gate.py` passes all thresholds: R@1 0.462, R@10 0.865,
  R@50 1.000, MRR 0.598 and cross-bag margin 0.088. Native retained-order
  differential coverage includes the same 52 registry phrases.
- Default-path evidence covers all five normal-user workloads cold/warm,
  including the two 100,000-candidate workloads. The adapter supplies the frozen
  augmented dictionary/short-word policy and applies reference contraction
  formatting to native word arrays. See `tests/parity/registry.py`.
- Optional SQLite CLI evidence covers prefix/diverse and cold/warm/rebuild;
  lower-level storage tests cover batching, missing/corrupt/schema errors and
  unchanged source bytes. WAL tests bind identities and scores to one snapshot.
- Versioned bounded caches retain normalized request semantics, all budgets,
  engine/ranking versions and exact corpus identities. Atomic SQLite transactions
  cover replacement and pruning. Reopening under smaller limits prunes existing
  entries. Busy/unavailable databases fall back to computation without destructive
  replacement. Corrupt payloads are ignored. No Python pickle is read.
- Current execution time is separate from saved ranking computation time;
  cache hits never replay old job identity, policy or terminal state. Native
  generation is reused after fresh exact corpus identity checks with cache
  format 2. Current candidate/deep limits and completed-work invariants still
  apply. Entry/payload bounds exclude SQLite page overhead.

## Optional-mode boundaries

- Native `solve` supports positive bigram rescoring and an optional read-only
  SQLite phrase table. Provenance requires a concrete `ngrams` table with unique
  text keys; dynamic views and duplicate-key corpora are rejected, not silently
  reinterpreted. Keep Python for those noncanonical database modes.
- Learned models remain opt-in rather than silently changing default solve. The native
  `learned::rank_result` reports schema, selected indices and exact loaded-file
  identity; absent/programmatic models do not fabricate file provenance.
  Follow-up: native training, grouped held-out evaluation and experiment commands
  now exist; desktop Settings can select a model for order selection. Models are
  loaded before cache lookup and their exact identity is included in the key.
- Refinement has native controlled APIs and differential coverage; it is not
  implicitly enabled in default solve. Follow-up: desktop Advanced options can
  enable the bounded native refinement stage. Website integration stays deferred.
- Regex exclusions are now supported through `exclude_regex`, with bounded,
  linear-time Rust regex syntax. Look-around/backreferences report explicit
  errors rather than introducing Python-style unbounded backtracking.
- Fully required answers are now ranked, including repeated-word multiplicity;
  native solve deliberately does not retain Python's zero-residual restriction.

## Acceptance decision

- The repository's official
  title-dump builder produced `.codex/temp/phase3-corpora/wiktionary.db` with
  2,501,689 rows. All 12 native/Python ranking pipeline cases pass against this
  real database, including unchanged database bytes. The optional combined
  Wikipedia build was stopped with partial files preserved; it is not a completed
  or validated corpus. Its matrix and native differential check were not run.
  `tools/run_phase3_matrix.py` runs the unchanged reference matrix with temporary
  output/cache paths redirected into the project scratch directory.
- The user clarified that feature coverage, correct results, useful ranking and
  speed matter more than exhaustive one-to-one Python equivalence. Existing
  functional, real-corpus, CLI and quality evidence closes phase 3's ranking/cache
  scope with the explicit support boundaries above. Further corpus matrices are
  optional, not blockers. Phase 2's remaining lexical/control work is now complete
  too; this closes phases 0–3, not the future application/hosting/release phases.

## Measured improvements after the acceptance-policy update

- Prepare at most 16,384 immutable per-request WordNet feature/frame/family entries;
  unknown/unprepared words retain the ordinary path. This avoids repeated
  morphological derivation without global mutable state. The existing 6,483-word
  probe verifies prepared/unprepared equality, including non-normalized input.
- Replace repeated category splitting/scanning with direct function-word lookup,
  preserving ambiguous-word precedence. All 300 grammar cases pass.
- Cache format 2 reuses completed generation as well as ranking. Exact corpus
  identities and current deployment limits are still checked; source changes,
  corruption, rebuild and cancellation tests pass. Completed hits do not emit a
  misleading generation stage. Regex changes participate in cache identity.
- `native-cli-optimized.json` records three cold/warm pairs across all five
  normal-user workloads, including result/count checks. The final direct-lookup
  change was measured on the two large workloads in
  `native-cli-direct-classification.json`: cold medians **8.18s / 4.74s**, warm
  **0.19s / 0.21s**, compared with recorded serial Python baselines
  **52.61s / 23.28s** cold and **0.36s / 0.34s** warm. Each report records the
  measured binary hash. Small-case native cold medians before that last lookup
  change were 0.85–1.41s (Python 2.46–3.29s).
- Original regressions and intermediate measurements are retained. These are
  fresh processes with warm OS pages and controlled puzzle caches, not universal
  speed guarantees or production deployment proof. No workload budgets were cut.
