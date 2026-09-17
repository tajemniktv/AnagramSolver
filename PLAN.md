# TajsAnagrams: Rust engine and desktop-first app plan

Status: phases 0–4 implemented, 2026-09-17. Production deployment still
requires explicit approval.

Current implementation: native engine plus an installed Windows desktop app.
Website integration and hosting remain deferred until explicitly resumed.
The Python-feature follow-up below extends the original desktop acceptance.
The remaining desktop/CLI feature-control gaps are implemented: generation-only,
standalone feature ranking, independent corpus tools, configurable training pools
and optimizer settings, automatic ranking workers, configurable cache/execution
limits, and paginated uncapped Extended-mode results. TajsAnagrams rebranding
preserves migrated data and one rotating executable backup. Historical Python
pickle/CLI syntax and unbounded regex features remain intentional exclusions.
Execution priority updated by the user: **finish phases 0, 1 and 3 fully first**.
Phases 0, 1 and 3 reached their acceptance checkpoints first; phase 2 is now
complete too. Phase 4's installed-app evidence is recorded in its own section;
broader distribution and production cutover remain future work.

Acceptance policy updated by the user: Python is a reference, not a requirement
for perpetual one-to-one equivalence. Preserve user-facing features and hard
correctness invariants; assess ranking with useful-result quality checks and
performance with representative workloads. Different scores, tie order or search
implementations are acceptable when justified by equal or better results.
Existing differential checks remain useful regression evidence, but expanding
them or running every optional corpus matrix is not a completion prerequisite.
Removing an exact-parity gate does not establish a speedup. The final native
measurements below establish improvements on the representative workloads;
they are not a guarantee for every possible input or machine.

### Current evidence and remaining gates

- Phase 0 reference-freeze gate passed: the pinned manifest is durable in
  `tests/reference/manifest.json`.
  `tools/reference_fixtures.py --check` verifies source/data identity and frozen
  normalization, generation, validation, synthetic phrase and model outputs.
  Corpus formats and distribution decisions are in `tests/reference/CORPORA.md`.
- The full default Python performance attempt exceeded its 180-second timeout
  on the second cold sample. `tests/reference/stress-attempt.json` preserves this
  failure explicitly; `user-performance.json` contains only completed runs.
  It is not a passing full-suite report. The 24-run bounded CLI baseline,
  12 separate stage measurements, 258 Python tests and 52-case ordering gate
  are captured in `tests/reference/`. All 52 ordering and five default CLI
  registry cases now have replay-verified golden outputs; see `ACCEPTANCE.md`.
  No native performance claim is made.
- Phase 1: structural schemas and TypeScript exist, as do generation/ranked CLI
  operations and job-state definitions. Adapter-owned deployment admission,
  hard deep limits and non-extending deadlines are enforced. Ranked execution
  emits actual stages, confirmed counts and terminal outcomes through the
  `JobStatus` contract. Phase-1 gate passed: exact consumed-input identities,
  typed error envelopes and semantic status validation are implemented. Forty-one
  Rust tests, 17 shared contract fixtures, generated-contract drift checks,
  TypeScript checks and strict Clippy pass. Native migration/performance and
  complete in-flight accounting remain phase-2/3 gates, not phase-1 claims.

Priority checkpoint: **phases 0 and 1 are complete against their stated gates**.
The frozen oracle check, all 16 generated artifacts, 17 Python/JavaScript contract
fixtures, TypeScript checks, workspace Rust tests and strict Clippy were verified
again on 2026-09-17. The recorded Python stress timeout remains a baseline finding,
not a passing performance result. Phases 2 and 3 are now complete for the native
generation/ranking/cache scope under the updated acceptance policy. See
`tests/reference/NATIVE_PHASE3_ACCEPTANCE.md` for the explicit support boundaries.

Final phase-2 checkpoint: bounded regex exclusions, required-word handling,
cooperative bulk operations and serial performance checks are complete. Native
solve also ranks fully required answers instead of retaining Python's restriction.
All 51 Rust tests, 16 generated artifacts, 17 cross-language fixtures, TypeScript,
strict Clippy, pinned-oracle verification and targeted grammar/WordNet checks pass.
The latest large-workload medians are 8.18s / 4.74s cold and 0.19s / 0.21s warm,
versus recorded serial Python baselines of 52.61s / 23.28s cold and 0.36s / 0.34s
warm. Earlier regressions remain recorded, not erased. See
`native-cli-direct-classification.json`, `native-cli-optimized.json` and the
original `native-cli-performance.json` under `tests/reference/`.

### Historical implementation evidence (chronological)

The notes below describe individual milestones, not the current remaining-work
list; later entries supersede statements that a component is not yet ported.

- Python reference commit: `c12e3d5519d653fd65c7ffbbb2a1597941fedd2a`.
  All 258 tests (including CLI integration) and the 52-case ordering gate passed
  locally on 2026-09-17. Remote CI has not been run for this commit.
- Rust workspace, normalization/letter inventories and lazy exact enumeration are
  implemented. Ordered prefix/diverse results match Python in 500 seeded cases,
  including repeat/clue modes and truncation. Cancellation/deadline outcomes are
  explicit. This is not yet the native application: corpus admission, ranking,
  contracts, UI and hosting remain unfinished.
- Native unigram reading and lexical admission now cover normalized duplicate
  counts, frequency/length/short-word rules, forced words, exclusions and stable
  candidate sorting. Ten Rust tests and strict Clippy pass, including malformed
  UTF-8 handling. Eight real-corpus cases match Python vocabulary counts and
  ordered prefix/diverse bags (`python tests/parity/corpus.py`).
- `anagram-cli generate DICTIONARY [UNIGRAMS]` accepts the version-1 JSON contract
  documented in `contracts/README.md`, with explicit validation errors and
  `generation_only` output. Required multiplicity, conflicts, cap detection and
  version rejection are exercised through the actual CLI process. Generated
  TypeScript/schema and job/ranking contracts are not implemented yet.
- `tools/capture_reference.py` captured SHA-256 identities for 21 source files
  and 73 locally available corpus files in `.codex/temp/reference-manifest.json`.
  Optional phrase/model identities and redistribution review remain outstanding.
- Next: generate schema/TypeScript contracts, complete corpus provenance/licensing
  and profile native generation before porting ranking. Neither small fixtures
  nor the eight corpus cases constitute the complete parity/performance gate.
- Initial ranking primitives are now native: lexical penalties, bigram corpus
  reading/edge evidence, pair potential, exact bigram order DP, morphology-family
  inheritance, hint informativeness and per-word-count pre-score percentiles.
  `python tests/parity/scoring.py` passes 200 seeded component/record comparisons
  against Python with exact order/tie checks and 1e-12 numeric tolerance. Grammar,
  retained-order/top-K and final ranking remain unported; the CLI still correctly
  advertises generation-only output. No native speedup claim has been made.
- Native immutable WordNet loading now includes POS indices, exception-based and
  regular morphology, per-lemma verb frames, features and valency categories.
  `python tests/parity/wordnet.py` matches Python on 6,483 words, including all
  noun/verb exceptions, function words and seeded inflections. Eleven Rust tests
  and strict Clippy pass. Phrase-structure grammar and final ranking are next;
  lexical-data parity alone does not prove grammar parity.
- Local grammatical classification/adjacency/start/end/potential/coverage and
  compact noun-phrase spans are now native. `python tests/parity/grammar.py`
  passes 262 cases, including all function-word pairs and the audited ambiguous
  noun/adjective heads, participles and postnominal PPs. Strict Clippy passes.
  Subject agreement, compact comparative/subordinate parsing, valency tails and
  base phrase-structure selection are also now native, with 270 differential
  cases covering every exposed helper plus final base structure fields.
  Extended auxiliary-chain/clause-validity/comparative rules, retained-order
  search and final score assembly still need implementation and parity proof.
- Native auxiliary-chain parsing/agreement now covers modal, perfect,
  progressive, passive and nested/negated forms. Graded comparative morphology
  and spans, finite-clause validity demotion, and article/auxiliary surface
  penalties are ported. The expanded grammar parity gate passes 293 cases with
  exact classification/consumption checks and component-score comparisons;
  strict Clippy passes. Combining these into the extended structure scorer and
  retained-order search is still unfinished.
- Extended auxiliary/comparative/parallel structure selection and combined local
  grammar are implemented; the grammar differential gate now passes 300 cases.
  The initial exact/k-best retained-order pool passes 24 exact/beam parity cases,
  including repeated words, unknown-word ties and six-word constructions. It
  preserves per-state insertion order and lexical final tie breaks. Diversity,
  k-opt refinement, final row scoring and corpus-based reranking remain separate
  unfinished steps; this does not complete phase 3.

Implementation update: native structural diversity passes 66 exact/beam order-pool
comparisons, including widened 64/72-order pools. Standalone bounded k-opt refinement,
refined endpoint pools and original-seed-preserving augmentation pass 160 differential
cases with exact order, evaluation-count and round-count checks. Strict Clippy passes.
Refinement remains opt-in, matching the Python facade's current behavior; it is not
silently enabled in default ranking. Final score assembly, optional corpus/model
rescoring, caches and the remaining phase 0–3 gates are still incomplete.

- Native prepared rows, WordNet morphology-family expansion, serial deep analysis
  and final base-score buckets now match the active Python facade in 12 complete
  pipeline cases (auto/exact/beam, empty/partial/all shortlists). Every row field,
  selected index, retained alternative and final ordering is compared. The native
  default retains 56 orders, matching the facade rather than the older layer.
- Phrase hierarchy and non-overlapping corpus cohesion scoring pass 400 seeded
  differential cases, including absent/negative counts, repeated/empty tokens and
  overlapping spans. Strict Clippy passes. SQLite storage, corpus admission and
  alternative-order rescoring are not yet connected; these component checks do
  not prove the optional phrase path or full ranked CLI parity.

## Outcome

Deliver a Windows desktop app the user can launch and try on this PC first,
powered by the existing Rust solver. Keep expensive searches local. A future
cross-platform release and page on tajemniktv.com can reuse the engine and UI,
but website integration, hosting and website-specific acceptance are deferred.

Python remains the working implementation and reference oracle until the Rust
replacement passes parity and performance gates. Do not promise a speedup before
measuring: Rust cannot remove the combinatorial cost of anagram enumeration.

## Repository findings and decisions

The website's default branch was inspected through GitHub on 2026-09-08:

- [package.json](https://github.com/tajemniktv/tajemniktv.com/blob/main/package.json):
  Svelte 5, SvelteKit 2, TypeScript, Vite, Tailwind 4, adapter-node, Node 24,
  pnpm 11; Vitest and Playwright already exist.
- [layout.css](https://github.com/tajemniktv/tajemniktv.com/blob/main/src/routes/layout.css):
  charcoal/violet styling, self-hosted Inter and JetBrains Mono, semantic color,
  spacing, radius and motion variables, and reduced-motion handling.
- [+layout.svelte](https://github.com/tajemniktv/tajemniktv.com/blob/main/src/routes/%2Blayout.svelte):
  existing main-site and project-site headers/footers and a shared main landmark.
- [hooks.server.ts](https://github.com/tajemniktv/tajemniktv.com/blob/main/src/hooks.server.ts):
  strict host gates, admin isolation, same-origin connect policy, and project
  subdomains limited to root HTML, metadata and runtime assets.
- [README.md](https://github.com/tajemniktv/tajemniktv.com/blob/main/README.md):
  documented production topology is Cloudflare Tunnel -> loopback Caddy -> private
  Docker services. PostgreSQL owns site content; the admin service is separate.
  This is repository evidence, not a live infrastructure verification.

Decisions:

1. Use **Svelte 5 + TypeScript** for the reusable UI. No React migration or second
   frontend framework in the website.
2. Deferred website option: a proposed main-site route **`/tools/anagram-solver`**. Keep the
   website's existing layout and SEO conventions. A subdomain is a later option:
   today's project-host gates do not accept arbitrary tool/API paths.
3. Use the Rust library behind a thin desktop adapter with typed commands/events
   and a bounded worker executor. Do not build an HTTP service just to enable the
   desktop app. Axum/Tokio remains a possible later hosted adapter, not phase 4.
4. First app: **Tauri desktop window + Svelte/TypeScript UI on Windows**, using
   the existing core directly. It must launch without Python, Node, a dev server
   or a separately started daemon. Detect the Windows WebView2 prerequisite and
   report any required setup. Tauri is now the desktop delivery path, not an
   optional wrapper after a browser-first release.
5. Keep website content/admin/database authority unchanged. The solver is a
   separate service, not compute inside SvelteKit request handlers or PostgreSQL.

## Ownership and proposed layout

Keep engine, contracts, shared UI and local packaging in this repository:

```text
Cargo.toml                    # Rust workspace, introduced in phase 1
crates/anagram-core/           # normalization, generation, ranking; no HTTP/UI
crates/anagram-cli/            # commands and machine-readable output
apps/desktop/                 # Tauri host + Svelte UI; bounded local job owner
packages/solver-ui/            # reusable controls only when extraction is useful
crates/anagram-server/         # deferred hosted jobs/HTTP adapter
contracts/                    # versioned wire schema and contract fixtures
tests/parity/                 # Python/Rust differential fixtures and harness
```

These are proposed paths, not existing components. Avoid splitting the Rust core
into many crates before ownership or compile-time evidence requires it.

When website work resumes, the website repository owns its route, SEO, host-theme adapter and a thin
same-origin API proxy. Consume a versioned solver-ui package/artifact pinned to a
release; use a local package link during development. Do not copy component
sources into both repositories or introduce a monorepo migration merely to share
them. Prove package consumption in both builds before stabilizing publication;
neither shared-package publication nor a website build blocks the desktop app.

The core owns immutable shared lexical data and explicit per-request mutable
state. Replace Python's scoped hooks/global locking with explicit dependencies;
do not transliterate monkeypatch layers into Rust global state.

## Phase 0 — Freeze behavior and establish measurements

Checklist convention: checked items are implemented and have the evidence noted
below or in Implementation evidence. An unchecked parent can contain completed
sub-items; its remaining scope is still required. Phase 0's reference-freeze gate
is passed; native migration/performance gates remain open.

- [x] Review and land the existing Python audit/improvement work separately from
      port commits; preserve unrelated changes. Record the exact reference commit.
- [x] Capture corpus versions/hashes, ranking configuration, cache state, machine
      details, candidate counts and reference outputs in reproducible fixtures.
  - [x] Capture local dictionary/frequency/WordNet hashes with
        `tools/capture_reference.py` (29 source and 73 corpus files).
  - [x] Complete durable pinned fixtures, optional input identities and machine/
        cache/budget metadata; the temporary manifest alone is insufficient.
    - [x] Store `tests/reference/manifest.json` pinned to the Python reference,
          covering 29 oracle/harness files and 73 corpus files. Capture now rejects
          oracle source drift instead of labeling the current HEAD as the oracle.
    - [x] Capture deterministic behavior fixtures and corpus licensing inventory.
    - [x] Capture measured cold/warm reports, separate stage timings and a truthful
          acceptance record, including the failed stress attempt.
- [x] Inventory dictionary, frequency, WordNet, phrase SQLite and optional feature
      ranker inputs; document formats, licenses and redistribution requirements.
      See `tests/reference/CORPORA.md`; unresolved data permissions prohibit
      bundling, not local comparison against the pinned, provisioned inputs.
- [x] Extend the reference harness to cover letter normalization, punctuation and
      Unicode behavior, repeated required words, alternative hints, exclusions,
      impossible inputs, empty results, exact cap exhaustion and deterministic ties.
- [x] Include prefix/diverse enumeration, exhaustive-generation semantics, grammar,
      ordering, optional phrase evidence and optional learned ranking in parity scope.
  - [x] Add seeded prefix/diverse generation, lexical/WordNet/grammar/order,
        refinement and phrase/cohesion differential harnesses in `tests/parity/`.
  - [x] Add learned-feature/model/ordering parity (`tests/parity/learned.py`,
        253 cases, including invalid and missing model inputs).
  - [x] Complete default-path/end-to-end oracle coverage.
    - [x] Verify native `solve` CLI against the real-corpus Python generator-export
          and reranker pipeline on eight prefix/diverse cases, including required
          words and hints (`tests/parity/solve.py`). All row components and counts
          match, including historical export quantization.
    - [x] Expand to the full quality suites, optional phrase/model fixtures and
          performance/cancellation/cache acceptance cases.
          `registry.json` freezes all 52 ordering and five normal-user CLI cases;
          both capture and independent replay pass. `behavior.json`, `gates.json`
          and measured reports cover the remaining reference inputs/outcomes.
          This freezes Python behavior; full native quality/cache parity is still
          required by phases 2/3, not implied by these reference fixtures.
- [x] Benchmark generation and ranking separately, then actual end-to-end cold and
      warm CLI runs using `benchmark_user_runs.py` and the existing quality gates.
      See `tests/reference/ACCEPTANCE.md`: bounded runs and quality gates pass;
      the default stress attempt times out and is not a full performance-suite pass.

Gate: a pinned, repeatable oracle and data manifest, not snapshots from a moving
worktree. Preserve the current default prefix strategy: earlier local 20k-cap
measurements showed diverse could cost substantially more without recall benefit
on the sampled cases. Re-evaluate on a broader set rather than generalizing that
small sample. Distinguish empty puzzle caches from cold corpus/OS caches.

## Phase 1 — Contracts and a minimal Rust vertical slice

- [x] Define typed request/result/error contracts with a schema version and
      generated or mechanically checked TypeScript types.
  - [x] Implement version-1 Rust generation request/result/errors and JSON CLI.
  - [x] Complete ranked/job contracts, JSON schema and TypeScript parity checks.
    - [x] Derive structural JSON Schema and TypeScript from Rust; add drift checks,
          shared Rust/Python/JavaScript fixtures and TypeScript negative type checks.
    - [x] Add a development ranked JSON CLI with separate generated/deep/shown
          counts and per-word-count results. Document it in `contracts/README.md`.
    - [x] Stabilize unified defaults, budgets, progress/provenance and generated
          cross-language schemas before treating it as the final public contract.
- [x] Separate semantic search options from deployment limits. Include input,
      hints/required/excluded words, word-count constraints, lexical options,
      generation strategy, candidate budget, deep-ranking budget and result limit.
  - [x] Add generated `DeploymentLimits`, independent of semantic requests, and
        CLI `--limits` for both operations. Reject over-budget requests without
        clamping; enforce actual family-expanded deep count before ordering;
        keep shared cancellation and the earlier of policy/caller deadlines.
        Fifteen cross-language fixtures, 21 Rust tests, strict Clippy and eight
        ranked CLI parity cases pass (including hard-deep rejection).
- [x] Specify validation precedence, defaults, normalization and stable error codes.
      Reject non-finite values and contradictory constraints; never relax silently.
  - [x] Validate generation semantics before corpus I/O for both native operations;
        reject request budgets above JavaScript's exact-integer range. Ranked
        validation uses the same generation validator before ranking constraints.
- [x] Define job states: queued, running, succeeded, cancelled, timed_out, failed.
      Include stage, counts, effective budgets, engine/data versions and cache flags.
  - [x] Define transport-neutral job/progress data and terminal-state transition
        rules, including unknown exhaustion and semantic versus hard deep budgets.
  - [x] Populate progress/provenance from the running engine and finish semantic
        validation; definitions alone do not prove runtime behavior.
    - [x] Populate ranked queued/running/terminal events, effective limits,
          generation exhaustion, completed deep-work counts and uncached flags.
          Successful results carry the terminal status; `--progress` emits JSONL
          on stderr without contaminating result stdout. Runtime cancellation,
          policy rejection, finalization and real-corpus parity checks pass.
    - [x] Populate exact corpus identities and finish status/error semantic
          validation. Completed order evaluations and corpus rows are preserved
          on interruption; unfinished individual scores/rows are not completed work.
- [x] Preserve the distinction between generated bags, deep-analyzed bags and shown
      rows. Distinguish exhausted search, generation cap and deadline termination;
      report unknown exhaustion when a deadline prevents the extra-candidate probe.
- [x] Implement corpus loading, normalization and a small deterministic generation
      path through the Rust CLI. Keep ranking in Python until its own phase.

Gate: schema fixtures pass in Python/Rust/TypeScript; a minimal Rust request works
end-to-end against the oracle. A generation-only result is not advertised as a
replacement for the ranked solver.

Phase-1 acceptance: all items above pass. `contracts/README.md` specifies v1
defaults, validation precedence, stable errors and identity encodings. Sixteen
generated schema/TypeScript artifacts and 17 shared fixtures cover the wire
boundary, including actual CLI result/error/progress output. Runtime tests cover
terminal-state semantics, cancellation during ranking/finalization, deadlines
between reaching a cap and its extra-candidate probe, immutable file snapshots
and concurrent SQLite WAL writes. Eight real-corpus ranked parity cases verify
the ten standard input identities as well as results. Optional SQLite identities
are independently checked against their canonical schema-and-row representation.
This is a contract/minimal-slice gate, not full native solver acceptance.

## Phase 2 — Port exact generation

Completion evidence: the checked items below cover generation semantics,
lexical policies, controls and serial measurements. Milestone notes retain their
historical limitations; later completed items supersede earlier open-work notes.

- [x] Port letter inventories, vocabulary filtering, required/hint handling,
      pruning and word-count traversal. Preserve repeated-word multiplicity.
  - [x] Port normalized inventories, basic lexical admission, required/hint
        multiplicity and word-count traversal; validate small and real-corpus cases.
  - [x] Complete frontend lexical policies and bounded exact-search pruning.
    - [x] Add `exclude_regex` to generation/solve contracts and lexical admission.
          Case-insensitive search filters dictionary words and hints, rejects
          required-word conflicts and returns `invalid_regex` for invalid or
          unsupported syntax. Bound patterns/source/compiled memory; use linear
          matching rather than unbounded backtracking. Rust regex syntax does
          not support Python look-around/backreferences; this deliberate boundary
          is documented in `contracts/README.md`. CLI behavior, cache identity,
          generated schemas and Python/JavaScript contract checks pass.
    - [x] Expose default-compatible short-word policy and normalized extra-short
          whitelist options in generated native contracts. Twenty-four real-corpus
          policy/traversal comparisons and cross-language contract checks pass.
          Regex frontend options remain outstanding.
    - [x] Expose normalized forbidden-letter filtering, reject required-word
          conflicts and unusable hints before corpus loading. Six actual CLI
          contract tests, shared schema/TypeScript checks and strict Clippy pass.
    - [x] Port reference clue-feasibility pruning: discard branches when every
          unmatched clue is behind the candidate cursor or cannot fit residual
          letters. Preserve candidate order and check interruption while scanning.
          Three generation tests, 500 seeded ordered parity cases and Clippy pass.
          See bounded dead-state memoization below.
    - [x] Add final-word signature lookup with admission-ordered index lists and
          monotone cursor lookup. Index construction honors cancellation/deadlines.
          Three generation tests, 500 seeded ordered parity cases and Clippy pass.
          Index memory/runtime costs remain part of the serial benchmark gate.
      - [x] Share one lazily built immutable signature index across all serial
            diverse strata instead of retaining a duplicate per stream. Keep
            clue-specific dead states separate. Three generation tests, 500
            seeded ordered comparisons and strict Clippy pass; earlier benchmark
            reports identify their pre-sharing binaries and are not new timings.
      - [x] Honor cancellation/deadlines during duplicate validation and between
            diverse-stream constructions. A pre-expired request with no possible
            strata reports timeout, not exhaustion. Four generation tests, 500
            seeded ordered cases and strict Clippy pass.
      - [x] Check interruption inside each stream's vocabulary setup and while
            creating diverse clue groups. Combine clue-index and min/max-length
            discovery into one checked scan rather than three unchecked scans.
            Deterministic mid-scan cancellation/timeout tests, all 500 seeded
            ordered generation cases and strict Clippy pass.
    - [x] Memoize fully explored, result-free generation states using residual
          inventory, starting candidate, remaining word count and matched clues.
          Limit each stream to 4096 entries; reaching this limit only disables
          additional memoization, never truncates search. Interrupted/live branches
          are not recorded as dead. Three generation tests, 500 seeded ordered
          parity cases and strict Clippy pass; performance is not yet measured.
    - [x] Port remaining per-candidate reference pruning: feasible word-length
          bounds, sparse letter-fit/subtraction and immediate exactly-one clue
          rejection before allocating a child frame. Three generation tests,
          500 seeded ordered parity cases and strict Clippy pass. Frontend lexical
          policy coverage and equal-workload time/RSS evidence remain open.
- [x] Implement bounded prefix and opt-in diverse traversal, including deduplication,
      redistribution of budget and the extra unique-bag truncation probe.
- [x] Add cooperative cancellation/deadline checks in expensive loops and corpus
      operations. Limits must bound work, not merely the displayed result count.
  - [x] Check cancellation/deadlines inside generation DFS and expose stop reasons.
  - [x] Propagate cancellation through corpus loading and all ranking loops;
        validate interrupted work and deadline probe semantics end-to-end.
    - [x] Add explicit per-run control to corpus readers, WordNet parsing,
          pre-scoring/preparation, exact/k-best ordering and corpus rescoring;
          interrupt SQLite's VM and expose CLI `--timeout-ms`. Targeted tests and
          unchanged ordering/WordNet/scoring parity pass.
    - [x] Finish cancellation coverage for bulk sorts/refinement and structured
          partial-progress outcomes. All unbounded ranking sorts use cooperative
          sorting; remaining ordinary sorts handle at most ten words per ranked
          bag or the fixed corpus-role list. Check WordNet ASCII conversion in
          chunks, refinement collection per row and final generation bag copying.
          Control/progress tests cover interruption inside enumeration/ordering,
          SQLite, corpus loading, rescoring and finalization. OS I/O and external
          offline scorer callbacks remain cooperative, not forcibly preemptible.
      - [x] Add controlled single/refined/augmented refinement APIs; check within
            permutation recursion and between scorer calls. Existing convenience
            APIs preserve reference behavior. Cancellation tests and 160 bounded
            refinement parity cases pass. Individual scorer calls must bound
            their own work; other ranking bulk-sort coverage remains open.
      - [x] Add stable cooperative index-merge sorting and use it for refinement
            pools. It preserves tie order and payload ownership on interruption,
            using two linear index buffers rather than cloning row payloads.
            Stability/cancellation tests and all 160 refinement parity cases pass.
      - [x] Use cooperative stable sorting in ranked row preparation, deep-family
            shortlist selection and final result buckets. The solve path calls
            controlled APIs; compatibility wrappers retain unlimited execution.
            All 32 Rust tests, 12 ranking pipeline cases, eight real-corpus ranked
            CLI cases and strict Clippy pass. Other remaining bulk sorts still
            need conversion; this does not close the parent gate.
      - [x] Convert all five corpus-selection/rescoring sorts to cooperative
            stable sorting, and check candidate-pool construction and phrase-owner
            expansion loops. Twelve complete ranking/corpus pipeline parity cases,
            five runtime progress tests and strict Clippy pass. Partial in-flight
            rescore counters and other modules' bulk sorts remain open.
      - [x] Convert k-best state/complete-path and retained-candidate sorts, plus
            pre-ranking percentile sorts and tie scans, to cooperative control.
            Sixty-six exact/beam ordering cases, 200 scoring parity cases, five
            control tests and strict Clippy pass with unchanged tie behavior.
      - [x] Add controlled vocabulary admission and route both native CLI
            operations through it, including dictionary/forced-word loops and
            the final frequency/length/lexical sort. All 32 Rust tests, eight
            real-corpus admission/prefix cases and strict Clippy pass. Legacy
            flag/deadline generation callers retain their existing DFS control;
            adapters use the new shared-Control entry point for admission too.
      - [x] Close the legacy generation API's admission gap: its borrowed
            cancellation flag/deadline now governs vocabulary admission and the
            stable sort, not just DFS. Reuse the same checked sorting/admission
            implementation without background polling or unsafe flag ownership.
            Cancel-on-read and pre-expired tests exercise both APIs; all eight
            control tests, 24 real-corpus short-policy cases and Clippy pass.
      - [x] Preserve cancellation/timeout error codes at the generation admission
            boundary instead of reporting interrupted reads/sorts as corpus errors.
            A deterministic cancel-on-read test and pre-expired deadline test pass;
            all six control tests and strict Clippy pass.
      - [x] Emit corpus-rescoring progress after every completed row and preserve
            it on cancellation, including cancellation from the observer itself.
            Six runtime progress tests and strict Clippy pass. Work inside an
            unfinished order/row remains outside confirmed-completion counters.
      - [x] Preserve completed order evaluations inside interrupted bags, batching
            live events every 256 evaluations and flushing exact counts on errors.
            Bag completion remains distinct. Seven control tests, six progress
            tests, 66 order parity cases and strict Clippy pass; unfinished
            individual scores/rows are not counted as completed work.
      - [x] Verify terminal partial-order accounting through the actual solve
            pipeline: cancel at evaluation 256 of the first six-word bag; terminal
            status retains 256 evaluations, zero completed bags and zero shown rows.
      - [x] Propagate control through diversity fingerprint construction, pairwise
            similarity scans, winner selection and output copying. Publish a deep
            row only after selection succeeds; interruption preserves completed
            evaluation counts. All 118 ordering cases, seven progress tests and
            strict Clippy pass without changing retained-order tie behavior.
      - [x] Check cancellation between final output-row copies and before marking
            any rows shown. Seven runtime progress tests and strict Clippy pass;
            partial output assembly is discarded on interruption.
      - [x] Add cancellation-aware exact bigram path DP for explicit offline
            callers, retaining its 16-word ceiling. A deadline test at that
            ceiling, all eight control tests, 200 scoring parity cases and
            Clippy pass. This utility is not silently added to default solve.
      - [x] Check control during default pre-ranking pair-potential construction
            and stable edge sorting. Two hundred scoring parity cases, seven
            runtime progress tests and strict Clippy pass.
      - [x] Propagate control through lexical frequency collection, stable score
            sorting and duplicate/junk accounting in default pre-ranking. All
            200 scoring cases, seven progress tests and strict Clippy pass.
      - [x] Check cancellation during export-record grouping, stable rank sorting
            and decimal-quantized input conversion. Default solve uses the
            controlled conversion; 12 ranking parity cases, seven progress tests
            and strict Clippy pass with the frozen quantization preserved.
- [x] Differential-test full candidate sets for small exhaustive cases and ordered
      capped prefixes for bounded cases. Add seeded randomized small alphabets.
- [x] Benchmark serial first; introduce bounded parallelism only where it helps and
      does not change deterministic output or consume unbounded memory.
  - [x] Add/run `tools/benchmark_native_generation.py`: four identical synthetic
        cases, three serial fresh-process samples per engine, exact ordered-output
        equality, release Rust binary identity and 2ms sampled root RSS. Results
        are in `tests/reference/native-generation-synthetic.json`: Python
        0.282–0.542s / 35.2–35.3MB; Rust 0.101–0.287s / 15.3–15.8MB.
        Includes startup/import/JSON costs, not isolated core timings; OS caches
        are not flushed. Real-corpus/equal-workload ranking benchmarks remain open.
  - [x] Measure generation from identical real-corpus-admitted vocabularies:
        `tools/benchmark_native_generation.py --real-corpus`, eight prefix/diverse
        cases at cap 200, three samples each, exact output equality. Python
        0.434–0.462s / 69.1–69.2MB; Rust 0.130–0.131s / 18.7–19.8MB.
        `tests/reference/native-generation-real.json` records corpus/candidate
        hashes without redistributing word lists. Admission runs outside measured
        workers; ranking and end-to-end corpus-loading measurements remain open.
  - [x] Complete 60 serial end-to-end measurements across all five normal-user
        workloads: three cold/warm pairs per engine, alternating engine order,
        original candidate/deep budgets, Python workers=1 and exact displayed
        result/count checks. `tools/benchmark_native_cli.py` records process-tree
        RSS and wall time in `tests/reference/native-cli-performance.json`.
        Cold native medians are 0.78/1.03/1.59s versus Python 2.46/2.88/3.29s
        for the small cases, but 81.11/27.32s versus 52.61/23.28s for the two
        100k cases. Native warm medians are 0.55–10.95s versus Python 0.28–0.36s:
        native still reloads corpora and repeats generation/preparation. No broad
        speedup or parallelism benefit is claimed. Fresh puzzle caches are not
        cold OS caches; sampled tree RSS may double-count shared pages. Further
        optimization must target these measured regressions rather than weaken
        semantic workloads. The complete report is not a migration-performance win.
  - [x] Move ranking-cache lookup ahead of pre-scoring/preparation and redundant
        bigram parsing. Retain fresh corpus snapshots/generation and enforce the
        current hard-deep limit on the cached shortlist. Eight progress tests and
        strict Clippy pass. Three cold/warm pairs of the unchanged 100k
        `user_testing_anagrams` workload preserve reference results/counts;
        `native-cli-early-cache.json` records a 1.21s warm median versus the prior
        10.76s. Cold times varied (29.80–40.15s), so no cold improvement is claimed.
        Corpus loading and generation still prevent Python-like subsecond warm
        latency on this workload; the full performance baseline remains visible.

Gate: exact generation parity, cancellation coverage and measured time/RSS at
equal workloads. Explain any intentional ordering change before accepting it.

## Phase 3 — Port ranking and cache behavior

Current completion focus: audit optional phrase/model provenance end-to-end;
close the full ranking/quality checks with explicit optional-mode support limits;
verify cache identity, bounds, recovery and concurrency against the final code.
Do not use further optimization milestones as a substitute for closing these
acceptance requirements, and do not claim completion from stale parent checkboxes.

Phase-3 acceptance audit: `tests/reference/NATIVE_PHASE3_ACCEPTANCE.md` records
the current proof and explicit optional-mode boundaries. All 49 Rust tests and
the eight component/pipeline parity scripts pass; the 52-case ordering quality
gate also passes. Versioned provenance and bounded atomic cache requirements
are closed. Wiktionary was built from the official title dump and passes 12
native/Python pipeline cases. Under the user's updated acceptance policy, this
and the existing CLI/quality checks are sufficient: the exhaustive combined
Wikimedia matrix is optional, not a phase-3 blocker. Its unfinished corpus build
was stopped with partial files preserved under `.codex/temp/phase3-corpora`.
Phase 3 is complete for ranking/cache functionality. The subsequent phase-2
checkpoint above covers lexical completion and measured performance improvements.

- [x] Port lexical scoring, grammar/valency, retained order candidates, top-K,
      clause/auxiliary/comparative rules, order refinement/diversity and cohesion
      as independently reviewable changes.
- [x] Preserve optional phrase-index and learned-ranker semantics, including
      missing-data behavior and versioned model/input provenance.
  - [x] Port phrase hierarchy/cohesion and positive-only retained-order rescoring;
        compare shortlist admission and result rows against Python.
  - [x] Validate read-only SQLite integration, batched queries, missing/corrupt/
        invalid-schema errors and unchanged database contents
        (`tests/parity/corpus_storage.py`, `tests/parity/ranking.py`).
  - [x] Port the 18-feature learned-model schema, bounded model loading, scoring
        and deterministic optional-model ranking; validate invalid/missing models.
        Like the reference, this remains an explicit offline scoring utility,
        not an automatically enabled default-solver stage. Python training remains
        available; no new runtime model policy has been invented.
  - [x] Complete versioned optional-input/model provenance in native run results.
    - [x] Loaded learned models retain exact source-byte SHA-256/length/role via
          `Model::identity`, without changing the Python-compatible serialized
          model format. Programmatically created weights do not claim a file
          identity. Three provenance tests, 253 learned parity cases and Clippy
          pass.
    - [x] Add the typed offline `learned::rank_result` envelope with schema version,
          selected indices, optional model schema and exact loaded-model identity.
          No-model results explicitly carry no model provenance; programmatic
          models carry their schema but no fabricated source-file identity.
          Controlled scoring/sorting, provenance tests, 253 parity cases and
          strict Clippy pass. This does not enable models in default solve.
    - [x] Add controlled model loading with checks before opening, during bounded
          reads and around parsing. Preserve the 64 KiB ceiling and exact-byte
          provenance. Cancellation-before-I/O and oversized-input tests pass,
          alongside all three provenance tests and strict Clippy.
- [x] Validate score components and useful final rankings against reference and
      quality fixtures. Document numeric tolerances without requiring perpetual
      identical ranking; preserve hard correctness and investigate quality loss.
  - [x] Compare component scores at 1e-12 tolerance, with exact classifications,
        selected rows, retained orders and final bucket order in differential tests.
  - [x] Verify full CLI rankings, supported optional modes and quality gates.
    - [x] Add `tests/parity/registry.py` and pass all five frozen normal-user
          workloads cold and warm, including both 100,000-candidate cases at
          their original deep budgets. Compare generation/deep counts, exhaustion,
          all displayed ranks/rounded scores and exact cold/warm native rows.
          Use the pinned normal-user augmented dictionary/short-word policy and
          reference contraction display formatting; do not substitute research
          dictionary inputs. Three smaller workloads and the two large workloads
          passed in separate runs. Optional-mode and performance gates remain.
    - [x] Fix a cache float-roundtrip regression exposed by the full registry:
          enable serde_json's exact float parser so cached score components do
          not shift by one ULP. Four cache tests (including exact-bit regression
          values), full cold/warm registry checks and strict Clippy pass.
    - [x] Exercise optional SQLite phrase evidence through the full ranked CLI
          for required-word prefix and diverse searches. Both cases match the
          Python export/reranking pipeline in all row fields, counts, ordering
          and canonical SQLite provenance (`tests/parity/solve.py --phrase-only`).
          The harness closes writer and reader connections before temporary-file
          cleanup on Windows. This adds two optional cases to the eight baseline
          cases; it is not full optional-corpus quality acceptance.
    - [x] Extend retained-order differential coverage to all 52 ordering-registry
          phrases at beam 256/top 50, alongside 66 existing cases. This exposed
          and fixed a grammar tie-break regression: edge summation now matches
          Python's compensated sum. All 118 cases pass with exact order checks
          and the existing numeric tolerance; no tolerance was widened.
    - [x] Use the same compensated edge-sum helper for direct local grammar
          evaluation and retained ordering, avoiding divergent scores for the
          same phrase. All 300 grammar cases, 118 ordering cases and Clippy pass.
    - [x] Match Python sum behavior for lexical averages/tails and selected
          bigram-edge averages too. All 200 scoring cases, 12 ranking pipelines,
          eight real-corpus ranked CLI cases and strict Clippy pass, validating
          shortlist and final-order behavior after the accumulation corrections.
- [x] Re-run `ci_ordering_gate.py`, refinement/phrase checks and real user cases.
  - [x] Re-run the 52-case Python ordering gate after native arithmetic fixes:
        R@1 0.462, R@10 0.865, R@50 1.000, MRR 0.598, cross-bag margin
        0.088; every threshold passes. Native differential refinement (160 cases)
        and phrase/cohesion (400 cases) also pass. This does not substitute for
        full native CLI quality/default-path parity or the real phrase-DB matrix.
- [x] Implement versioned, bounded caches keyed by normalized semantic options,
      engine/ranking version and corpus identity. Keep budgets/strategy in keys
      whenever they affect results. Do not deserialize Python pickle in Rust.
  - [x] Define a versioned ranked-request SHA-256 key over normalized lexical
        strings, all serialized semantic settings/budgets, engine/ranking
        versions and role-sorted exact input identities. Preserve constraint
        order and required multiplicity conservatively. Reject incomplete,
        duplicate-role or malformed identities. Identity regression test and
        strict Clippy pass. This is the key foundation only: bounded persistent
        storage, solve integration and cache-hit admission remain unfinished.
- [x] Preserve atomic writes, corruption recovery, concurrent writer safety and
      rebuild bypass. Distinguish cached computation timings from request latency.
  - [x] Enforce reduced entry/payload limits when reopening an existing cache,
        rather than waiting for its next write. Transactional FIFO pruning is
        shared with publication. Five cache tests, eight progress tests and
        strict Clippy pass, including persisted-size checks after lowering bounds.
  - [x] Add a dedicated native SQLite storage layer with transactional replacement
        and FIFO pruning by entry count and total payload bytes. Verify persisted
        SHA-256 integrity before returning payloads; remove corrupt entries,
        reject oversized writes and support lookup bypass for rebuilds. Preserve
        original computation milliseconds separately. Lock conflicts return
        errors without partial publication or eviction; database-level corruption
        is non-destructive and still needs adapter fail-open handling. Three cache
        tests and strict Clippy pass. Bounds exclude SQLite page overhead; this
        layer is not yet wired into solve, so no runtime cache-hit claim is made.
  - [x] Integrate optional ranking-cache reuse in `solve_cached_observed` after
        fresh corpus identity capture, generation and hard-deep admission. Reuse
        only final ranked payloads, never old job IDs/policies/lifecycle states.
        Keep the phrase snapshot open from identity capture through scoring.
        Eight progress tests and three cache tests cover cold/warm equality,
        policy rejection on a hit, corpus invalidation, rebuild, corrupt-entry
        recovery, cancellation before publication and non-destructive fallback
        for an unreadable database. Strict Clippy and all ten real-corpus CLI
        parity cases pass. CLI cache options, exposed timing provenance and
        generation/corpus-load cache performance remain unfinished.
  - [x] Expose `solve --cache FILE`, `--rebuild-cache`, `--cache-max-entries`
        and `--cache-max-bytes` with bounded defaults (128 entries / 64 MiB
        retained payloads). Reject missing, duplicate and incompatible options.
        Actual executable cold/warm/rebuild checks preserve exact output rows
        and terminal progress; the targeted CLI test and strict Clippy pass.
        Document fresh versus reused work in `contracts/README.md`. Timing
        reporting and generation/corpus-load performance remain separate work.
  - [x] Expose optional cache-enabled result timings with current `execution_ms`
        separate from original `ranking_computation_ms`. Fresh ranking duration
        excludes cache serialization/writes; request execution excludes adapter
        JSON transport. Regenerate schema/TypeScript and verify all 17 shared
        fixtures. Actual CLI tests verify timing preservation on a hit; two
        optional SQLite real-corpus cases each pass cold/warm/rebuild parity
        against Python, including all row fields and original timing reuse.
        Strict Clippy and contract checks pass. This is timing provenance, not
        a native performance acceptance result.

Gate: no unexplained quality regression, reproducible rankings and complete
default-path parity. Explicitly list unsupported optional modes and retain Python
for them until ported; do not declare the migration complete prematurely.

## Phase 4 — Windows desktop app to try on this PC

- [x] Build a Tauri Windows app with a native window, bundled Svelte/TypeScript
      UI and the existing Rust core. Deliver a launchable local release, not just
      a dev-server page, CLI or mockup. No Python/Node runtime dependency.
- [x] Add a thin typed desktop bridge for capabilities and job start/status/
      result/cancel using the existing contracts. Keep solver policy and ranking
      in Rust; do not duplicate them in frontend code. No HTTP server required.
- [x] Start with one active solve and a bounded queue/executor. Keep the window
      responsive; propagate cancellation/deadlines to the engine and acknowledge
      worker completion. Closing the app must not orphan solving processes.
- [x] Support the existing local corpora on this PC with explicit path selection,
      validation and actionable missing-data errors. Remember user preferences;
      keep corpora, settings, solver cache and WebView profile beside the executable
      in the writable per-user install. Share data across releases; do not download per job.
- [x] Provide input, hints/required/excluded words, useful advanced options,
      run/cancel, progress, ranked results, copy/export and clear empty/error
      states. Distinguish generated/deep/shown counts, exhaustion and truncation.
- [x] Show local execution and effective budgets; retain the native engine's
      explicit limits and optional-input support boundaries. Use a narrow Tauri
      capability surface rather than arbitrary shell/filesystem frontend access.
- [x] Provide usable light/dark styling, keyboard navigation, readable scaling,
      reduced-motion support and a compact desktop layout. Keep UI transport
      behind a typed adapter so a future website does not require another solver.
      Do not build website theming/SEO/shared-package infrastructure prematurely.
- [x] On successful normal local app builds, dogfood into
      `%LOCALAPPDATA%\Programs\TajemnikTV\AnagramSolver` and restart the app.
      Keep one current release and one rotating rollback backup. Never dogfood
      CI or test-only builds. Verify installed startup, not only dev-mode startup.
- [x] Exercise real solving, cancellation, missing corpora, restart/settings
      persistence and copy/export in the installed Windows app. Use focused UI
      automation where supported and get the user's visual/usability feedback.

Completed on this PC (2026-09-17). Native Windows automation exercised the
installed app, not a browser mock: real 10,000-candidate solve (241 analyzed,
21 shown), Copy all, native JSON export, missing-corpus error/recovery,
cancellation acknowledgement and terminal state, light/dark/system appearance,
keyboard submission and zoom through 150%. Restart restored input and theme.
User testing supplied text export and screenshots and caught the candidate-budget
step bug, now fixed and retested. Cancellation is no longer presented as failure.
Three desktop tests and strict Clippy pass; final Svelte checks and release build
pass. Current and single rollback executable hashes are verified. Original user
preferences were restored. See `apps/desktop/VERIFICATION.md` for proof boundaries.

Runtime-data follow-up: normal local installs now provision the ten required
corpus files (20.52 MiB) into `%LOCALAPPDATA%\Programs\TajemnikTV\AnagramSolver\corpora`,
shared across executable releases. First copies are hash-verified; existing data
is preserved. Settings/cache/WebView state live in adjacent `user-data`, migrated
from the old app-local profile. Default repository paths migrate automatically while custom paths
remain unchanged. The installed app needs neither the repository nor the CLI;
the Rust engine is linked into the desktop executable. A real solve using the
provisioned paths, migration test, four desktop tests and strict Clippy pass.

Gate: the user can launch the installed app on this PC and solve real puzzles
without a terminal, dev server, Python or Node. The real engine, not a mocked
backend, must drive the UI. Website integration, mobile web testing, public
hosting and other operating systems are not phase-4 prerequisites.

### Desktop/native Python-feature follow-up (2026-09-17)

- [x] Native pairwise ranker training and grouped held-out evaluation, feature-pool
      building and explicit model ranking CLI. Model selection in desktop Settings;
      loaded-byte identity participates in provenance and cache invalidation.
- [x] Opt-in bounded order refinement in the live solve pipeline and desktop.
- [x] Settings Experimental tab for explicit, bounded local training from prepared
      groups, held-out evaluation and separate model activation. No automatic
      downloads or training. Automated desktop checks cover publication without
      activation and cancellation/shutdown; manual testing deferred by request.
- [x] Desktop maintenance: Data download/offline preparation, separate corpus-set
      updates and native validation; Experimental labelled-case dataset builder
      and persistent held-out reports; Diagnostics identities, timings and cache
      information. Shared native CLI library, no subprocess/runtime dependency.
- [x] Parallel phrase preparation (up to eight parsing workers), bounded duplicate
      aggregation, sorted SQLite writes and compact single-B-tree storage. Serial/
      parallel counts match; a 150,000-title synthetic benchmark improved from
      8.188 s to 1.606 s with identical rows (not a full Wikimedia speed guarantee).
- [x] Explicit 1–32 ranking workers with deterministic row assembly, aggregate
      progress, cooperative cancellation and joined worker lifetimes; default is 1.
- [x] Desktop extended/exhaustive generation (budget 0) and quick/balanced presets.
      Standard limits remain the default; extended mode is deliberately session-only.
- [x] Force recompute and clear-result-cache controls; preserve settings and corpora.
- [x] Contraction formatting shared by native result display and text export;
      original normalized word arrays remain available in JSON.
- [x] Python-free dictionary augmentation, phrase-title SQLite builder (text/gzip,
      multiple input dumps), and explicit data download/preparation script. No
      downloads occur during ordinary solving or app builds.
- [x] Installed-window smoke: two-worker refined force-recompute solve completed
      (9,999 generated, 241 analyzed, 21 shown); new Advanced/Settings controls
      inspected and original search settings restored. Exhaustive admission,
      model selection semantics and cache clearing also have backend tests.

Automated coverage includes serial/parallel equality, cancellation, model-byte
cache invalidation, refinement, exhaustive admission, non-destructive cache clear,
native training/build/rank round trip, gzip corpus ingestion and no-clobber outputs.
See `NATIVE_TOOLS.md` for commands and retained intentional boundaries. No new
large-workload speedup or learned-model quality claim is made by these changes.

## Phase 5 — Website integration and constrained hosting (deferred)

Do not start this phase until the user explicitly resumes website work. No
website repository, VPS or production changes are needed for phase 4. When this
phase resumes, recheck the historical website findings above and validate a
same-origin service boundary, Host/Origin handling, scoped theme tokens and the
real hosted UI independently of the desktop bridge.

- [ ] Add the SvelteKit page and same-origin `/api/anagram/*` proxy in the website
      repository. Allowlist upstream paths/methods and sizes; never accept an
      arbitrary upstream URL. Keep upstream configuration server-only.
- [ ] Keep the Rust service private on the container network. Preserve admin host
      isolation and the site's same-origin CSP. Inspect actual Caddy/Tunnel config
      before integration; do not assume repository documentation proves live state.
- [ ] Start conservatively with one active hosted job, one CPU worker and a small
      bounded queue. Derive candidate/deep-ranking, input/word-count, elapsed-time,
      memory, result-size and queue limits from measurements under the VPS quota.
- [ ] Enforce limits in the Rust service regardless of UI values. Reject public
      unlimited generation, oversized requests and unsupported expensive options.
- [ ] Apply admission control/rate limits before expensive work, bounded result
      retention and cache eviction. Keep per-job ownership tokens, prevent result
      enumeration and avoid logging raw puzzle inputs by default.
- [ ] Cancel queued and running jobs safely; expire abandoned work. On restart,
      report expired/lost jobs rather than fabricate success. No durable queue
      infrastructure is needed for the initial single-instance deployment.
- [ ] Add container CPU/memory limits, non-root/read-only runtime where practical,
      pre-provisioned corpus data and health/readiness endpoints. Do not download
      corpora for each request or borrow the website's database credentials.
- [ ] Test overload, cancellation latency, memory pressure and website responsiveness
      alongside real host workloads. Tune admission limits before opening access.

Gate: measured acceptable impact on the VPS, known rollback and explicit approval
before deployment. If the host cannot afford even the minimum useful search,
publish the tool page with local-download guidance and hosted solving disabled.
Do not let a solver overload degrade the main site.

## Phase 6 — Distribution and cutover

This is broader distribution after the local desktop milestone, not a blocker
for trying the Windows app. Website/VPS-specific work stays deferred with phase 5.

- [ ] Build/test native Windows, Linux and macOS artifacts, including the actual VPS
      architecture. Cross-compilation alone is not runtime validation.
- [ ] Package static UI plus Rust runtime; choose per-user writable data/cache paths,
      support offline reuse, and verify startup without Python/Node installed.
- [ ] Package or fetch legally redistributable corpora with version/hash validation,
      interruption recovery, clear download progress and license notices.
- [ ] Add release CI: Rust formatting/lints/tests, differential/quality gates,
      desktop UI type checks and integration tests. Add website build/output-leak
      checks only when website integration resumes.
- [ ] Benchmark optimized Rust release builds at equal corpus/options/budgets over
      multiple runs. Publish end-to-end latency, peak RSS, candidate counts and
      quality—not only hot-loop throughput or a single timing.
- [ ] Make Rust the default only after gates pass; retain a documented Python
      fallback through at least the initial release validation period.

Optional later work: browser WebAssembly prototype if
corpus size, browser storage, startup and worker constraints prove acceptable.
It is not required for the desktop release. WASM needs its own data/threading and
browser validation plan; native Rust alone does not deliver browser-local solving.

## First implementation batch

1. Freeze the Python reference and corpus manifest; land pending improvements.
2. Add wire contracts and differential fixtures.
3. Add the Rust workspace and exact generator vertical slice.
4. Measure and complete generation parity before expanding the port.

Keep these reviewable separately. Do not combine an engine rewrite, website
redesign and production infrastructure change into one patch.
