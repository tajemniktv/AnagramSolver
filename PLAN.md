# AnagramSolver: Rust engine and shared UI plan

Status: implementation in progress, 2026-09-17. Production deployment still
requires explicit approval.

Current implementation goal: **phases 0–3 fully**, per the user's scope update.
Phases 4–6 remain future roadmap context and are not part of this active goal.

### Implementation evidence

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

Build one behavior-tested Rust solver that powers a cross-platform local app
and a page on tajemniktv.com. Share the solver controls and results UI, not the
entire website shell. Keep expensive testing and larger searches on the PC;
offer only bounded, measured workloads on the CPU-constrained VPS.

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
2. Start with a proposed main-site route **`/tools/anagram-solver`**. Keep the
   website's existing layout and SEO conventions. A subdomain is a later option:
   today's project-host gates do not accept arbitrary tool/API paths.
3. Use a Rust library plus thin CLI/server adapters. Proposed server stack:
   Axum/Tokio for HTTP and a dedicated, bounded CPU executor, optionally Rayon.
4. First standalone release is a local executable serving the same UI in a
   browser. It must not require Node or Python at runtime. Tauri packaging can
   follow if a native window/installer adds value; it is not a prerequisite.
5. Keep website content/admin/database authority unchanged. The solver is a
   separate service, not compute inside SvelteKit request handlers or PostgreSQL.

## Ownership and proposed layout

Keep engine, contracts, shared UI and local packaging in this repository:

```text
Cargo.toml                    # Rust workspace, introduced in phase 1
crates/anagram-core/           # normalization, generation, ranking; no HTTP/UI
crates/anagram-cli/            # commands and machine-readable output
crates/anagram-server/         # jobs, policy, HTTP, local static UI hosting
packages/solver-ui/            # reusable Svelte components + typed API client
apps/standalone/               # small Svelte/Vite shell, built into local package
contracts/                    # versioned wire schema and contract fixtures
tests/parity/                 # Python/Rust differential fixtures and harness
```

These are proposed paths, not existing components. Avoid splitting the Rust core
into many crates before ownership or compile-time evidence requires it.

The website repository owns its route, SEO, host-theme adapter and a thin
same-origin API proxy. Consume a versioned solver-ui package/artifact pinned to a
release; use a local package link during development. Do not copy component
sources into both repositories or introduce a monorepo migration merely to share
them. Prove package consumption in both builds before stabilizing publication.

The core owns immutable shared lexical data and explicit per-request mutable
state. Replace Python's scoped hooks/global locking with explicit dependencies;
do not transliterate monkeypatch layers into Rust global state.

## Phase 0 — Freeze behavior and establish measurements

- [x] Review and land the existing Python audit/improvement work separately from
      port commits; preserve unrelated changes. Record the exact reference commit.
- [ ] Capture corpus versions/hashes, ranking configuration, cache state, machine
      details, candidate counts and reference outputs in reproducible fixtures.
- [ ] Inventory dictionary, frequency, WordNet, phrase SQLite and optional feature
      ranker inputs; document formats, licenses and redistribution requirements.
- [ ] Extend the reference harness to cover letter normalization, punctuation and
      Unicode behavior, repeated required words, alternative hints, exclusions,
      impossible inputs, empty results, exact cap exhaustion and deterministic ties.
- [ ] Include prefix/diverse enumeration, exhaustive-generation semantics, grammar,
      ordering, optional phrase evidence and optional learned ranking in parity scope.
- [ ] Benchmark generation and ranking separately, then actual end-to-end cold and
      warm CLI runs using `benchmark_user_runs.py` and the existing quality gates.

Gate: a pinned, repeatable oracle and data manifest, not snapshots from a moving
worktree. Preserve the current default prefix strategy: earlier local 20k-cap
measurements showed diverse could cost substantially more without recall benefit
on the sampled cases. Re-evaluate on a broader set rather than generalizing that
small sample. Distinguish empty puzzle caches from cold corpus/OS caches.

## Phase 1 — Contracts and a minimal Rust vertical slice

- [ ] Define typed request/result/error contracts with a schema version and
      generated or mechanically checked TypeScript types.
- [ ] Separate semantic search options from deployment limits. Include input,
      hints/required/excluded words, word-count constraints, lexical options,
      generation strategy, candidate budget, deep-ranking budget and result limit.
- [ ] Specify validation precedence, defaults, normalization and stable error codes.
      Reject non-finite values and contradictory constraints; never relax silently.
- [ ] Define job states: queued, running, succeeded, cancelled, timed_out, failed.
      Include stage, counts, effective budgets, engine/data versions and cache flags.
- [ ] Preserve the distinction between generated bags, deep-analyzed bags and shown
      rows. Distinguish exhausted search, generation cap and deadline termination;
      report unknown exhaustion when a deadline prevents the extra-candidate probe.
- [ ] Implement corpus loading, normalization and a small deterministic generation
      path through the Rust CLI. Keep ranking in Python until its own phase.

Gate: schema fixtures pass in Python/Rust/TypeScript; a minimal Rust request works
end-to-end against the oracle. A generation-only result is not advertised as a
replacement for the ranked solver.

## Phase 2 — Port exact generation

- [ ] Port letter inventories, vocabulary filtering, required/hint handling,
      pruning and word-count traversal. Preserve repeated-word multiplicity.
- [ ] Implement bounded prefix and opt-in diverse traversal, including deduplication,
      redistribution of budget and the extra unique-bag truncation probe.
- [ ] Add cooperative cancellation/deadline checks in expensive loops and corpus
      operations. Limits must bound work, not merely the displayed result count.
- [ ] Differential-test full candidate sets for small exhaustive cases and ordered
      capped prefixes for bounded cases. Add seeded randomized small alphabets.
- [ ] Benchmark serial first; introduce bounded parallelism only where it helps and
      does not change deterministic output or consume unbounded memory.

Gate: exact generation parity, cancellation coverage and measured time/RSS at
equal workloads. Explain any intentional ordering change before accepting it.

## Phase 3 — Port ranking and cache behavior

- [ ] Port lexical scoring, grammar/valency, retained order candidates, top-K,
      clause/auxiliary/comparative rules, order refinement/diversity and cohesion
      as independently reviewable changes.
- [ ] Preserve optional phrase-index and learned-ranker semantics, including
      missing-data behavior and versioned model/input provenance.
- [ ] Compare score components, final rankings and ties against Python. Define any
      floating-point tolerance explicitly; do not mask ranking regressions with it.
- [ ] Re-run `ci_ordering_gate.py`, refinement/phrase checks and real user cases.
- [ ] Implement versioned, bounded caches keyed by normalized semantic options,
      engine/ranking version and corpus identity. Keep budgets/strategy in keys
      whenever they affect results. Do not deserialize Python pickle in Rust.
- [ ] Preserve atomic writes, corruption recovery, concurrent writer safety and
      rebuild bypass. Distinguish cached computation timings from request latency.

Gate: no unexplained quality regression, reproducible rankings and complete
default-path parity. Explicitly list unsupported optional modes and retain Python
for them until ported; do not declare the migration complete prematurely.

## Phase 4 — Local service and shared UI

- [ ] Implement a versioned capabilities endpoint and job create/status/result/
      cancel endpoints. Start with bounded polling; add SSE only if useful.
- [ ] Use one bounded CPU pool shared by jobs, not one pool per request. Keep HTTP
      handling responsive while solving. An async timeout must not leave CPU work
      running: cancellation must reach and be acknowledged by the engine.
- [ ] Serve built UI assets and API from one loopback origin. Validate Host and
      Origin, use a per-launch authorization mechanism for mutations, and avoid
      permissive CORS. Support safe port selection and clean shutdown.
- [ ] Build shared input/advanced options, run/cancel, progress, ranked results,
      copy/export and clear empty/error states. Label bounded searches honestly.
- [ ] Show compute location and effective limits. Local larger budgets are explicit;
      loading the public website never silently connects to a localhost daemon.
- [ ] Keep UI API access behind a typed client so local and hosted shells only
      supply base URL, capabilities and theme—not separate solving behavior.

Theme contract: scoped `.anagram-solver` components consume `--as-*` properties
with standalone defaults. The website wrapper maps, for example:

```css
.anagram-solver {
  --as-background: var(--color-surface);
  --as-text: var(--color-ink);
  --as-muted: var(--color-ink-muted);
  --as-accent: var(--color-purple);
  --as-border: var(--color-line);
  --as-radius: var(--radius-md);
  --as-font: var(--font-sans);
  --as-font-mono: var(--font-mono);
}
```

Do not import the website's global `layout.css`, body backgrounds, initialization
overlay, navigation or footer into standalone. Give standalone a compact header
and usable light/dark defaults. Avoid global Tailwind resets in the shared package;
test inherited website styles as well as clean standalone styles. Do not create a
second main landmark inside the website's existing main.

Gate: identical functional Playwright scenarios against the real Rust service in
both a standalone shell and website integration; keyboard/mobile/reduced-motion,
contrast and human visual review. Mock-server tests supplement, not replace this.

## Phase 5 — Website integration and constrained hosting

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

- [ ] Build/test native Windows, Linux and macOS artifacts, including the actual VPS
      architecture. Cross-compilation alone is not runtime validation.
- [ ] Package static UI plus Rust runtime; choose per-user writable data/cache paths,
      support offline reuse, and verify startup without Python/Node installed.
- [ ] Package or fetch legally redistributable corpora with version/hash validation,
      interruption recovery, clear download progress and license notices.
- [ ] Add release CI: Rust formatting/lints/tests, differential/quality gates,
      UI type checks/unit/browser tests and website build/output-leak checks.
- [ ] Benchmark optimized Rust release builds at equal corpus/options/budgets over
      multiple runs. Publish end-to-end latency, peak RSS, candidate counts and
      quality—not only hot-loop throughput or a single timing.
- [ ] Make Rust the default only after gates pass; retain a documented Python
      fallback through at least the initial release validation period.

Optional later work: Tauri native packaging; browser WebAssembly prototype if
corpus size, browser storage, startup and worker constraints prove acceptable.
Neither is required for the first release. WASM needs its own data/threading and
browser validation plan; native Rust alone does not deliver browser-local solving.

## First implementation batch

1. Freeze the Python reference and corpus manifest; land pending improvements.
2. Add wire contracts and differential fixtures.
3. Add the Rust workspace and exact generator vertical slice.
4. Measure and complete generation parity before expanding the port.

Keep these reviewable separately. Do not combine an engine rewrite, website
redesign and production infrastructure change into one patch.
