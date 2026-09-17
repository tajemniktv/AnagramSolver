# Native wire contracts

The first implemented operation is `generate`, schema version 1. It is a
generation-only development interface, not the ranked application endpoint.
Unknown request fields and unsupported schema versions are rejected.

Run the CLI with a local dictionary and optional unigram count file:

```text
anagram-cli generate DICTIONARY [UNIGRAMS] < request.json
```

Example request (all scalar fields currently explicit):

```json
{
  "schema_version": 1,
  "text": "ate",
  "required": [],
  "hints": [],
  "excluded": [],
  "min_words": 1,
  "max_words": 3,
  "min_word_length": 1,
  "max_word_length": 20,
  "min_zipf": 0,
  "candidate_budget": 100,
  "allow_repeat": true,
  "strategy": "prefix",
  "hint_mode": "any"
}
```

Constraint arrays contain individual words, not phrases or comma-separated lists.
Required-word multiplicity is preserved. Bounds include required words. Input
normalization follows the Python NFKD/ASCII-ignore behavior. At least one hint
must match (`any`), or exactly one distinct hint (`exactly_one`). Impossible hints
are removed only while another usable hint remains; otherwise validation fails.
Required words never override exclusions.

`exclude_regex` optionally supplies case-insensitive search patterns over
normalized words (default `[]`), in both generate and solve requests. Patterns
filter dictionary words and hints, and conflicting required words are rejected.
The linear-time Rust regex syntax supports anchors, classes, groups, alternation
and repetition, but not backreferences or look-around; unsupported or malformed
patterns return `invalid_regex`, never silently fall back. Limits: 64 patterns,
16 KiB combined UTF-8 source, 2 MiB compiled size and 2 MiB DFA cache. These
explicit bounds keep user expressions from introducing unbounded backtracking.

`candidate_budget: 0` means unlimited generation. A deployment with a finite
candidate limit rejects it. Without an explicit policy this development CLI is
local-only and unbounded; that default is not a public-service configuration.
`kind: generation_only` explicitly distinguishes unranked word bags. `stop` is
`exhausted`, `candidate_cap`, `cancelled`, or `timed_out`; only the first proves
complete enumeration. The extra-bag probe distinguishes exact cap exhaustion
from actual truncation. CLI cancellation/deadline configuration is not exposed
through the request yet; `--timeout-ms N` supplies a local execution deadline.
The engine accepts an explicit shared cancellation control. A timeout in corpus
loading is an error with code `timed_out`, not a false empty/exhausted result.

Failures exit with status 2 and emit `{schema_version: 1, error: {code, message}}`.
Stable codes include `unsupported_version`, `invalid_limits`,
`invalid_word`, `empty_input`, `conflicting_constraints`, `unavailable_letters`,
`impossible_hints`, `missing_frequency_data`, `corpus_error`, `invalid_search`,
`usage`, `input_error`, `request_too_large`, `invalid_json`, `invalid_timeout`,
`invalid_ranking_limits`, `invalid_regex`, `ranking_error`,
`invalid_deployment_limits`, `deployment_limit_exceeded`, `invalid_job_id`,
`serialization_error`, `cancelled`, and `timed_out`. The complete error envelope
has the generated `ErrorResponse` schema; `SolverError` describes its nested
code/message object. Messages provide details but are not stable dispatch keys.

## Ranked development operation

### Optional native ranking cache

`solve --cache FILE` enables a dedicated SQLite ranking cache. The default bounds
are 128 entries and 64 MiB of retained payloads; override them with positive
`--cache-max-entries N` and `--cache-max-bytes N` values. SQLite page/journal
overhead is additional. The parent directory must already exist. Never use a
corpus database as the cache file. These flags are deployment controls, not
semantic JSON request fields, and are rejected for `generate`.

`--rebuild-cache` requires `--cache FILE`: it bypasses lookup and replaces the
matching entry after successful computation. Unavailable/busy cache storage
falls back to ordinary computation; an unreadable database is not deleted or
replaced. Corrupt individual payloads are ignored and rebuilt.

Cache hits still capture exact corpus identities, but bypass generation,
pre-scoring, preparation and deep/corpus ranking. Cache format 2 stores the
completed generation count and exhaustion/truncation outcome; older entries
are misses. The current candidate budget and completed-count invariants are
validated, and the current hard-deep policy
is checked against the cached shortlist count, never the displayed row count.
The key binds that count to the exact semantic request and consumed corpora.
`status.cache.hit` identifies reuse; on a hit, generation and ranking counters
describe the reused result's completed work, not fresh evaluations. Job status,
deployment budgets and cancellation belong to the current request. Original
ranking computation milliseconds are exposed separately when caching is enabled:

- `timings.execution_ms` measures this solve invocation, including corpus loading,
  cache access and observers, but excluding CLI JSON input/output transport.
- `timings.ranking_computation_ms` measures deep/corpus ranking and final bucket
  assembly, excluding cache serialization/writes. On a hit it is the original
  saved duration, not fresh work and not this request's elapsed time.

These millisecond measurements can be zero for short operations. The optional
`timings` object is omitted when no cache configuration was requested. Cache
failure still reports fresh execution/ranking timing if solving succeeds.

```text
anagram-cli solve DICTIONARY UNIGRAMS BIGRAMS WORDNET [PHRASE_DB] < request.json
```

The request has a `generation` object containing the complete request above,
plus these explicit fields:

```json
{
  "deep_per_group": 5000,
  "deep_all": false,
  "order_mode": "auto",
  "beam_width": 128,
  "exact_max_words": 5,
  "retained_orders": 56,
  "phrase_rescore_top": 300,
  "phrase_bonus_max": 5.0,
  "positive_bigrams": true,
  "result_limit_per_group": 20
}
```

`order_mode` also accepts `exact` and `beam`. Word counts above 10 are rejected
rather than silently approximated. The deep shortlist expands selected morphology
families, so `deep_per_group` is not a hard work cap. Like the Python ranked
frontend, this operation rejects required words that consume the entire target.

Results use `kind: ranked`, separate `generated`, `deep_analyzed`, `shown`,
`orders_evaluated` and `corpus_rescored` counts, `generation_stop`, and ranked
`buckets` keyed by word count. Rows include lexical/grammar/structure/corpus score
components. The generator's historical decimal export quantization is preserved
before preparation because it affects ranking and ties.

This is still a development interface: no persistent cache or complete
interruption coverage yet. Python remains the default application. Full native
parity/performance acceptance and the application adapters remain required before
replacement/frontend adoption.

## Generated types and checks

Rust structs are the structural source of truth. `tools/generate_contracts.py`
uses Schemars' Serde-compatible derivation to produce checked-in JSON Schema
2020-12 and TypeScript files in `contracts/generated/`. Do not edit generated
files. Schema validation covers wire shape; cross-field constraints and version
acceptance remain engine validation, not TypeScript refinements.

```text
python tools/generate_contracts.py
python tools/generate_contracts.py --check
npm --prefix contracts ci --ignore-scripts
npm --prefix contracts run check
python -m pip install --target .codex/temp/schema-python jsonschema==4.26.0
python tests/parity/contracts.py
cargo test --workspace --locked
```

`JobStatus` defines queued/running/succeeded/cancelled/timed_out/failed states,
stage, counts, effective budgets, version identities, cache flags and exhaustion.
Terminal states cannot restart or change outcome. `unknown` exhaustion is required
when interruption prevents proving exhaustion, even if the number collected equals
the candidate cap. Family-expanded shortlist size and a deployment hard deep limit
are separate fields. Ranked execution now populates these definitions; it is
still synchronous, not a background job scheduler or HTTP service.

Scalars in generation/ranking requests remain explicit (no hidden defaults);
constraint arrays default to empty. The fixtures check Rust deserialization and
independent Python/JavaScript schema consumers; TypeScript includes negative type
checks. Request integers must be at most 9,007,199,254,740,991 so JavaScript does
not silently round them; schemas and native semantic validation enforce that
bound. TypeScript `number` alone is not a runtime validator.

Validation precedence is JSON shape first, generation schema version, generation
numeric ranges, normalized nonempty input, constraint tokens, required/excluded
conflicts and required letter availability, then usable hints. Ranked requests
apply that same validator before ranking limits. Fully required answers (including
repeated required words) are ranked normally when they consume exactly the target;
native solve no longer reproduces Python's zero-residual restriction.
Corpus I/O follows semantic validation; a missing corpus must not hide an invalid
request. An already-expired execution control takes precedence over work errors.
Frequency-data availability is checked when the generation input is loaded.
The version-1 default policy is shared: semantic scalar options are required,
constraint lists default to empty, and absent deployment limits mean unbounded
local execution. No hidden candidate/deep budget is supplied by an adapter.
JSON Schema checks shape/ranges; `request::validate`, `solve::validate` and
`JobStatus::validate` own cross-field semantic checks. TypeScript types are not
runtime validators. Status validation checks lifecycle/error consistency,
completed-work bounds, unique well-formed data identities, and complete provenance
before success. Debug/test builds check every emitted status against these rules.

## Adapter-owned deployment limits

Generation requests also support `short_policy` (`none`, `common`, `all`, default
`common`) and `extra_short_words` (default empty). Extra words are normalized into
the short whitelist, not injected into the dictionary. Like Python, the common
and explicit whitelist may bypass minimum length even under `none`; maximum
length, exclusions and frequency filtering still apply unless separately forced.

`forbid_chars` defaults to an empty string and uses the same letter normalization
as the target. Required words cannot bypass forbidden letters; conflicts fail
before corpus loading. Forbidden hints are removed from the usable hint set, and
an entirely unusable requested set returns `impossible_hints`.

Both CLI operations accept `--limits policy.json` (at most 64 KiB). The generated
`DeploymentLimits` contract is separate from the semantic request. Every field
defaults to `null` (no deployment limit); unknown fields and invalid numeric
limits are rejected. The future service must supply its own fixed policy, not
accept a caller-selected policy file.

Supported fields: `max_input_bytes` (UTF-8 bytes of the target text),
`max_normalized_letters`, `max_candidates`, `max_deep_analyzed`, `max_beam_width`,
`max_retained_orders`, and `timeout_ms`. Count/size limits are positive exact
integers; zero timeout is an immediate deadline. The CLI's separate 1 MiB JSON
body limit remains in force.

Requests exceeding admission limits return `deployment_limit_exceeded`; the
engine does **not** silently change the candidate budget, beam, or retained
orders. The hard deep limit is checked against the actual family-expanded
shortlist, before ordering, rather than only `deep_per_group`. It can therefore
reject a request after generation/preparation. Policy deadlines share cancellation
with the caller and can shorten, never extend, an existing `--timeout-ms` deadline.
Malformed policy configuration returns `invalid_deployment_limits` before request
execution. Library callers use `solve_with_limits`; the ordinary local wrapper
supplies the explicit unbounded policy.

## Ranked execution progress

`solve --progress ...` emits one `JobStatus` JSON object per stderr line while
stdout remains the single result/error JSON object. Successful ranked results
also contain the terminal `status`. Library adapters use `solve_observed` with
their own job ID and callback; the one-shot CLI uses `local`. The generation-only
primitive does not advertise job events.

Only admitted requests create an execution. Invalid requests, invalid policies
and an already-expired caller control can fail before a queued event. Accepted
executions emit queued, running stages, and exactly one terminal outcome. Failed
and interrupted executions preserve the stage where they stopped and the last
confirmed counts. The observer sees terminal failures even though no result is
returned. No rows are marked shown on failure/cancellation.

Counters acknowledge completed work: generation reports after enumeration;
deep analysis reports after each completed bag; corpus rescoring reports after
each completed row, preserving that count if the pass is interrupted. Evaluations
inside an interrupted bag are included once each order score completes; deep bag
counts still require a completed bag. Order events are batched every 256 scores
with an exact final count on interruption. Unfinished corpus rows are not included. These are not live inner
loop counters or a claim of complete interruption accounting. Generation
exhaustion is independent of job success: a cancelled ranking job can still have
exhausted generation. Interruption before the generation probe finishes leaves
exhaustion unknown.

Effective candidate/deep/result limits and the remaining deadline at execution
admission are populated. Engine/ranking versions and false cache flags reflect
the current uncached engine. `versions.data_complete` is false until all enabled
inputs are identified; failed/early events can contain only a partial inventory.

## Data identity representations

Each identity contains a stable role, representation, presence flag, byte count
and lowercase SHA-256. No local filesystem paths are exposed in wire provenance.

- `file_bytes`: exact dictionary/unigram/bigram or WordNet bytes consumed by the
  run. Optional absent WordNet files use `present: false`, zero bytes and the
  empty SHA-256; an empty **present** file is distinct. Unigram/bigram parsers
  reuse immutable captured buffers for both scoring passes instead of reopening
  a possibly changed path. These buffers add their raw corpus sizes to peak
  memory until the second pass finishes; native performance remains to be measured.
- `phrase_rows_v1`: a logical SQLite read-snapshot identity, **not** a database
  file hash. Hash UTF-8 compact JSON `["schema", create_table_sql]` followed by LF,
  then compact JSON `[text,n,count]` plus LF for every row, ordered by
  `text COLLATE BINARY,n,count`. SQL NULL encodes as JSON null. Byte count refers
  to this canonical stream. The table definition is included because collations
  can affect queries. Dynamic views and duplicate text keys cannot provide this
  deterministic identity and are rejected when provenance is requested.

SQLite's read transaction covers MAX(n), identity capture and subsequent score
queries, including when another connection commits WAL changes. Hashing and
reads honor the execution control. Optional learned models remain explicit
offline utilities, not silently enabled solver inputs.

The offline library's `learned::rank_result` returns versioned selected indices,
model schema and optional exact loaded-model identity. Programmatic models have
no source-byte identity; baseline-only results have neither model field populated.
The supplied execution control covers scoring and stable sorting. This library
envelope is separate from the default solver and its generated transport schemas.
