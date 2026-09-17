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
Stable codes currently include `unsupported_version`, `invalid_limits`,
`invalid_word`, `empty_input`, `conflicting_constraints`, `unavailable_letters`,
`impossible_hints`, `missing_frequency_data`, `corpus_error`, `invalid_search`,
`usage`, `input_error`, `request_too_large`, and `invalid_json`.

## Ranked development operation

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

This is still a development interface: no persistent cache, corpus identities,
emitted job/progress events or complete interruption coverage yet. Python remains the
default application. Still required before frontend adoption: semantic contract
validation, deployment limits, corpus provenance and complete parity/performance
acceptance gates.

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
are separate fields. These are transport-neutral definitions; the current CLI does
not yet emit progress events, enforce those deployment limits or create jobs.

Scalars in generation/ranking requests remain explicit (no hidden defaults);
constraint arrays default to empty. The fixtures check Rust deserialization and
independent Python/JavaScript schema consumers; TypeScript includes negative type
checks. Request integers must be at most 9,007,199,254,740,991 so JavaScript does
not silently round them; schemas and native semantic validation enforce that
bound. TypeScript `number` alone is not a runtime validator.

Validation precedence is JSON shape first, generation schema version, generation
numeric ranges, normalized nonempty input, constraint tokens, required/excluded
conflicts and required letter availability, then usable hints. Ranked requests
apply that same validator before ranking limits and zero-residual restrictions.
Corpus I/O follows semantic validation; a missing corpus must not hide an invalid
request. An already-expired execution control takes precedence over work errors.
Frequency-data availability is checked when the generation input is loaded.
Populated progress/provenance still needs completion before the phase-1 gate
is passed.

## Adapter-owned deployment limits

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
