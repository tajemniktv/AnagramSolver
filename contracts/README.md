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

`candidate_budget: 0` means unlimited generation and will not be accepted by the
future public service. This CLI has no deployment policy; it is local-only.
`kind: generation_only` explicitly distinguishes unranked word bags. `stop` is
`exhausted`, `candidate_cap`, `cancelled`, or `timed_out`; only the first proves
complete enumeration. The extra-bag probe distinguishes exact cap exhaustion
from actual truncation. CLI cancellation/deadline configuration is not exposed
yet; the engine accepts both controls.

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
job/progress contract or end-to-end cancellation yet. Python remains the default
application. Still required before frontend adoption: generated schema/TypeScript
types, capability/job contracts, deployment limits, corpus provenance and complete
parity/performance acceptance gates.
