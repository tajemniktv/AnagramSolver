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

Still required before frontend adoption: generated schema/TypeScript types,
ranked results, capability/job contracts, deployment limits and corpus provenance
in responses. Do not infer those features from this preliminary operation.
