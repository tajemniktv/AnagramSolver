# Historical benchmark provenance

These are retained measurements, not claims that today's executable has the
same timings. Do not rewrite historical hashes to match a newer executable.

- The `native-cli-*.json` registry fingerprint
  `e99e4c3affbd5d8577c83ba93f7ecb3d200025262a1728439a9fd8a82a567735`
  is SHA-256 of the checked-in `registry.json` with Windows CRLF line endings.
  The same content with Git's LF endings hashes to
  `636bca38c068e765b3cdd2415db8e49ed469b852b6d3a09ed0369409ba906a2f`.
  This is a byte-format difference, not a different case registry.
- Both generation reports were rerun during review with the current harness,
  which records the exact Python-source fingerprint. This replaces the original
  synthetic report's missing hash and the real report's unverifiable source
  association with fresh measurements, rather than rewriting historical hashes.
  Three samples remain illustrative, not a statistically robust speed guarantee.
- `native-cli-word-features-shakira.json` contains native-only runs. Its original
  generic methodology description is superseded by the corrected report text;
  no Python measurements are implied by this artifact.
