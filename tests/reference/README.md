# Frozen Python reference

The oracle is commit `c12e3d5519d653fd65c7ffbbb2a1597941fedd2a`, not the current
native implementation's HEAD. `manifest.json` records 29 committed oracle/harness
files and 73 local corpus files. Corpus contents are not included in Git.

Regenerate with `python tools/capture_reference.py` only when intentionally
recapturing evidence. The tool rejects changed oracle sources, records canonical
Git blob identities alongside worktree hashes, and accepts explicit `--phrase-db`
and `--ranker-model` inputs. Empty optional lists mean those inputs were not used
in this baseline; they do not mean optional behavior has passed all gates.

## Reproduction and proof boundaries

- `CORPORA.md`: formats, provenance and explicit no-bundle decisions where data
  redistribution permission is unverified.
- `behavior.json`: frozen Python normalization, exhaustive/capped prefix/diverse
  bags (including exact-cap exhaustion), repeated required tokens, hints,
  exclusions, impossible/empty results, frontend validation, synthetic SQLite
  phrase evidence and explicit model/tie outputs. Primitive generation fixtures
  and frontend validation are labeled separately; they are not interchangeable.
  Run `python tools/reference_fixtures.py --check` to check both oracle identity
  and these outputs; omit `--check` only for intentional recapture.
- `user-performance.json` and `stress-attempt.json`: partial default 2,000-cap
  performance run and its 180-second timeout. Not a passing full suite.
- `registry.json`: all 52 ordering cases and all five `normal_user_cli` cases,
  at their actual default/registry semantic settings, with exact cold/warm
  outputs. `python tools/reference_registry.py --check` replays and compares
  them. Only verbose diagnostics are replaced with JSON transport. Associated
  `registry-measurements.json` timings are informational, not an isolated
  performance comparison; use the dedicated performance reports for that.
- `user-performance-bounded.json`: representative actual CLI cold/warm runs.
  Reproduce using `benchmark_user_runs.py --case dog_ball --case phone_charge
  --samples 3 --output tests/reference/user-performance-bounded.json`.
- `stages.json`: separate actual generator and ranker subprocess times from
  `python tools/reference_baseline.py stages`; cap 100, two words-input cases,
  both strategies, three samples. No per-stage memory attribution is claimed.
- `gates.json`: existing Python tests (integration enabled), ordering gate and
  refinement report, plus synthetic phrase/model parity. Reproduce using
  `python tools/reference_baseline.py gates`. Refinement is informational; its
  exit status alone is not a quality threshold.

Install benchmark-only `psutil==7.2.2` into `.codex/temp/reference-runtime` and
add that directory to `PYTHONPATH`. Set `TEMP` and `TMP` to this project's
`.codex/temp` before running the original benchmark directly. The wrapper tools
do that themselves. Run measurements sequentially, not alongside builds/gates.

Cold means **fresh puzzle caches**, not empty corpus or OS caches. In warm
reports, `search.seconds` is cached computation time, not current latency;
top-level `seconds` is observed latency. Peak tree RSS samples can double-count
shared pages. Machine identity is in the manifest and original benchmark report.
Synthetic optional data proves semantics, not real Wikimedia-corpus quality or
trained-model benefit. The final acceptance status remains tracked in PLAN.md.
