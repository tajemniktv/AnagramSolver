# Frozen Python reference

The oracle is commit `c12e3d5519d653fd65c7ffbbb2a1597941fedd2a`, not the current
native implementation's HEAD. `manifest.json` records 29 committed oracle/harness
files and 73 local corpus files. Corpus contents are not included in Git.

Regenerate with `python tools/capture_reference.py` only when intentionally
recapturing evidence. The tool rejects changed oracle sources, records canonical
Git blob identities alongside worktree hashes, and accepts explicit `--phrase-db`
and `--ranker-model` inputs. Empty optional lists mean those inputs were not used
in this baseline; they do not mean optional behavior has passed all gates.

Phase 0 is still in progress: corpus licensing inventory, durable behavior
fixtures, benchmark configuration/results and the complete acceptance report
must be added here before declaring the phase complete. Cold benchmark runs must
say whether only puzzle caches are cold; do not claim cold OS caches or downloads.
