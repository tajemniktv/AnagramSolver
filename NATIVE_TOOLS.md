# Native CLI and maintenance tools

Build with `cargo build --locked -p anagram-cli --release`. The native CLI
does not require Python. See `contracts/README.md` for JSON solve/generate
requests, validation, limits, progress and result contracts.

Maintenance commands (JSON input/output; `--timeout-ms N` is supported):

- `prepare-dictionary BASE UNIGRAMS NEW_OUTPUT`
- `build-phrases NEW_DATABASE TITLE_SOURCE...`
- `ranker-build WORDNET_DICT [PHRASE_DB]`
- `ranker-train NEW_MODEL.json`
- `ranker-rank MODEL.json`

Data/model outputs are non-overwriting. Training reports use held-out
bag groups rather than evaluating the final model on its training data.
Phrase preparation uses bounded parallel batches and staged publication.
Use `tools/prepare-native-data.ps1` for explicit downloads or offline sources.
Python remains at the repository root as a reference in this PR.
Desktop controls and Python archival are separate follow-up PRs.
