# Native CLI and maintenance tools

Build with `cargo build --locked -p anagram-cli --release`. The native CLI
does not require Python. See `contracts/README.md` for JSON solve/generate
requests, validation, limits, progress and result contracts.

Maintenance commands (JSON output; `--timeout-ms N` is supported):

- `prepare-dictionary BASE UNIGRAMS NEW_OUTPUT`
- `build-phrases NEW_DATABASE TITLE_SOURCE...`
- `ranker-build WORDNET_DICT [PHRASE_DB] < cases.json`
- `ranker-train NEW_MODEL.json < groups.json`
- `ranker-rank MODEL.json < items.json`

Dictionary and phrase preparation read plain/gzip corpus files, not JSON stdin.
`cases.json` is an array of `{ "answer": "we are home", "acceptable_orders":
["we are home"] }` records; `groups.json` is the dataset-builder output.
`items.json` contains `{key, features, baseline_score}` records (18 features).
Use shell stdin redirection or a pipeline appropriate to your shell.

Data/model outputs are non-overwriting. Training reports use held-out
bag groups rather than evaluating the final model on its training data.
Phrase preparation uses bounded parallel batches and staged publication.
Use `tools/prepare-native-data.ps1` for explicit downloads or offline sources.
Python remains at the repository root as a reference in this PR.
Desktop controls and Python archival are separate follow-up PRs.
