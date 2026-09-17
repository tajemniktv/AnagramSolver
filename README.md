# TajsAnagrams

Desktop anagram solver powered by the native Rust engine, with a shared Svelte UI.
The installed application does not require Python or a separate CLI executable.

- [Desktop app and local builds](apps/desktop/README.md)
- [Native CLI and data/model tools](NATIVE_TOOLS.md)
- [Implementation plan and progress](PLAN.md)
- [Legacy Python reference](archive/README.md)

## Repository layout

- `crates/`: native engine, CLI and shared data/model tooling.
- `apps/`: desktop application and shared UI.
- `contracts/`: generated request/result schemas and types.
- `tools/` and `tests/parity/`: active build, contract and reference-validation tools.
- `archive/`: retired Python solver, benchmark runners and their unit tests.

Python validation utilities remain active; the archived solver is retained only as
a reference. Existing repository corpora stay in `.anagram_data/`; installed app
data and settings are unaffected by the archive move.

Run the legacy unit suite from `archive/` with
`python -m unittest discover -s tests -v`. Native parity scripts still run from
the repository root and explicitly import the archived reference.
