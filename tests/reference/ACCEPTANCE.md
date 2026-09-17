# Reference acceptance record — 2026-09-17

Oracle: `c12e3d5519d653fd65c7ffbbb2a1597941fedd2a`. Data/machine identity:
`manifest.json`. The frozen Python sources were not edited for these captures.

## Verified

- Frozen behavior capture reproduces exactly, including Unicode decomposition,
  repeated required words, alternative clues, excluded words, empty/impossible
  searches, exact-cap versus truncated enumeration and optional synthetic inputs.
- All **258 Python tests**, with CLI integration enabled, pass. This includes
  the existing cache/concurrency/recovery tests; `gates.json` contains the report.
- The **52-case ordering gate** passes: recall@1 0.462, recall@10 0.865,
  recall@50 1.000, MRR 0.598. The cross-bag grammar margin is 0.088.
- Existing refinement A/B ran successfully and remains informational. Synthetic
  phrase/cohesion (400 cases) and learned-feature/model parity (253 cases) pass.
- **24 actual CLI measurements** completed: two registry cases, prefix/diverse,
  three cold/warm pairs each. Cold and warm result arrays match exactly.
- **12 separate generation/ranking measurements** completed at cap 100; see
  `stages.json` for commands, counts, outputs and actual subprocess times.

## Performance findings, not guarantees

At cap 2,000, `dog_ball` cold medians were 9.61 s prefix and 17.82 s diverse;
`phone_charge` was 9.08 s prefix and 20.69 s diverse. Warm medians were
0.41–0.45 s. Diverse generated the expected dog-answer bag in all three runs
where prefix did not, but neither displayed that answer in its top results.
Neither strategy generated the phone-answer bag at this cap. Do not generalize
these two cases into a default-strategy change or a quality claim.

The default eight-case stress attempt did **not** finish. Its first
`actions_words` prefix cold sample took 157.96 s, with sampled peak process-tree
RSS 10,223,185,920 bytes; its warm run took 0.48 s. The second cold sample hit the
180 s timeout. Later cases/strategies were not attempted by that aborted run.
`stress-attempt.json` identifies the failure; missing samples are not successes.

## Reference-freeze gate passed

`registry.json` now captures all 52 ordering cases and all five cases assigned to
the `normal_user_cli` suite. Capture and a fresh `--check` replay both pass,
including actual default/registry budgets, expected displayed phrases and exact
cold/warm rankings. Registry phrase-ordering/refinement/feature-ranker/full case
IDs are all included in the 52-case ordering set; their optional algorithms are
also exercised by the dedicated gates and synthetic fixtures. Performance cases
remain measured workloads, not an assertion that bounded search must recover
their answers.

Together with `behavior.json`, the source/data manifest, full Python test/gate
reports and explicit performance failure record, this satisfies phase 0's pinned,
repeatable reference requirement. It does **not** claim full native CLI quality
or native cache parity, which still belong to phases 2/3. Real Wikimedia-derived
optional data and trained models are not enabled in this baseline; their fixture
provenance and distribution requirements are explicit rather than fabricated.

The stress timeout is now a known, reproducible measurement boundary, not a
reason to retry expensive runs indefinitely or claim a native speedup. Native
performance and full ranked/cache parity remain later gates. Data packaging
permissions remain as documented in `CORPORA.md`; no third-party data is bundled.
