from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one match in {path}: {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


topk = ROOT / "anagram_rerank_topk_impl.py"

old_probe = '''def _corpus_probe_scores(
    bucket: Sequence[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
) -> dict[int, float]:
    """Cheap positive-only corpus evidence for phrase-rescore admission.

    Full phrase scoring performs several n-gram lookups per retained order. Doing
    that for every deep row would erase most of the late-stage shortlist's cost
    advantage. This probe instead batches only whole-order phrase lookups and
    combines them with the already in-memory positive bigram model.

    Missing corpus evidence remains exactly neutral: a zero score never removes
    a row selected by PRE or FINAL.
    """
    scores = {id(row): 0.0 for row in bucket}
    phrase_owners: dict[str, set[int]] = defaultdict(set)

    for row in bucket:
        row_id = id(row)
        for candidate in _row_phrase_candidates(row):
            if collocation is not None:
                colloc, _ = collocation.score(candidate.order)
                scores[row_id] = max(scores[row_id], 0.55 * colloc)

            if phrase_index is not None:
                phrase_owners[" ".join(candidate.order)].add(row_id)

    if phrase_index is not None and phrase_owners:
        # PhraseIndex.counts() batches its SQLite queries internally. Any whole
        # retained order attested by the corpus is stronger admission evidence
        # than isolated bigrams, matching PhraseIndex.score()'s hierarchy.
        for phrase, count in phrase_index.counts(tuple(phrase_owners)).items():
            if count <= 0:
                continue
            exact = min(
                1.0,
                0.72 + 0.28 * math.log10(count + 1.0) / 5.0,
            )
            for row_id in phrase_owners.get(phrase, ()):
                scores[row_id] = max(scores[row_id], exact)

    return scores
'''

new_probe = '''def _corpus_probe_scores(
    bucket: Sequence[core.Row],
    *,
    collocation: core.PositiveBigramModel | None,
    phrase_index: core.PhraseIndex | None,
) -> dict[int, float]:
    """Cheap positive-only corpus evidence for phrase-rescore admission.

    Full phrase scoring performs several n-gram lookups per retained order. Doing
    that for every deep row would erase most of the late-stage shortlist's cost
    advantage. This probe therefore uses the grammar winner for the broad
    in-memory bigram check, while whole-phrase database hits may inspect all
    retained orders so a strongly attested alternative can still rescue a bag.

    Missing corpus evidence remains exactly neutral: a zero score never removes
    a row selected by PRE or FINAL.
    """
    if collocation is None and phrase_index is None:
        return {}

    scores = {id(row): 0.0 for row in bucket}
    phrase_owners: dict[str, set[int]] = defaultdict(set)

    for row in bucket:
        row_id = id(row)
        candidates = _row_phrase_candidates(row)

        # Bigram probing is intentionally limited to the grammar winner. The
        # full retained-order collocation scan belongs to the expensive rescore
        # stage after this bounded admission pass.
        if collocation is not None and candidates:
            colloc, _ = collocation.score(candidates[0].order)
            scores[row_id] = max(scores[row_id], 0.55 * colloc)

        # Whole-phrase hits are cheap to batch and much stronger evidence, so
        # allow any retained grammatical order to contribute here.
        if phrase_index is not None:
            for candidate in candidates:
                phrase_owners[" ".join(candidate.order)].add(row_id)

    if phrase_index is not None and phrase_owners:
        # PhraseIndex.counts() batches its SQLite queries internally. Any whole
        # retained order attested by the corpus is stronger admission evidence
        # than isolated bigrams, matching PhraseIndex.score()'s hierarchy.
        for phrase, count in phrase_index.counts(tuple(phrase_owners)).items():
            if count <= 0:
                continue
            exact = min(
                1.0,
                0.72 + 0.28 * math.log10(count + 1.0) / 5.0,
            )
            for row_id in phrase_owners.get(phrase, ()):
                scores[row_id] = max(scores[row_id], exact)

    return scores
'''
replace_once(topk, old_probe, new_probe)

old_select = '''def _select_phrase_rescore_rows(
    bucket: Sequence[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
) -> tuple[list[Row], int]:
'''
new_select = '''def _select_phrase_rescore_rows(
    bucket: Sequence[core.Row],
    *,
    collocation: core.PositiveBigramModel | None,
    phrase_index: core.PhraseIndex | None,
    top_per_group: int,
) -> tuple[list[core.Row], int]:
'''
replace_once(topk, old_select, new_select)

old_quota = '''    baseline_by_id = {id(row): row for row in (*by_final, *by_pre)}
    probe_scores = _corpus_probe_scores(
        bucket,
        collocation=collocation,
        phrase_index=phrase_index,
    )
    by_corpus = [
        row
        for row in sorted(
            bucket,
            key=lambda r: (
                -probe_scores.get(id(r), 0.0),
                -max(r.final, r.pre_score),
                r.words,
            ),
        )
        if probe_scores.get(id(row), 0.0) > 0.0
    ][:top_per_group]

    corpus_added = sum(1 for row in by_corpus if id(row) not in baseline_by_id)
'''
new_quota = '''    baseline_by_id = {id(row): row for row in (*by_final, *by_pre)}
    # Probe only rows that are not already guaranteed a slot via PRE/FINAL.
    # This both reduces work and prevents baseline rows from consuming the
    # bounded corpus-admission quota.
    corpus_pool = [row for row in bucket if id(row) not in baseline_by_id]
    probe_scores = _corpus_probe_scores(
        corpus_pool,
        collocation=collocation,
        phrase_index=phrase_index,
    )
    by_corpus = [
        row
        for row in sorted(
            corpus_pool,
            key=lambda r: (
                -probe_scores.get(id(r), 0.0),
                -max(r.final, r.pre_score),
                r.words,
            ),
        )
        if probe_scores.get(id(row), 0.0) > 0.0
    ][:top_per_group]

    corpus_added = len(by_corpus)
'''
replace_once(topk, old_quota, new_quota)

old_print = '''        chosen, corpus_added = _select_phrase_rescore_rows(
            bucket,
            collocation=collocation,
            phrase_index=phrase_index,
            top_per_group=top_per_group,
        )
        if corpus_added:
            print(
                f"Corpus-probe shortlist {word_count} words: "
                f"added {corpus_added:,} candidate(s) beyond PRE/FINAL."
            )

'''
new_print = '''        chosen, _ = _select_phrase_rescore_rows(
            bucket,
            collocation=collocation,
            phrase_index=phrase_index,
            top_per_group=top_per_group,
        )

'''
replace_once(topk, old_print, new_print)


tests = ROOT / "tests" / "test_phrase_rescore_selection.py"
text = tests.read_text(encoding="utf-8")
text = text.replace(
    "import unittest\nfrom types import SimpleNamespace\n",
    "import unittest\nfrom types import SimpleNamespace\nfrom unittest.mock import patch\n",
    1,
)
insert = '''
    def test_baseline_corpus_hit_does_not_consume_corpus_quota(self):
        baseline = _row("baseline", pre=0.99, final=0.99)
        outside = _row("outside", pre=0.10, final=0.11)
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(baseline)] = (
            _candidate("very", "famous", "baseline"),
        )
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(outside)] = (
            _candidate("useful", "outside", "phrase"),
        )

        chosen, added = reranker._select_phrase_rescore_rows(
            [baseline, outside],
            collocation=None,
            phrase_index=_PhraseCounts(
                {
                    "very famous baseline": 1000,
                    "useful outside phrase": 2,
                }
            ),
            top_per_group=1,
        )

        self.assertEqual(added, 1)
        self.assertEqual({id(row) for row in chosen}, {id(baseline), id(outside)})

    def test_multi_slot_union_deduplicates_and_preserves_final_ordering(self):
        pre_winner = _row("pre", pre=0.99, final=0.50)
        final_winner = _row("final", pre=0.40, final=0.98)
        overlap = _row("overlap", pre=0.90, final=0.89)
        corpus_one = _row("corpus-one", pre=0.30, final=0.25)
        corpus_two = _row("corpus-two", pre=0.20, final=0.19)
        rows = [pre_winner, final_winner, overlap, corpus_one, corpus_two]

        phrases = {
            id(pre_winner): ("known", "pre"),
            id(final_winner): ("known", "final"),
            id(overlap): ("known", "overlap"),
            id(corpus_one): ("corpus", "one"),
            id(corpus_two): ("corpus", "two"),
        }
        for row in rows:
            reranker._ORDER_CANDIDATES_BY_ROW_ID[id(row)] = (
                _candidate(*phrases[id(row)]),
            )

        chosen, added = reranker._select_phrase_rescore_rows(
            rows,
            collocation=None,
            phrase_index=_PhraseCounts(
                {
                    "known overlap": 1000,
                    "corpus one": 10,
                    "corpus two": 5,
                }
            ),
            top_per_group=2,
        )

        self.assertEqual(added, 2)
        self.assertEqual(len(chosen), 5)
        self.assertEqual(len({id(row) for row in chosen}), 5)
        self.assertEqual(
            [row.words[0] for row in chosen],
            ["pre", "final", "overlap", "corpus-one", "corpus-two"],
        )

    def test_no_corpus_signal_skips_candidate_materialization(self):
        row = _row("ordinary", pre=0.5, final=0.5)
        with patch.object(
            reranker,
            "_row_phrase_candidates",
            side_effect=AssertionError("should not materialize candidates"),
        ):
            scores = reranker._corpus_probe_scores(
                [row],
                collocation=None,
                phrase_index=None,
            )
        self.assertEqual(scores, {})

    def test_bigram_probe_checks_only_grammar_winner(self):
        row = _row("alternatives", pre=0.1, final=0.1)
        grammar_winner = ("grammar", "winner")
        alternate = ("strong", "alternate")
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(row)] = (
            _candidate(*grammar_winner),
            _candidate(*alternate),
        )

        scores = reranker._corpus_probe_scores(
            [row],
            collocation=_Collocation(alternate),
            phrase_index=None,
        )

        self.assertEqual(scores[id(row)], 0.0)
'''
marker = "\n\nif __name__ == \"__main__\":\n"
if marker not in text:
    raise RuntimeError("test insertion marker missing")
text = text.replace(marker, "\n" + insert + marker, 1)
tests.write_text(text, encoding="utf-8")
