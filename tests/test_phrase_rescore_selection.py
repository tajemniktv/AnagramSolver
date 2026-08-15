from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import anagram_rerank_topk_impl as reranker


class _PhraseCounts:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def counts(self, phrases):
        return {phrase: self._counts[phrase] for phrase in phrases if phrase in self._counts}


class _Collocation:
    def __init__(self, winning_order: tuple[str, ...] | None):
        self.winning_order = winning_order

    def score(self, order):
        return (1.0, 1.0) if tuple(order) == self.winning_order else (0.0, 0.0)


def _row(name: str, *, pre: float, final: float):
    return SimpleNamespace(words=(name,), pre_score=pre, final=final)


def _candidate(*words: str):
    return SimpleNamespace(order=tuple(words))


class PhraseRescoreSelectionTests(unittest.TestCase):
    def setUp(self):
        reranker._ORDER_CANDIDATES_BY_ROW_ID.clear()

    def tearDown(self):
        reranker._ORDER_CANDIDATES_BY_ROW_ID.clear()

    def test_whole_phrase_probe_can_admit_row_outside_pre_and_final(self):
        baseline = _row("baseline", pre=0.95, final=0.96)
        attested = _row("attested", pre=0.10, final=0.11)
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(baseline)] = (
            _candidate("unlikely", "wording"),
        )
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(attested)] = (
            _candidate("quiet", "rivers", "flow"),
        )

        chosen, added = reranker._select_phrase_rescore_rows(
            [baseline, attested],
            collocation=None,
            phrase_index=_PhraseCounts({"quiet rivers flow": 3}),
            top_per_group=1,
        )

        self.assertEqual(added, 1)
        self.assertEqual({id(row) for row in chosen}, {id(baseline), id(attested)})

    def test_corpus_absence_does_not_expand_baseline_shortlist(self):
        baseline = _row("baseline", pre=0.95, final=0.96)
        unseen = _row("unseen", pre=0.10, final=0.11)
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(baseline)] = (
            _candidate("ordinary", "sentence"),
        )
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(unseen)] = (
            _candidate("another", "ordinary", "sentence"),
        )

        chosen, added = reranker._select_phrase_rescore_rows(
            [baseline, unseen],
            collocation=None,
            phrase_index=_PhraseCounts({}),
            top_per_group=1,
        )

        self.assertEqual(added, 0)
        self.assertEqual([id(row) for row in chosen], [id(baseline)])

    def test_bigram_probe_works_without_phrase_database(self):
        baseline = _row("baseline", pre=0.95, final=0.96)
        collocated = _row("collocated", pre=0.10, final=0.11)
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(baseline)] = (
            _candidate("weak", "pairing"),
        )
        strong_order = ("natural", "word", "pairing")
        reranker._ORDER_CANDIDATES_BY_ROW_ID[id(collocated)] = (
            SimpleNamespace(order=strong_order),
        )

        chosen, added = reranker._select_phrase_rescore_rows(
            [baseline, collocated],
            collocation=_Collocation(strong_order),
            phrase_index=None,
            top_per_group=1,
        )

        self.assertEqual(added, 1)
        self.assertEqual({id(row) for row in chosen}, {id(baseline), id(collocated)})


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


if __name__ == "__main__":
    unittest.main()
