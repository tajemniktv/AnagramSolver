from __future__ import annotations

import unittest
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
