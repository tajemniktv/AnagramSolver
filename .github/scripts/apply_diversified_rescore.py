from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def patch_topk() -> None:
    path = ROOT / "anagram_rerank_topk_impl.py"
    text = path.read_text(encoding="utf-8")

    if "def _select_phrase_rescore_rows(" in text:
        raise SystemExit("top-k rescore selection already patched")

    text = text.replace(
        "import multiprocessing\nimport sys\n",
        "import math\nimport multiprocessing\nimport sys\n",
        1,
    )

    marker = "\ndef apply_phrase_rescore(\n"
    if marker not in text:
        raise SystemExit("apply_phrase_rescore marker not found")

    helpers = r'''

def _corpus_probe_scores(
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


def _select_phrase_rescore_rows(
    bucket: Sequence[Row],
    *,
    collocation: PositiveBigramModel | None,
    phrase_index: PhraseIndex | None,
    top_per_group: int,
) -> tuple[list[Row], int]:
    """Diversified late-stage shortlist without sacrificing existing winners.

    PRE and FINAL keep their historical full quotas. A third bounded channel
    admits rows whose retained grammatical orders have positive corpus evidence.
    This improves recall for word bags that look mediocre under order-agnostic
    lexical scoring but form a strongly attested phrase in one retained order.
    """
    by_final = sorted(
        bucket,
        key=lambda r: (-r.final, -r.pre_score, r.words),
    )[:top_per_group]
    by_pre = sorted(
        bucket,
        key=lambda r: (-r.pre_score, -r.final, r.words),
    )[:top_per_group]

    baseline_by_id = {id(row): row for row in (*by_final, *by_pre)}
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
    chosen_by_id = {
        id(row): row for row in (*by_final, *by_pre, *by_corpus)
    }
    chosen = sorted(
        chosen_by_id.values(),
        key=lambda r: (-max(r.final, r.pre_score), r.words),
    )
    return chosen, corpus_added
'''
    text = text.replace(marker, helpers + marker, 1)

    old = r'''    rescored = 0
    for bucket in by_wc.values():
        by_final = sorted(
            bucket,
            key=lambda r: (-r.final, -r.pre_score, r.words),
        )[:top_per_group]
        by_pre = sorted(
            bucket,
            key=lambda r: (-r.pre_score, -r.final, r.words),
        )[:top_per_group]

        chosen_by_id = {id(row): row for row in (*by_final, *by_pre)}
        chosen = sorted(
            chosen_by_id.values(),
            key=lambda r: (-max(r.final, r.pre_score), r.words),
        )

        for row in chosen:
'''
    new = r'''    rescored = 0
    for word_count, bucket in sorted(by_wc.items()):
        chosen, corpus_added = _select_phrase_rescore_rows(
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

        for row in chosen:
'''
    if old not in text:
        raise SystemExit("phrase-rescore shortlist block not found")
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")


def add_selection_tests() -> None:
    path = ROOT / "tests" / "test_phrase_rescore_selection.py"
    path.write_text(
        '''from __future__ import annotations

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
''',
        encoding="utf-8",
    )


def expand_full_generalization_cases() -> None:
    path = ROOT / "anagram_benchmarks.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordinary_full = {"water_cold", "machine_quietly", "choice_cost"}
    found: set[str] = set()
    for case in payload["cases"]:
        case_id = str(case.get("id", ""))
        if case_id in ordinary_full:
            case["full"] = True
            found.add(case_id)
    missing = ordinary_full - found
    if missing:
        raise SystemExit(f"benchmark cases not found: {sorted(missing)}")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    patch_topk()
    add_selection_tests()
    expand_full_generalization_cases()


if __name__ == "__main__":
    main()
