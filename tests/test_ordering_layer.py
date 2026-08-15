from __future__ import annotations

import importlib
import unittest
from itertools import pairwise
from unittest.mock import patch

import anagram_rerank as rerank
import anagram_rerank_core as core


class _Structure:
    def __init__(self, norm: float, kind: str = "clause") -> None:
        self.norm = norm
        self.valency = 1.0
        self.coverage = 1.0
        self.kind = kind


class OrderingLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        rerank._clear_order_side_tables()
        self.addCleanup(rerank._clear_order_side_tables)

    @staticmethod
    def _row(words: tuple[str, ...], *, pre: float = 70.0, final: float | None = None):
        row = rerank.Row(
            words=words,
            word_count=len(words),
            old_rank=1,
            old_pre=1.0,
            lex=0.8,
            fam=0.8,
            old_pair=0.5,
            hint=0.0,
            zavg=5.0,
            zmin=4.0,
            old_pcov=0.0,
            hints=(),
        )
        row.deep = True
        row.wn_coverage = 1.0
        row.v13_pre = pre
        if final is not None:
            row.final = final
            row.base_final = final
        return row

    @staticmethod
    def _candidate(order: tuple[str, ...], *, objective: float = 0.88):
        return rerank.OrderCandidate(
            order, 2.0, 0.80, 0.90, 1.0, 1.0, "clause", objective
        )

    @staticmethod
    def _candidate_table():
        for name in ("_ORDER_CANDIDATES_BY_ROW_ID", "_ORDER_CANDIDATES_BY_INDEX"):
            table = getattr(rerank.impl, name, None)
            if isinstance(table, dict):
                return table
        raise AssertionError("No retained-order side table found")

    @classmethod
    def _install_candidate(cls, row, candidate) -> None:
        cls._candidate_table()[id(row)] = (candidate,)
        # Newer implementations may use list indices; one-row tests use index 0.
        if hasattr(rerank.impl, "_ORDER_CANDIDATES_BY_INDEX"):
            rerank.impl._ORDER_CANDIDATES_BY_INDEX[0] = (candidate,)
        row.best_order = candidate.order
        row.grammar_raw = candidate.grammar_raw
        row.grammar_norm = candidate.grammar_norm
        row.structure_norm = candidate.structure_norm
        row.valency_norm = candidate.valency_norm
        row.syntax_coverage = candidate.syntax_coverage
        row.phrase_kind = candidate.phrase_kind
        row.final = rerank.impl._order_base_final(row, candidate)
        row.base_final = row.final

    def test_rank_orders_is_input_order_independent(self) -> None:
        def tables(words, _lex):
            pref = {
                ("a", "b"): 3.0,
                ("b", "c"): 3.0,
                ("a", "c"): 2.9,
                ("c", "b"): 2.8,
                ("b", "a"): 1.0,
                ("c", "a"): 0.5,
            }
            n = len(words)
            pair = tuple(
                tuple(
                    0.0 if i == j else pref.get((words[i], words[j]), 0.2)
                    for j in range(n)
                )
                for i in range(n)
            )
            return pair, tuple(0.0 for _ in words), tuple(0.0 for _ in words)

        def local(order, pair, _starts, _ends):
            return sum(pair[a][b] for a, b in pairwise(order)) / (len(order) - 1)

        def structure(words, _lex):
            if words == ("a", "b", "c"):
                return _Structure(1.0)
            if words == ("a", "c", "b"):
                return _Structure(0.95)
            return _Structure(0.7)

        with (
            patch.object(rerank.impl, "_order_local_tables", tables),
            patch.object(rerank.impl, "_local_raw_indices", local),
            patch.object(rerank.impl, "phrase_structure", structure),
        ):
            rankings = []
            for bag in (("c", "a", "b"), ("a", "b", "c"), ("b", "c", "a")):
                candidates, evaluated = rerank.impl.rank_orders(
                    bag, object(), order_mode="exact", exact_max_words=6, top_k=3
                )
                self.assertEqual(evaluated, 6)
                rankings.append(tuple(candidate.order for candidate in candidates))

        self.assertEqual(rankings[0], rankings[1])
        self.assertEqual(rankings[1], rankings[2])
        self.assertEqual(rankings[0][:2], (("a", "b", "c"), ("a", "c", "b")))

    def test_best_order_returns_real_structure_for_winner(self) -> None:
        candidate = self._candidate(("a", "b", "c"))
        sentinel = object()
        with (
            patch.object(rerank.impl, "rank_orders", return_value=((candidate,), 7)),
            patch.object(rerank.impl, "phrase_structure", return_value=sentinel) as structure,
        ):
            raw, order, returned_structure, evaluated = rerank.impl.best_order(
                ("c", "b", "a"), object(), order_mode="exact"
            )

        self.assertEqual(raw, candidate.grammar_raw)
        self.assertEqual(order, candidate.order)
        self.assertIs(returned_structure, sentinel)
        self.assertEqual(evaluated, 7)
        structure.assert_called_once_with(candidate.order, unittest.mock.ANY)

    def test_phrase_evidence_can_choose_retained_alternative(self) -> None:
        row = self._row(("a", "b", "c"))
        grammar_winner = self._candidate(("a", "b", "c"), objective=0.88)
        corpus_winner = rerank.OrderCandidate(
            ("a", "c", "b"), 1.9, 0.78, 0.89, 1.0, 1.0, "clause", 0.86
        )
        self._install_candidate(row, grammar_winner)
        table = self._candidate_table()
        table[id(row)] = (grammar_winner, corpus_winner)
        if hasattr(rerank.impl, "_ORDER_CANDIDATES_BY_INDEX"):
            rerank.impl._ORDER_CANDIDATES_BY_INDEX[0] = (grammar_winner, corpus_winner)

        class Collocation:
            def score(self, order: tuple[str, ...]) -> tuple[float, float]:
                return (0.9, 1.0) if order == corpus_winner.order else (0.1, 1.0)

        rerank.apply_phrase_rescore(
            [row], collocation=Collocation(), phrase_index=None,
            top_per_group=10, bonus_max=5.0,
        )

        self.assertEqual(row.best_order, corpus_winner.order)
        self.assertGreater(row.phrase_bonus, 0.0)
        self.assertFalse(table)

    def test_no_corpus_evidence_preserves_grammar_winner(self) -> None:
        row = self._row(("a", "b", "c"))
        grammar_winner = self._candidate(("a", "b", "c"), objective=0.88)
        alternative = rerank.OrderCandidate(
            ("a", "c", "b"), 2.1, 0.90, 0.95, 1.0, 1.0, "clause", 0.86
        )
        self._install_candidate(row, grammar_winner)
        table = self._candidate_table()
        table[id(row)] = (grammar_winner, alternative)
        if hasattr(rerank.impl, "_ORDER_CANDIDATES_BY_INDEX"):
            rerank.impl._ORDER_CANDIDATES_BY_INDEX[0] = (grammar_winner, alternative)

        class EmptyCollocation:
            def score(self, _order: tuple[str, ...]) -> tuple[float, float]:
                return 0.0, 0.0

        rerank.apply_phrase_rescore(
            [row], collocation=EmptyCollocation(), phrase_index=None,
            top_per_group=10, bonus_max=5.0,
        )

        self.assertEqual(row.best_order, grammar_winner.order)
        self.assertEqual(row.phrase_bonus, 0.0)

    def test_phrase_shortlist_is_union_of_final_and_pre(self) -> None:
        best_final = self._row(("a", "b", "c"), pre=10.0, final=90.0)
        best_pre = self._row(("d", "e", "f"), pre=99.0, final=20.0)
        middle = self._row(("g", "h", "i"), pre=50.0, final=50.0)
        rows = [best_final, best_pre, middle]

        table = self._candidate_table()
        for index, row in enumerate(rows):
            candidate = self._candidate(row.words)
            table[id(row)] = (candidate,)
            if hasattr(rerank.impl, "_ORDER_CANDIDATES_BY_INDEX"):
                rerank.impl._ORDER_CANDIDATES_BY_INDEX[index] = (candidate,)
            row.best_order = candidate.order
            row.grammar_raw = candidate.grammar_raw
            row.grammar_norm = candidate.grammar_norm
            row.structure_norm = candidate.structure_norm
            row.valency_norm = candidate.valency_norm
            row.syntax_coverage = candidate.syntax_coverage
            row.phrase_kind = candidate.phrase_kind

        class Collocation:
            def score(self, _order: tuple[str, ...]) -> tuple[float, float]:
                return 0.5, 1.0

        rescored = rerank.apply_phrase_rescore(
            rows, collocation=Collocation(), phrase_index=None,
            top_per_group=1, bonus_max=5.0,
        )

        self.assertEqual(rescored, 2)
        self.assertGreater(best_final.phrase_bonus, 0.0)
        self.assertGreater(best_pre.phrase_bonus, 0.0)
        self.assertEqual(middle.phrase_bonus, 0.0)

    def test_cache_rejects_nonfinite_and_out_of_range_values(self) -> None:
        base = rerank._row_to_cache_dict(self._row(("a", "b", "c")))
        bad_values = (
            ("lex", float("nan")),
            ("hint", float("inf")),
            ("zavg", 1e100),
            ("old_pre", -1.0),
            ("v13_pre", 101.0),
            ("word_count", 0),
            ("old_rank", 0),
        )
        for field, value in bad_values:
            with self.subTest(field=field, value=value):
                item = dict(base)
                item[field] = value
                self.assertIsNone(rerank._row_from_cache_dict(item))

    def test_prepare_rows_uses_captured_core_delegate(self) -> None:
        rows = [self._row(("c", "a", "b")), self._row(("f", "d", "e"))]
        lex = object()
        with patch.object(rerank, "_CORE_PREPARE_ROWS") as delegate:
            # Reproduce main()'s runtime rebinding. A dynamic core.prepare_rows
            # lookup here would recurse back into rerank.prepare_rows.
            with patch.object(core, "prepare_rows", rerank.prepare_rows):
                rerank.prepare_rows(rows, lex)

        delegate.assert_called_once_with(rows, lex)
        self.assertEqual(rows[0].words, ("a", "b", "c"))
        self.assertEqual(rows[1].words, ("d", "e", "f"))

    def test_importing_frontend_does_not_mutate_core(self) -> None:
        originals = (
            core.best_order,
            core.deep_analyze,
            core.apply_phrase_rescore,
            core.DeepResult,
        )
        importlib.reload(rerank)
        self.assertEqual(
            originals,
            (core.best_order, core.deep_analyze, core.apply_phrase_rescore, core.DeepResult),
        )


if __name__ == "__main__":
    unittest.main()
