from __future__ import annotations

import unittest
from unittest.mock import patch

import anagram_rerank as rerank


class _Structure:
    def __init__(self, norm: float, kind: str = "clause") -> None:
        self.norm = norm
        self.valency = 1.0
        self.coverage = 1.0
        self.kind = kind


class OrderingLayerTests(unittest.TestCase):
    def test_rank_orders_is_input_order_independent(self) -> None:
        def tables(words, lex):
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

        def local(order, pair, starts, ends):
            return sum(pair[a][b] for a, b in zip(order, order[1:])) / (len(order) - 1)

        def structure(words, lex):
            if words == ("a", "b", "c"):
                return _Structure(1.0)
            if words == ("a", "c", "b"):
                return _Structure(0.95)
            return _Structure(0.7)

        with (
            patch.object(rerank, "_order_local_tables", tables),
            patch.object(rerank, "_local_raw_indices", local),
            patch.object(rerank, "phrase_structure", structure),
        ):
            rankings = []
            for bag in (("c", "a", "b"), ("a", "b", "c"), ("b", "c", "a")):
                candidates, evaluated = rerank.rank_orders(
                    bag,
                    object(),
                    order_mode="exact",
                    exact_max_words=6,
                    top_k=3,
                )
                self.assertEqual(evaluated, 6)
                rankings.append(tuple(candidate.order for candidate in candidates))

        self.assertEqual(rankings[0], rankings[1])
        self.assertEqual(rankings[1], rankings[2])
        self.assertEqual(rankings[0][0], ("a", "b", "c"))
        self.assertEqual(rankings[0][1], ("a", "c", "b"))

    def test_phrase_evidence_can_choose_a_retained_alternative(self) -> None:
        row = rerank.Row(
            words=("a", "b", "c"),
            word_count=3,
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
        row.v13_pre = 70.0

        grammar_winner = rerank.OrderCandidate(
            ("a", "b", "c"), 2.0, 0.80, 0.90, 1.0, 1.0, "clause", 0.88
        )
        corpus_winner = rerank.OrderCandidate(
            ("a", "c", "b"), 1.9, 0.78, 0.89, 1.0, 1.0, "clause", 0.86
        )
        rerank._ORDER_CANDIDATES_BY_ROW_ID[id(row)] = (
            grammar_winner,
            corpus_winner,
        )

        row.best_order = grammar_winner.order
        row.grammar_raw = grammar_winner.grammar_raw
        row.grammar_norm = grammar_winner.grammar_norm
        row.structure_norm = grammar_winner.structure_norm
        row.valency_norm = grammar_winner.valency_norm
        row.syntax_coverage = grammar_winner.syntax_coverage
        row.phrase_kind = grammar_winner.phrase_kind
        row.final = rerank._order_base_final(row, grammar_winner)
        row.base_final = row.final

        class Collocation:
            def score(self, order):
                if order == corpus_winner.order:
                    return 0.9, 1.0
                return 0.1, 1.0

        rerank.apply_phrase_rescore(
            [row],
            collocation=Collocation(),
            phrase_index=None,
            top_per_group=10,
            bonus_max=5.0,
        )

        self.assertEqual(row.best_order, corpus_winner.order)
        self.assertGreater(row.phrase_bonus, 0.0)

    def test_no_corpus_evidence_preserves_grammar_winner(self) -> None:
        row = rerank.Row(
            words=("a", "b", "c"),
            word_count=3,
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
        row.v13_pre = 70.0

        grammar_winner = rerank.OrderCandidate(
            ("a", "b", "c"), 2.0, 0.80, 0.90, 1.0, 1.0, "clause", 0.88
        )
        alternative = rerank.OrderCandidate(
            ("a", "c", "b"), 2.1, 0.90, 0.95, 1.0, 1.0, "clause", 0.86
        )
        rerank._ORDER_CANDIDATES_BY_ROW_ID[id(row)] = (grammar_winner, alternative)
        row.best_order = grammar_winner.order
        row.grammar_raw = grammar_winner.grammar_raw
        row.grammar_norm = grammar_winner.grammar_norm
        row.structure_norm = grammar_winner.structure_norm
        row.valency_norm = grammar_winner.valency_norm
        row.syntax_coverage = grammar_winner.syntax_coverage
        row.phrase_kind = grammar_winner.phrase_kind
        row.final = rerank._order_base_final(row, grammar_winner)
        row.base_final = row.final

        class EmptyCollocation:
            def score(self, order):
                return 0.0, 0.0

        rerank.apply_phrase_rescore(
            [row],
            collocation=EmptyCollocation(),
            phrase_index=None,
            top_per_group=10,
            bonus_max=5.0,
        )

        self.assertEqual(row.best_order, grammar_winner.order)
        self.assertEqual(row.phrase_bonus, 0.0)

    def test_phrase_shortlist_is_union_of_final_and_pre(self) -> None:
        def make_row(words, pre, final):
            row = rerank.Row(
                words=words,
                word_count=3,
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
            candidate = rerank.OrderCandidate(
                words, 2.0, 0.8, 0.9, 1.0, 1.0, "clause", 0.88
            )
            rerank._ORDER_CANDIDATES_BY_ROW_ID[id(row)] = (candidate,)
            row.best_order = words
            row.grammar_raw = candidate.grammar_raw
            row.grammar_norm = candidate.grammar_norm
            row.structure_norm = candidate.structure_norm
            row.valency_norm = 1.0
            row.syntax_coverage = 1.0
            row.phrase_kind = "clause"
            row.final = final
            row.base_final = final
            return row

        best_final = make_row(("a", "b", "c"), 10.0, 90.0)
        best_pre = make_row(("d", "e", "f"), 99.0, 20.0)
        middle = make_row(("g", "h", "i"), 50.0, 50.0)

        rescored = rerank.apply_phrase_rescore(
            [best_final, best_pre, middle],
            collocation=None,
            phrase_index=None,
            top_per_group=1,
            bonus_max=5.0,
        )

        self.assertEqual(rescored, 2)


if __name__ == "__main__":
    unittest.main()
